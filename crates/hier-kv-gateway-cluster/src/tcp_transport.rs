//! Default TCP-based [`ClusterTransport`] implementation.
//!
//! Wire format: length-prefixed JSON — a 4-byte big-endian length header
//! followed by the JSON-serialized [`ClusterMessage`] payload. Each message
//! is sent on its own short-lived TCP connection (request/response style);
//! this is intentionally simple and is sufficient for the gossip fanout
//! volumes the rest of the crate generates (a handful of small messages per
//! `gossip_interval_ms` tick).
//!
//! Inbound handling: [`start`](TcpClusterTransport::start) binds a
//! [`tokio::net::TcpListener`] and spawns one task per accepted connection.
//! Each task reads a single framed message and forwards it to an mpsc
//! channel; the receiver is handed out via
//! [`messages`](TcpClusterTransport::messages) for the
//! [`GossipEngine`](crate::gossip::GossipEngine) message loop to consume.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{debug, warn};

use hier_kv_gateway_core::error::{HierKvGatewayError, Result};
use hier_kv_gateway_core::ids::{InstanceId, RegionId};

use crate::member::{ClusterMember, MemberList};
use crate::messages::ClusterMessage;
use crate::transport::ClusterTransport;

/// Maximum allowed frame size (4 MiB). Generous enough for large
/// `CkfBarrierSnapshot` messages while still bounding malicious / corrupted
/// length headers.
const MAX_FRAME_SIZE: usize = 4 * 1024 * 1024;

/// TCP-based cluster transport.
///
/// See the module docs for the wire format and concurrency model.
pub struct TcpClusterTransport {
    /// Shared member list — used for `broadcast` and `members`.
    members: Arc<MemberList>,
    /// Sender half of the inbound message channel. Wrapped in a Mutex so
    /// multiple listener tasks can push messages concurrently.
    inbound_tx: Mutex<Option<mpsc::Sender<ClusterMessage>>>,
    /// Receiver half of the inbound message channel. Handed out exactly once
    /// via [`messages`](Self::messages).
    inbound_rx: Mutex<Option<mpsc::Receiver<ClusterMessage>>>,
    /// Shutdown flag shared with listener tasks.
    running: Arc<AtomicBool>,
    /// Own identity, populated by `start`.
    self_id: Mutex<Option<InstanceId>>,
    /// Listener task handle, stored so `stop` can detach cleanly.
    listener_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl TcpClusterTransport {
    /// Construct a transport backed by the given shared [`MemberList`].
    ///
    /// The transport does not bind any ports until
    /// [`start`](ClusterTransport::start) is called.
    pub fn new(members: Arc<MemberList>) -> Self {
        Self {
            members,
            inbound_tx: Mutex::new(None),
            inbound_rx: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            self_id: Mutex::new(None),
            listener_handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl ClusterTransport for TcpClusterTransport {
    async fn start(
        &self,
        self_id: &InstanceId,
        _region: &RegionId,
        addr: &str,
    ) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Ok(()); // already started
        }

        *self.self_id.lock() = Some(self_id.clone());

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!(
                "TcpClusterTransport failed to bind {}: {}",
                addr, e
            )))?;

        let (tx, rx) = mpsc::channel::<ClusterMessage>(1024);
        *self.inbound_tx.lock() = Some(tx);
        *self.inbound_rx.lock() = Some(rx);

        let running = self.running.clone();
        let inbound_tx_clone = self.inbound_tx.lock().clone();
        let inbound_tx = match inbound_tx_clone {
            Some(t) => t,
            None => {
                return Err(HierKvGatewayError::Internal(
                    "inbound_tx disappeared during start()".to_string(),
                ))
            }
        };

        let handle = tokio::spawn(async move {
            debug!("TcpClusterTransport listener started");
            loop {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // Accept in a select-like loop so we can react to shutdown.
                let accept_result = tokio::select! {
                    biased;
                    _ = tokio::signal::ctrl_c() => break,
                    r = listener.accept() => r,
                };
                let (stream, peer_addr) = match accept_result {
                    Ok(s) => s,
                    Err(e) => {
                        warn!(error = %e, "TcpClusterTransport accept failed");
                        continue;
                    }
                };
                let tx = inbound_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound(stream, tx).await {
                        debug!(peer = %peer_addr, error = %e, "inbound connection failed");
                    }
                });
            }
            debug!("TcpClusterTransport listener exited");
        });

        *self.listener_handle.lock() = Some(handle);
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        if !self.running.swap(false, Ordering::SeqCst) {
            return Ok(()); // not running
        }
        // Drop the sender to unblock any pending sends in listener tasks.
        *self.inbound_tx.lock() = None;
        // Detach the listener task — it will observe `running = false` and exit.
        if let Some(handle) = self.listener_handle.lock().take() {
            handle.abort();
        }
        Ok(())
    }

    async fn send(&self, target: &str, msg: &ClusterMessage) -> Result<()> {
        let mut stream = TcpStream::connect(target)
            .await
            .map_err(|e| HierKvGatewayError::Internal(format!(
                "TcpClusterTransport: failed to connect to {}: {}",
                target, e
            )))?;
        write_frame(&mut stream, msg).await?;
        Ok(())
    }

    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()> {
        let self_id = self.self_id.lock().clone();
        let targets: Vec<String> = self
            .members
            .alive_members()
            .into_iter()
            .filter(|m| {
                if let Some(ref self_id) = self_id {
                    m.instance_id != *self_id
                } else {
                    true
                }
            })
            .map(|m| m.addr)
            .collect();
        for addr in targets {
            // Best-effort: log failures but don't abort the whole broadcast.
            if let Err(e) = self.send(&addr, msg).await {
                debug!(target = %addr, error = %e, "broadcast send failed");
            }
        }
        Ok(())
    }

    fn messages(&self) -> mpsc::Receiver<ClusterMessage> {
        self.inbound_rx
            .lock()
            .take()
            .unwrap_or_else(|| mpsc::channel(1).1)
    }

    fn members(&self) -> Vec<ClusterMember> {
        self.members.all_members()
    }
}

impl Drop for TcpClusterTransport {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

/// Read exactly one framed [`ClusterMessage`] from `stream` and forward it
/// to `tx`. Returns an error on malformed framing or premature EOF.
async fn handle_inbound(
    mut stream: TcpStream,
    tx: mpsc::Sender<ClusterMessage>,
) -> Result<()> {
    let msg = read_frame(&mut stream).await?;
    // Push to the inbound channel; if the receiver is gone (engine stopped),
    // the message is silently dropped.
    let _ = tx.send(msg).await;
    Ok(())
}

/// Write a length-prefixed JSON frame to `stream`.
async fn write_frame(stream: &mut TcpStream, msg: &ClusterMessage) -> Result<()> {
    let payload = serde_json::to_vec(msg).map_err(|e| {
        HierKvGatewayError::Internal(format!("TcpClusterTransport: serialize failed: {}", e))
    })?;
    let len = payload.len() as u32;
    if payload.len() > MAX_FRAME_SIZE {
        return Err(HierKvGatewayError::Internal(format!(
            "TcpClusterTransport: frame too large ({} > {})",
            payload.len(),
            MAX_FRAME_SIZE
        )));
    }
    stream.write_all(&len.to_be_bytes()).await.map_err(|e| {
        HierKvGatewayError::Internal(format!("TcpClusterTransport: write length failed: {}", e))
    })?;
    stream.write_all(&payload).await.map_err(|e| {
        HierKvGatewayError::Internal(format!("TcpClusterTransport: write payload failed: {}", e))
    })?;
    stream.flush().await.map_err(|e| {
        HierKvGatewayError::Internal(format!("TcpClusterTransport: flush failed: {}", e))
    })?;
    Ok(())
}

/// Read a length-prefixed JSON frame from `stream` and deserialize it.
async fn read_frame(stream: &mut TcpStream) -> Result<ClusterMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.map_err(|e| {
        HierKvGatewayError::Internal(format!("TcpClusterTransport: read length failed: {}", e))
    })?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 || len > MAX_FRAME_SIZE {
        return Err(HierKvGatewayError::Internal(format!(
            "TcpClusterTransport: invalid frame length {} (max {})",
            len, MAX_FRAME_SIZE
        )));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await.map_err(|e| {
        HierKvGatewayError::Internal(format!("TcpClusterTransport: read payload failed: {}", e))
    })?;
    serde_json::from_slice(&payload).map_err(|e| {
        HierKvGatewayError::Internal(format!(
            "TcpClusterTransport: deserialize failed: {}",
            e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::member::MemberStatus;
    use hier_kv_gateway_core::ids::{InstanceId, RegionId};

    #[tokio::test]
    async fn send_and_receive_round_trip() {
        // Pick a free port by binding to :0, then closing and rebinding.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound_addr = listener.local_addr().unwrap();
        drop(listener);

        let members = Arc::new(MemberList::new());
        let transport = TcpClusterTransport::new(members.clone());
        transport
            .start(&InstanceId::new("self"), &RegionId::new("r1"), &bound_addr.to_string())
            .await
            .unwrap();

        // Seed a peer entry so broadcast has a target.
        members.upsert(ClusterMember {
            instance_id: InstanceId::new("self"),
            region: RegionId::new("r1"),
            addr: bound_addr.to_string(),
            last_pong_unix: 0,
            status: MemberStatus::Alive,
            meta_digest: Default::default(),
        });

        let mut rx = transport.messages();

        let msg = ClusterMessage::Ping {
            sender: InstanceId::new("self"),
            region: RegionId::new("r1"),
            meta_digest: Default::default(),
            timestamp: 42,
        };
        transport
            .send(&bound_addr.to_string(), &msg)
            .await
            .unwrap();

        let received = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for message")
            .expect("channel closed");
        match received {
            ClusterMessage::Ping { timestamp, .. } => assert_eq!(timestamp, 42),
            _ => panic!("expected Ping, got something else"),
        }

        transport.stop().await.unwrap();
    }

    #[tokio::test]
    async fn invalid_frame_length_is_rejected() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound_addr = listener.local_addr().unwrap();
        drop(listener);

        let members = Arc::new(MemberList::new());
        let transport = TcpClusterTransport::new(members);
        transport
            .start(&InstanceId::new("self"), &RegionId::new("r1"), &bound_addr.to_string())
            .await
            .unwrap();
        let mut rx = transport.messages();

        // Send a frame with length 0 (invalid).
        let mut stream = TcpStream::connect(bound_addr).await.unwrap();
        stream.write_all(&0u32.to_be_bytes()).await.unwrap();
        stream.flush().await.unwrap();

        // The reader should error and not produce a message.
        let result = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await;
        assert!(
            result.is_err() || result.unwrap().is_none(),
            "expected no message for invalid frame"
        );

        transport.stop().await.unwrap();
    }
}
