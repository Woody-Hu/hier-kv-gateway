//! Decision telemetry sinks.
//!
//! The gateway emits one [`DecisionEvent`] per request (see
//! [`crate::handlers::chat_completions`]). This module provides the concrete
//! [`DecisionEventSink`] implementations selected by
//! [`TelemetryConfig`](hier_kv_gateway_core::config::TelemetryConfig):
//!
//! * [`RingBufferSink`] — always-on in-memory ring buffer backing the
//!   `GET /admin/decision_events` endpoint; capacity is configurable and the
//!   buffer can be disabled with `buffer_size = 0`.
//! * [`TracingSink`] — emits each event as a structured `tracing` record on
//!   the `hier_kv_gateway.decision_events` target, so the existing log
//!   pipeline (Loki / ELK / journald) can carry the decision stream without
//!   any new transport.
//! * [`NdjsonFileSink`] — appends events to a local NDJSON file from a
//!   background writer task; `emit` is a non-blocking channel send, so the
//!   request hot path never touches the filesystem.
//!
//! [`build_telemetry`] assembles the configured sinks into a single fan-out
//! sink plus the shared ring-buffer handle the admin endpoint reads from.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::io::AsyncWriteExt;
use tracing::{error, warn};

use hier_kv_gateway_core::config::{TelemetryConfig, TelemetryMode};
use hier_kv_gateway_core::decision_event::{DecisionEvent, DecisionEventSink, MultiSink};

/// `tracing` target used by [`TracingSink`]; filter with
/// `RUST_LOG=hier_kv_gateway.decision_events=info`.
pub const DECISION_EVENT_TARGET: &str = "hier_kv_gateway.decision_events";

/// Shared in-memory ring buffer of the most recent decision events.
///
/// Cloning is cheap (inner `Arc`); the admin endpoint holds one clone and the
/// [`RingBufferSink`] holds another.
#[derive(Clone, Debug)]
pub struct DecisionEventBuffer {
    inner: Arc<Mutex<VecDeque<DecisionEvent>>>,
    capacity: usize,
}

impl DecisionEventBuffer {
    /// Create a buffer keeping at most `capacity` events (oldest evicted).
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity.min(4096)))),
            capacity,
        }
    }

    /// Push one event, evicting the oldest when at capacity.
    pub fn push(&self, event: DecisionEvent) {
        if self.capacity == 0 {
            return;
        }
        let mut buf = self.inner.lock();
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(event);
    }

    /// Newest-last snapshot of at most `limit` buffered events.
    ///
    /// `limit == 0` (or a limit above the current length) returns everything.
    pub fn snapshot(&self, limit: usize) -> Vec<DecisionEvent> {
        let buf = self.inner.lock();
        let len = buf.len();
        let keep = if limit == 0 { len } else { limit.min(len) };
        buf.iter().skip(len - keep).cloned().collect()
    }

    /// Number of events currently buffered.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Sink feeding the in-memory ring buffer backing `GET /admin/decision_events`.
pub struct RingBufferSink {
    buf: DecisionEventBuffer,
}

impl RingBufferSink {
    /// Create a sink writing into `buf`.
    pub fn new(buf: DecisionEventBuffer) -> Self {
        Self { buf }
    }
}

impl DecisionEventSink for RingBufferSink {
    fn emit(&self, event: &DecisionEvent) {
        self.buf.push(event.clone());
    }
}

/// Sink emitting each event as a structured `tracing` record.
///
/// The event is serialized once to JSON and logged on
/// [`DECISION_EVENT_TARGET`]; downstream log shippers treat it as an opaque
/// JSON payload.
#[derive(Clone, Copy, Debug, Default)]
pub struct TracingSink;

impl DecisionEventSink for TracingSink {
    fn emit(&self, event: &DecisionEvent) {
        match serde_json::to_string(event) {
            Ok(json) => tracing::info!(target: DECISION_EVENT_TARGET, event = %json, "decision event"),
            Err(e) => warn!(error = %e, "failed to serialize decision event for tracing sink"),
        }
    }
}

/// Sink appending events to an NDJSON file via a background writer task.
///
/// `emit` serializes the event and sends it over an unbounded channel; the
/// spawned task owns the file handle and writes one JSON object per line.
/// When the channel is full the send still succeeds (unbounded) — memory
/// growth is bounded in practice by the writer draining every line, and a
/// writer failure is logged once and disables further writes.
pub struct NdjsonFileSink {
    tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl NdjsonFileSink {
    /// Open (create/append) `path` and spawn the background writer task.
    ///
    /// Must be called from within a Tokio runtime. Returns an error when the
    /// file cannot be opened.
    pub async fn spawn(path: &str) -> std::io::Result<Self> {
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let path_owned = path.to_string();
        tokio::spawn(async move {
            let mut writer = tokio::io::BufWriter::new(file);
            let mut disabled = false;
            while let Some(line) = rx.recv().await {
                if disabled {
                    continue;
                }
                if let Err(e) = writer.write_all(line.as_bytes()).await {
                    error!(path = %path_owned, error = %e, "decision-event writer failed; disabling file sink");
                    disabled = true;
                    continue;
                }
                // Flush periodically so the file stays readable by tailing
                // analysis jobs; BufWriter keeps this cheap.
                if let Err(e) = writer.flush().await {
                    error!(path = %path_owned, error = %e, "decision-event writer flush failed; disabling file sink");
                    disabled = true;
                }
            }
            let _ = writer.flush().await;
        });
        Ok(Self { tx })
    }
}

impl DecisionEventSink for NdjsonFileSink {
    fn emit(&self, event: &DecisionEvent) {
        if let Ok(mut line) = serde_json::to_string(event) {
            line.push('\n');
            // Unbounded send only fails when the receiver was dropped
            // (shutdown); drop the event silently in that case.
            let _ = self.tx.send(line);
        }
    }
}

/// Result of [`build_telemetry`]: the fan-out sink plus the admin buffer.
pub struct Telemetry {
    /// The sink handlers should emit decision events into.
    pub sink: Arc<dyn DecisionEventSink>,
    /// In-memory buffer backing `GET /admin/decision_events`; `None` when
    /// `buffer_size = 0`.
    pub buffer: Option<DecisionEventBuffer>,
}

/// Assemble the sink chain from the telemetry configuration.
///
/// The ring buffer sink is included whenever `buffer_size > 0`; the
/// durable/streaming sink follows `mode`. When nothing is enabled the
/// returned sink is a [`hier_kv_gateway_core::decision_event::NoopSink`], so
/// the hot path pays only one virtual call.
pub async fn build_telemetry(cfg: &TelemetryConfig) -> Telemetry {
    let mut sinks: Vec<Box<dyn DecisionEventSink>> = Vec::new();

    let buffer = if cfg.buffer_size > 0 {
        let buf = DecisionEventBuffer::new(cfg.buffer_size);
        sinks.push(Box::new(RingBufferSink::new(buf.clone())));
        Some(buf)
    } else {
        None
    };

    match cfg.mode {
        TelemetryMode::None => {}
        TelemetryMode::Tracing => sinks.push(Box::new(TracingSink)),
        TelemetryMode::File => match NdjsonFileSink::spawn(&cfg.file_path).await {
            Ok(sink) => sinks.push(Box::new(sink)),
            Err(e) => {
                error!(path = %cfg.file_path, error = %e, "failed to open decision-event file; file sink disabled")
            }
        },
    }

    let sink: Arc<dyn DecisionEventSink> = match sinks.len() {
        0 => Arc::new(hier_kv_gateway_core::decision_event::NoopSink),
        1 => Arc::from(sinks.pop().expect("len checked")),
        _ => Arc::new(MultiSink::new(sinks)),
    };

    Telemetry { sink, buffer }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::decision_event::DecisionOutcome;

    fn sample_event(id: &str) -> DecisionEvent {
        DecisionEvent {
            event_id: id.to_string(),
            timestamp_unix_ms: 1_700_000_000_000,
            gateway_instance: "gw-1".to_string(),
            gateway_region: "r1".to_string(),
            request_id: format!("req-{id}"),
            model: "m".to_string(),
            session_id: None,
            strategy: "hybrid".to_string(),
            weights: None,
            candidates: Vec::new(),
            attempts: Vec::new(),
            selected_backend: Some("r1/a".to_string()),
            kv_overlap: 3,
            prompt_blocks: 10,
            routing_latency_us: 100,
            total_latency_us: 900,
            outcome: DecisionOutcome::Success,
        }
    }

    #[test]
    fn ring_buffer_evicts_oldest_at_capacity() {
        let buf = DecisionEventBuffer::new(2);
        buf.push(sample_event("e1"));
        buf.push(sample_event("e2"));
        buf.push(sample_event("e3"));
        let snap = buf.snapshot(0);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].event_id, "e2");
        assert_eq!(snap[1].event_id, "e3");
    }

    #[test]
    fn ring_buffer_snapshot_limit_returns_newest() {
        let buf = DecisionEventBuffer::new(8);
        for i in 0..5 {
            buf.push(sample_event(&format!("e{i}")));
        }
        let snap = buf.snapshot(2);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].event_id, "e3");
        assert_eq!(snap[1].event_id, "e4");
        // Limit beyond length returns everything.
        assert_eq!(buf.snapshot(99).len(), 5);
    }

    #[test]
    fn zero_capacity_buffer_discards() {
        let buf = DecisionEventBuffer::new(0);
        buf.push(sample_event("e1"));
        assert!(buf.is_empty());
    }

    #[test]
    fn ring_buffer_sink_pushes_events() {
        let buf = DecisionEventBuffer::new(4);
        let sink = RingBufferSink::new(buf.clone());
        sink.emit(&sample_event("e1"));
        assert_eq!(buf.len(), 1);
    }

    #[tokio::test]
    async fn ndjson_file_sink_writes_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("decision-events-test-{}.ndjson", uuid::Uuid::new_v4()));
        let path_str = path.to_string_lossy().to_string();

        let sink = NdjsonFileSink::spawn(&path_str).await.unwrap();
        sink.emit(&sample_event("e1"));
        sink.emit(&sample_event("e2"));
        drop(sink); // close the channel so the writer task drains and exits

        // Wait for the writer to flush by polling the file.
        let mut content = String::new();
        for _ in 0..100 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.lines().count() >= 2 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(content.lines().count(), 2, "file content: {content}");
        let first: DecisionEvent = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(first.event_id, "e1");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn build_telemetry_none_mode_with_buffer() {
        let cfg = TelemetryConfig {
            mode: TelemetryMode::None,
            file_path: String::new(),
            buffer_size: 4,
        };
        let t = build_telemetry(&cfg).await;
        assert!(t.buffer.is_some());
        t.sink.emit(&sample_event("e1"));
        assert_eq!(t.buffer.as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn build_telemetry_zero_buffer_disables_endpoint_storage() {
        let cfg = TelemetryConfig {
            mode: TelemetryMode::None,
            file_path: String::new(),
            buffer_size: 0,
        };
        let t = build_telemetry(&cfg).await;
        assert!(t.buffer.is_none());
        // Emitting must not panic with no sinks at all.
        t.sink.emit(&sample_event("e1"));
    }
}
