//! Cluster transport abstraction.
//!
//! [`ClusterTransport`] is the transport layer trait of hier-kv-gateway-cluster: it
//! abstracts message send/receive, broadcast, and member view capabilities, so the
//! upper layer can swap in different transport implementations (default Gossip Bus,
//! can be replaced with NATS, gRPC mesh, QUIC mesh, etc.).
//!
//! Design points:
//! - `start` receives self identity (`InstanceId` / `RegionId` / bound address), and begins sending and receiving messages once complete.
//! - `messages` returns a [`tokio::sync::mpsc::Receiver`]; the implementation pushes all
//!   [`ClusterMessage`]s received from peers into it. The upper layer (e.g. [`crate::gossip::GossipEngine`])
//!   holds the receiver and processes it in the event loop.
//! - `members` returns the current member list known to the transport layer (usually the
//!   implementation maintains a view consistent with [`crate::member::MemberList`] internally,
//!   or delegates to upstream member management).

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::{InstanceId, RegionId};

use crate::member::ClusterMember;
use crate::messages::ClusterMessage;

/// Cluster transport layer abstraction.
///
/// All methods are designed to be non-blocking (async), allowing the implementation to use
/// any async runtime and IO model internally.
#[async_trait]
pub trait ClusterTransport: Send + Sync {
    /// Start the transport layer.
    ///
    /// `self_id` is this instance's identifier, `region` is the Region it resides in, and
    /// `addr` is the bound address (e.g. `0.0.0.0:7946`). After a successful start, the
    /// transport layer should begin listening for inbound messages and push received
    /// messages through the channel returned by [`messages`](ClusterTransport::messages).
    async fn start(&self, self_id: &InstanceId, region: &RegionId, addr: &str) -> Result<()>;

    /// Stop the transport layer, closing the listener and all connections.
    async fn stop(&self) -> Result<()>;

    /// Send a message to the specified target address.
    ///
    /// `target` is in the form `host:port`. The implementation must guarantee atomic delivery
    /// of a single message (or report a failure).
    async fn send(&self, target: &str, msg: &ClusterMessage) -> Result<()>;

    /// Broadcast a message to all currently known alive members.
    async fn broadcast(&self, msg: &ClusterMessage) -> Result<()>;

    /// Returns the receiver for inbound messages.
    ///
    /// Typically the implementation spawns a task internally after
    /// [`start`](ClusterTransport::start) that reads messages from the underlying IO and
    /// pushes them to this channel.
    ///
    /// Note: calling this method takes ownership of the receiver; multiple calls usually
    /// return the same receiver or a newly created one (depending on the implementation).
    fn messages(&self) -> tokio::sync::mpsc::Receiver<ClusterMessage>;

    /// Returns a snapshot of the currently known member list.
    ///
    /// Usually the implementation delegates to a shared [`crate::member::MemberList`].
    fn members(&self) -> Vec<ClusterMember>;
}
