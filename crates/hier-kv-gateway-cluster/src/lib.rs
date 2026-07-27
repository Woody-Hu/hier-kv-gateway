//! Cluster communication layer for Hier KV Gateway.
//!
//! This crate implements the cluster communication capability between groups of
//! Hier KV Gateway gateway instances, mainly including:
//!
//! - [`transport`]: defines the cluster transport abstraction [`ClusterTransport`], so the upper layer can swap in different transport implementations (default Gossip Bus, can be replaced with NATS, gRPC mesh, etc.).
//! - [`gossip`]: Gossip protocol implementation [`GossipEngine`],
//!   maintains the member list, heartbeat probing, and cross-instance metadata digest synchronization.
//! - [`member`]: cluster member list [`MemberList`] and member state machine [`MemberStatus`].
//! - [`ckf_relay`]: [`KvRelay`], publishes the local Cuckoo Filter projection to the
//!   cross-Region Gossip Bus via barrier snapshot + sequenced delta.
//! - [`messages`]: cluster message type [`ClusterMessage`] and metadata digest / entry definitions.

pub mod transport;
pub mod gossip;
pub mod member;
pub mod ckf_relay;
pub mod messages;
pub mod shared_state;
