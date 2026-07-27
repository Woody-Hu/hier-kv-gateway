//! Core identifier types reused across modules in the Hier KV Gateway system.
//!
//! These types uniquely identify regions, backend instances, indexer domains,
//! connection pools, requests, sessions, and workers. Most identifiers are
//! stored as [`Arc<str>`](std::sync::Arc) for low-cost cloning and sharing
//! across async tasks.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Region identifier.
///
/// Internally uses [`Arc<str>`], so cloning and passing across tasks is cheap;
/// also supports comparison based on string content (`Hash`/`Eq`/`Ord`), not on
/// `Arc` pointer addresses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionId(pub Arc<str>);

impl RegionId {
    /// Create a [`RegionId`] from a string literal.
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    /// Get the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for RegionId {
    fn eq(&self, other: &Self) -> bool {
        // Compare string contents, not Arc pointers
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for RegionId {}

impl Hash for RegionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Only hash the string content, keeping Eq / Hash consistent
        self.0.as_ref().hash(state);
    }
}

impl PartialOrd for RegionId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.as_ref().cmp(other.0.as_ref())
    }
}

impl From<String> for RegionId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for RegionId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Region tier.
///
/// Determines the region's position in the three-tier topology of the Hier KV Gateway,
/// which in turn affects routing strategy and cache migration paths.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RegionTier {
    /// Cloud region, often acting as the central scheduling node.
    Cloud,
    /// Edge region, an access point close to users.
    Edge,
    /// Device-tier region, deployed on terminal devices.
    Device,
}

/// Backend instance identifier.
///
/// Uniquely identifies the same backend service process across different replicas,
/// typically represented as a hostname or Pod name.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendInstanceId(pub Arc<str>);

impl BackendInstanceId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for BackendInstanceId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for BackendInstanceId {}

impl Hash for BackendInstanceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for BackendInstanceId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for BackendInstanceId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for BackendInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Backend identifier, composed of a region and an instance.
///
/// `region` describes the [`RegionId`] the backend resides in, and `instance` is
/// the instance identifier within that region. Together they uniquely identify a
/// backend process across the cluster.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendId {
    pub region: RegionId,
    pub instance: BackendInstanceId,
}

impl BackendId {
    pub fn new(region: impl Into<RegionId>, instance: impl Into<BackendInstanceId>) -> Self {
        Self {
            region: region.into(),
            instance: instance.into(),
        }
    }

    /// Parse a `"<region>/<instance>"` string into a [`BackendId`].
    ///
    /// Splits on the *first* `/` only; both parts must be non-empty. This is
    /// the canonical parser for the wire form produced by [`Display`](std::fmt::Display)
    /// — prefer it over ad-hoc `split('/')` reimplementations.
    pub fn parse(s: &str) -> Option<Self> {
        let slash = s.find('/')?;
        let region = &s[..slash];
        let instance = &s[slash + 1..];
        if region.is_empty() || instance.is_empty() {
            return None;
        }
        Some(Self::new(region, instance))
    }
}

impl std::str::FromStr for BackendId {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::parse(s)
            .ok_or_else(|| format!("invalid backend id {s:?}, expected '<region>/<instance>'"))
    }
}

impl PartialEq for BackendId {
    fn eq(&self, other: &Self) -> bool {
        self.region == other.region && self.instance == other.instance
    }
}

impl Eq for BackendId {}

impl Hash for BackendId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.region.hash(state);
        self.instance.hash(state);
    }
}

impl PartialOrd for BackendId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BackendId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.region
            .cmp(&other.region)
            .then_with(|| self.instance.as_str().cmp(other.instance.as_str()))
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.region, self.instance)
    }
}

/// Indexer domain identifier.
///
/// Indexers are partitioned by domain; multiple backends can share the same
/// indexer domain and thus share a KV cache index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexerDomainId(pub u64);

impl IndexerDomainId {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
}

impl From<u64> for IndexerDomainId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Connection pool identifier, composed of an indexer domain and a region.
///
/// Backends within the same pool share an indexer domain and are typically located
/// in the same region to enable locality-aware routing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolId {
    pub domain: IndexerDomainId,
    pub region: RegionId,
}

impl PartialEq for PoolId {
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain && self.region == other.region
    }
}

impl Eq for PoolId {}

impl Hash for PoolId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.domain.hash(state);
        self.region.hash(state);
    }
}

/// Gateway instance identifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceId(pub Arc<str>);

impl InstanceId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for InstanceId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for InstanceId {}

impl Hash for InstanceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for InstanceId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for InstanceId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Request identifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestId(pub Arc<str>);

impl RequestId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for RequestId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for RequestId {}

impl Hash for RequestId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Session identifier.
///
/// Multiple requests within the same session can be affinity-routed to hit the
/// same backend's KV cache.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionId(pub Arc<str>);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(Arc::from(s.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for SessionId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for SessionId {}

impl Hash for SessionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Worker identifier and its data-parallel rank.
///
/// `worker_id` identifies the worker globally; `dp_rank` is the worker's rank in
/// its data-parallel group. `dp_rank = 0` typically means DP is not enabled or
/// this is the first rank.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerWithRank {
    pub worker_id: u64,
    pub dp_rank: u32,
}

impl WorkerWithRank {
    pub fn new(worker_id: u64, dp_rank: u32) -> Self {
        Self { worker_id, dp_rank }
    }

    /// Construct from worker_id only; dp_rank defaults to 0.
    pub fn from_worker_id(worker_id: u64) -> Self {
        Self {
            worker_id,
            dp_rank: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_id_eq_ignores_arc_pointer() {
        // Two RegionIds with identical content but non-shared Arcs should be equal
        let a = RegionId::new("us-east-1");
        let b = RegionId::new("us-east-1");
        // Force independent allocation to avoid Arc::from reusing a singleton
        let c = RegionId(Arc::from(String::from("us-east-1").as_str()));
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn region_id_hash_consistent_with_eq() {
        let mut set = std::collections::HashSet::new();
        set.insert(RegionId::new("eu-west-1"));
        // Different Arc but same content should hit the same bucket
        assert!(set.contains(&RegionId(Arc::from(
            String::from("eu-west-1").as_str()
        ))));
    }

    #[test]
    fn region_id_ord_lexicographic() {
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        assert!(a < b);
    }

    #[test]
    fn backend_id_round_trip_json() {
        let id = BackendId::new("us-east-1", "worker-0");
        let s = serde_json::to_string(&id).unwrap();
        let back: BackendId = serde_json::from_str(&s).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn backend_id_ord_compound() {
        // Compare by region first, then by instance
        let a = BackendId::new("r1", "i1");
        let b = BackendId::new("r1", "i2");
        let c = BackendId::new("r2", "i1");
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn worker_with_rank_order() {
        let a = WorkerWithRank::new(1, 0);
        let b = WorkerWithRank::new(1, 1);
        let c = WorkerWithRank::new(2, 0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn backend_id_parse_round_trip() {
        let id = BackendId::new("us-east-1", "worker-0");
        let parsed = BackendId::parse(&id.to_string()).unwrap();
        assert_eq!(id, parsed);
        assert_eq!(parsed.region.as_str(), "us-east-1");
        assert_eq!(parsed.instance.as_str(), "worker-0");
        // FromStr agrees with parse
        let via_from_str: BackendId = "us-east-1/worker-0".parse().unwrap();
        assert_eq!(id, via_from_str);
    }

    #[test]
    fn backend_id_parse_rejects_malformed() {
        assert!(BackendId::parse("no-slash").is_none());
        assert!(BackendId::parse("/empty-region").is_none());
        assert!(BackendId::parse("empty-instance/").is_none());
        assert!(BackendId::parse("").is_none());
        assert!("no-slash".parse::<BackendId>().is_err());
        // Splits on the first slash only
        let multi = BackendId::parse("r1/a/b").unwrap();
        assert_eq!(multi.instance.as_str(), "a/b");
    }
}
