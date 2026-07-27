//! Routing history: maintains session-to-affinity backend mappings with TTL-based expiry.
//!
//! The read path uses sharded locks via [`dashmap::DashMap`]; the write path is
//! also free of global locks. Expiry cleanup requires external periodic calls
//! to [`RoutingHistory::evict_expired`].

use std::time::{SystemTime, UNIX_EPOCH};

use hier_kv_gateway_core::ids::{BackendId, SessionId};
use dashmap::DashMap;

/// A single session affinity record.
#[derive(Debug, Clone)]
pub struct SessionAffinity {
    /// Last bound backend.
    pub backend: BackendId,
    /// Last used time (Unix seconds).
    pub last_used_unix: u64,
    /// KV overlap length between the backend and the request at routing time.
    pub kv_overlap_at_route: u32,
}

/// Routing history table.
#[derive(Default)]
pub struct RoutingHistory {
    sessions: DashMap<SessionId, SessionAffinity>,
}

impl RoutingHistory {
    /// Create an empty routing history table.
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// Read a session's affinity record.
    pub fn get(&self, session: &SessionId) -> Option<SessionAffinity> {
        self.sessions.get(session).map(|r| r.value().clone())
    }

    /// Write or update a session's affinity record (auto-updates `last_used_unix`).
    pub fn set(
        &self,
        session: SessionId,
        backend: BackendId,
        kv_overlap_at_route: u32,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.sessions.insert(
            session,
            SessionAffinity {
                backend,
                last_used_unix: now,
                kv_overlap_at_route,
            },
        );
    }

    /// Remove a session affinity record.
    pub fn remove(&self, session: &SessionId) {
        self.sessions.remove(session);
    }

    /// Evict all records where `last_used_unix + ttl < now`; returns the number evicted.
    pub fn evict_expired(&self, ttl: std::time::Duration) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let ttl_secs = ttl.as_secs();
        let mut removed = 0usize;
        self.sessions.retain(|_, aff| {
            if aff.last_used_unix.saturating_add(ttl_secs) < now {
                removed += 1;
                false
            } else {
                true
            }
        });
        removed
    }

    /// Current number of sessions.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

impl std::fmt::Debug for RoutingHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingHistory")
            .field("sessions", &self.sessions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn backend(n: u8) -> BackendId {
        BackendId::new(format!("r{n}"), format!("i{n}"))
    }

    #[test]
    fn set_and_get() {
        let h = RoutingHistory::new();
        let s = SessionId::new("sess-1");
        let b = backend(7);
        h.set(s.clone(), b.clone(), 5);
        let aff = h.get(&s).expect("should exist");
        assert_eq!(aff.backend, b);
        assert_eq!(aff.kv_overlap_at_route, 5);
    }

    #[test]
    fn evict_expired_removes_old_entries() {
        let h = RoutingHistory::new();
        let s = SessionId::new("sess-1");
        // Manually insert an old entry with last_used_unix = 0
        h.sessions.insert(
            s.clone(),
            SessionAffinity {
                backend: backend(1),
                last_used_unix: 0,
                kv_overlap_at_route: 0,
            },
        );
        let removed = h.evict_expired(Duration::from_secs(1));
        assert!(removed >= 1);
        assert!(h.get(&s).is_none());
    }
}
