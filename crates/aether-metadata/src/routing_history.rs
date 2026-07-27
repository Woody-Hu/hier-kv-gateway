//! 路由历史：维护会话到亲和 backend 的映射，支持 TTL 自动过期。
//!
//! 读路径通过 [`dashmap::DashMap`] 分片锁；写路径同样无全局锁。
//! 过期清理需要外部周期性调用 [`RoutingHistory::evict_expired`]。

use std::time::{SystemTime, UNIX_EPOCH};

use aether_core::ids::{BackendId, SessionId};
use dashmap::DashMap;

/// 单条会话亲和记录。
#[derive(Debug, Clone)]
pub struct SessionAffinity {
    /// 上次绑定的 backend。
    pub backend: BackendId,
    /// 上次使用时间（Unix 秒）。
    pub last_used_unix: u64,
    /// 路由决策时该 backend 与请求的 KV 重叠长度。
    pub kv_overlap_at_route: u32,
}

/// 路由历史表。
#[derive(Default)]
pub struct RoutingHistory {
    sessions: DashMap<SessionId, SessionAffinity>,
}

impl RoutingHistory {
    /// 创建一个空的路由历史表。
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// 读取会话的亲和记录。
    pub fn get(&self, session: &SessionId) -> Option<SessionAffinity> {
        self.sessions.get(session).map(|r| r.value().clone())
    }

    /// 写入或更新会话的亲和记录（自动更新 `last_used_unix`）。
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

    /// 移除某条会话亲和。
    pub fn remove(&self, session: &SessionId) {
        self.sessions.remove(session);
    }

    /// 清理所有 `last_used_unix + ttl < now` 的记录，返回清理数量。
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

    /// 当前会话数量。
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// 是否为空。
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
        // 手动塞入一个 last_used_unix = 0 的旧条目
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
