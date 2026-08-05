//! Request coalescing (single-flight) configuration.
//!
//! This module holds only the *data* half of request coalescing — the
//! [`CoalescingConfig`] section. The *behaviour* half lives in the API crate
//! (`hier_kv_gateway_api::coalescer::RequestCoalescer`) so the core stays free
//! of tokio / dashmap dependencies.
//!
//! ## What it does
//!
//! When `[coalescing] enabled = true`, the gateway collapses a burst of
//! concurrent *identical* non-streaming requests into a single downstream
//! forward: the first request becomes the "leader" and executes the forward;
//! concurrent waiters attach to the leader's in-flight future and all observe
//! the same response. This is the classic *single-flight* pattern (Go's
//! `golang.org/x/sync/singleflight`, Rust's `tower::buffer` / `dupe` crates,
//! Caddy's `supercache`).
//!
//! Open-source precedent:
//! - **LiteLLM**'s `cache` mechanism (Redis / in-memory) deduplicates by a
//!   request hash, but it is a *response cache* (persists across requests)
//!   rather than in-flight dedup. We mirror only the in-flight dedup here.
//! - **vLLM**'s scheduler merges identical sequences inside a continuous
//!   batch (`--enable-prefix-caching` + sequence dedup) — same idea, one
//!   layer down.
//! - **Envoy**'s `request mirroring` + `cache` and **nginx**'s
//!   `proxy_cache` `lock` option (`proxy_cache_lock on`) implement the same
//!   "one request populates, others wait" semantics at the proxy layer.
//!
//! ## Scope and honesty
//!
//! Coalescing is applied **only to non-streaming requests**. Streaming
//! responses (SSE) cannot be transparently shared without buffering and
//! replaying the whole stream to late joiners, which changes latency
//! characteristics and risks unbounded memory under long generations. We
//! therefore do *not* coalesce streaming requests — each gets its own
//! forward — and document this explicitly so benchmarks are not gamed.
//!
//! Coalesced waiters receive the leader's response verbatim, including the
//! leader's `request_id` and routing-decision headers. This is the standard
//! single-flight tradeoff (cf. Go `singleflight` returning the leader's
//! result) and is reflected in the `X-Hier-KV-Gateway-Coalesced: waiter`
//! response header.

use serde::{Deserialize, Serialize};

/// Coalescing configuration section (`[coalescing]` in TOML).
///
/// All fields carry defaults so existing configurations keep parsing unchanged
/// (`enabled = false` ⇒ no coalescing, every request gets its own forward).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CoalescingConfig {
    /// Master switch. When `false` every request is forwarded independently.
    pub enabled: bool,
    /// Time-to-live (milliseconds) for an in-flight entry after the leader's
    /// forward completes. Within this window, late-arriving identical requests
    /// are still served the cached leader response (cheap dedup) rather than
    /// triggering a fresh forward. `0` disables the post-completion cache:
    /// the entry is dropped the moment the leader finishes, so only truly
    /// concurrent (in-flight) requests are coalesced.
    pub ttl_ms: u64,
    /// Soft cap on the number of simultaneously in-flight coalesced entries.
    /// When exceeded, new requests bypass coalescing and forward directly
    /// (the coalescer degrades to a no-op rather than blocking or OOMing).
    /// `0` means unbounded.
    pub max_inflight: usize,
}

impl Default for CoalescingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // 50 ms post-completion window: long enough to catch a burst of
            // near-simultaneous retries, short enough that a genuinely later
            // request re-runs the forward (so the response reflects current
            // backend state, not a stale leader result).
            ttl_ms: 50,
            max_inflight: 1024,
        }
    }
}

impl CoalescingConfig {
    /// Whether the coalescer should be constructed at all. `false` lets the
    /// API layer skip the `DashMap` allocation entirely.
    pub fn active(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        let c = CoalescingConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.ttl_ms, 50);
        assert_eq!(c.max_inflight, 1024);
        assert!(!c.active());
    }

    #[test]
    fn parses_explicit_values() {
        let toml_text = r#"
enabled = true
ttl_ms = 100
max_inflight = 256
"#;
        let c: CoalescingConfig = toml::from_str(toml_text).unwrap();
        assert!(c.enabled);
        assert_eq!(c.ttl_ms, 100);
        assert_eq!(c.max_inflight, 256);
        assert!(c.active());
    }

    #[test]
    fn absent_section_uses_default() {
        // A config with no [coalescing] section must deserialize to the default.
        let c: CoalescingConfig = toml::from_str("").unwrap();
        assert!(!c.enabled);
    }
}
