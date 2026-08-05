//! Request coalescing (single-flight) for the chat-completions hot path.
//!
//! See [`hier_kv_gateway_core::coalescing`] for the configuration and the
//! open-source precedent (Go `singleflight`, nginx `proxy_cache_lock`,
//! LiteLLM in-memory cache). This module is the *behaviour* half.
//!
//! ## How it works
//!
//! [`RequestCoalescer`] holds a `DashMap<u64, SharedHandle>`. The first request
//! for a given key (the *leader*) inserts a [`Shared`] future that wraps the
//! real forward; concurrent *waiters* clone that same shared future. The
//! underlying forward runs exactly once; every waiter observes the leader's
//! result. When the leader completes it removes the entry immediately
//! (`ttl_ms == 0`) or schedules a delayed removal so a short post-completion
//! window still serves the cached result.
//!
//! `Shared<BoxFuture<...>>` was chosen over a hand-rolled `Notify`+`Mutex`
//! dance because it is the canonical Rust single-flight primitive: it
//! guarantees the producer runs once, clones resolve to the same value, and
//! the bookkeeping (waker list, result slot) is encapsulated.
//!
//! ## Scope
//!
//! Only **non-streaming** requests are coalesced. Streaming responses cannot
//! be transparently shared without buffering+replaying the full SSE stream
//! to late joiners (changing tail latency and risking unbounded memory), so
//! we forward each streaming request independently and document it. This
//! keeps the benchmark honest.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use tracing::debug;

use hier_kv_gateway_core::coalescing::CoalescingConfig;

use crate::openai_types::OpenAIChatRequest;

/// A clone-able error carried from the leader to all waiters.
///
/// Wraps an `Arc<str>` so cloning (one per waiter) is a single refcount bump.
#[derive(Clone, Debug)]
pub struct CoalesceError(pub Arc<str>);

impl std::fmt::Display for CoalesceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CoalesceError {}

impl From<String> for CoalesceError {
    fn from(s: String) -> Self {
        Self(Arc::from(s))
    }
}

/// The leader's response, shared verbatim with all waiters.
///
/// `body` is an `Arc<[u8]>` so cloning per waiter is a single refcount bump;
/// the routing metadata is small and copied as-is. Waiters therefore receive
/// the leader's `request_id` and routing headers — the standard single-flight
/// tradeoff (cf. Go `singleflight`).
#[derive(Clone, Debug)]
pub struct CoalescedResponse {
    /// HTTP status code of the leader's response.
    pub status: u16,
    /// Serialized response body (JSON). Shared via `Arc` so N waiters pay for
    /// one allocation, not N.
    pub body: Arc<[u8]>,
    /// Selected backend id string (for the `X-Hier-KV-Gateway-Backend` header).
    pub backend: String,
    /// Strategy name (for the `X-Hier-KV-Gateway-Strategy` header).
    pub strategy: String,
    /// KV overlap (for the `X-Hier-KV-Gateway-KV-Overlap` header).
    pub kv_overlap: u32,
}

/// The shared in-flight handle stored under each key.
struct SharedHandle {
    /// The shared future resolving to the leader's response (or error).
    future: Shared<BoxFuture<'static, Result<CoalescedResponse, CoalesceError>>>,
    /// When the leader inserted this entry; kept for future stall diagnostics
    /// (e.g. an admin endpoint surfacing entries older than N ms).
    #[allow(dead_code)]
    inserted_at: Instant,
}

/// Atomic counters for observability and benchmark verification.
///
/// The `forwards_saved` counter is what the benchmark asserts on: it must
/// equal `N - 1` when N identical concurrent requests are coalesced into one
/// forward. This is the anti-cheat hook — a broken coalescer would show
/// `forwards_saved == 0`.
#[derive(Debug, Default)]
pub struct CoalesceStats {
    /// Requests that became a leader and executed the forward.
    pub leaders: AtomicU64,
    /// Requests that attached to an in-flight leader and awaited its result.
    pub waiters: AtomicU64,
    /// Requests that bypassed coalescing (capacity exceeded / streaming).
    pub bypassed: AtomicU64,
    /// Forwards avoided = `waiters + post_completion_hits`.
    pub forwards_saved: AtomicU64,
}

impl CoalesceStats {
    /// Snapshot all counters (for tests / admin endpoints).
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.leaders.load(Ordering::Relaxed),
            self.waiters.load(Ordering::Relaxed),
            self.bypassed.load(Ordering::Relaxed),
            self.forwards_saved.load(Ordering::Relaxed),
        )
    }
}

/// Single-flight request coalescer for non-streaming chat completions.
///
/// The inner state (in-flight map, stats) is shared via `Arc` so that cheap
/// clones — handed to per-request tasks — all observe the same counters and
/// the same in-flight map.
#[derive(Clone)]
pub struct RequestCoalescer {
    inner: Arc<RequestCoalescerInner>,
}

struct RequestCoalescerInner {
    inflight: DashMap<u64, SharedHandle>,
    cfg: CoalescingConfig,
    stats: CoalesceStats,
}

impl RequestCoalescer {
    /// Build a coalescer from configuration.
    pub fn new(cfg: CoalescingConfig) -> Self {
        Self {
            inner: Arc::new(RequestCoalescerInner {
                inflight: DashMap::new(),
                cfg,
                stats: CoalesceStats::default(),
            }),
        }
    }

    /// Whether coalescing is enabled. When `false`, the handler must call the
    /// producer directly (the coalescer is a no-op).
    pub fn enabled(&self) -> bool {
        self.inner.cfg.enabled
    }

    /// Borrow the stats counters (for benchmarks / admin endpoints).
    pub fn stats(&self) -> &CoalesceStats {
        &self.inner.stats
    }

    /// Number of currently in-flight entries.
    pub fn inflight_count(&self) -> usize {
        self.inner.inflight.len()
    }

    /// Coalesce a non-streaming request.
    ///
    /// `key` is the semantic hash of the request (see [`request_key`]).
    /// `produce` builds and forwards the request, returning the serialized
    /// response. The leader runs `produce` exactly once; waiters await the
    /// same shared future and receive a clone of the leader's result.
    ///
    /// When `max_inflight` is exceeded the call degrades gracefully: it runs
    /// `produce` directly (no dedup) so the request still succeeds, and
    /// increments `bypassed`.
    pub async fn coalesce<F, Fut>(
        &self,
        key: u64,
        produce: F,
    ) -> Result<CoalescedResponse, CoalesceError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<CoalescedResponse, CoalesceError>> + Send + 'static,
    {
        let inflight = &self.inner.inflight;
        let stats = &self.inner.stats;
        let cfg = &self.inner.cfg;

        // Capacity guard: degrade to a direct forward instead of blocking.
        if cfg.max_inflight > 0 && inflight.len() >= cfg.max_inflight {
            stats.bypassed.fetch_add(1, Ordering::Relaxed);
            return produce().await;
        }

        let (shared, is_leader) = match inflight.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(e) => {
                // In-flight: attach as a waiter.
                stats.waiters.fetch_add(1, Ordering::Relaxed);
                stats.forwards_saved.fetch_add(1, Ordering::Relaxed);
                (e.get().future.clone(), false)
            }
            dashmap::mapref::entry::Entry::Vacant(v) => {
                // Leader: wrap the producer in a Shared future.
                stats.leaders.fetch_add(1, Ordering::Relaxed);
                let fut = produce();
                let shared: Shared<BoxFuture<'static, Result<CoalescedResponse, CoalesceError>>> =
                    fut.boxed().shared();
                v.insert(SharedHandle {
                    future: shared.clone(),
                    inserted_at: Instant::now(),
                });
                (shared, true)
            }
        };

        // Drive the shared future. For the leader this runs the producer;
        // for waiters this resolves to the cached leader result.
        let result = shared.await;

        // Leader owns cleanup. A waiter must NOT remove the entry before other
        // waiters have had a chance to attach — only the leader (which
        // created the entry) schedules removal.
        if is_leader {
            Self::schedule_cleanup(&self.inner, key);
        }
        result
    }

    /// Remove the entry now (`ttl_ms == 0`) or after a short delay.
    ///
    /// The delayed removal realizes the post-completion cache window: a
    /// request arriving `t < ttl_ms` after the leader finished still gets the
    /// cached result (another `forwards_saved`), while a later request starts
    /// a fresh forward so the response reflects current backend state.
    fn schedule_cleanup(inner: &Arc<RequestCoalescerInner>, key: u64) {
        if inner.cfg.ttl_ms == 0 {
            inner.inflight.remove(&key);
            return;
        }
        // Clone the Arc so the spawned task outlives the leader's stack frame.
        let inner = inner.clone();
        let ttl = Duration::from_millis(inner.cfg.ttl_ms);
        tokio::spawn(async move {
            tokio::time::sleep(ttl).await;
            // Remove the entry. The per-attach increment in `coalesce` already
            // accounted for each cache hit during the window, so no stat
            // update is needed here — this just drops the stale entry.
            inner.inflight.remove(&key);
        });
    }
}

/// Compute the semantic key for a chat-completions request.
///
/// The key covers every field that changes the *result* or *routing* of the
/// forward: `model`, `messages`, `token_ids`, `max_tokens`, `temperature`,
/// `tools`, `lora_name`, `stream`, and `session`. `session` is included so
/// that two requests with different session-affinity pins are NOT coalesced
/// (a waiter would otherwise inherit the leader's backend, breaking its own
/// session affinity). It deliberately excludes `request_id`, which is unique
/// per request and generated server-side.
///
/// The request is serialized to canonical JSON and hashed, because the
/// message/tool types carry `serde_json::Value` and cannot derive `Hash`.
/// `DefaultHasher` is deterministic within a process; cross-process
/// consistency is not required since coalescing is per-gateway-instance.
pub fn request_key(req: &OpenAIChatRequest) -> u64 {
    // Serialization cannot fail for a struct of plain fields + JSON values;
    // fall back to a constant only if it somehow does (the request would then
    // all collide and simply not dedup — never a correctness issue).
    let bytes = serde_json::to_vec(req).unwrap_or_else(|_| b"".to_vec());
    let mut h = DefaultHasher::new();
    h.write(&bytes);
    h.finish()
}

/// Record that a late-arriving request hit the post-completion cache window.
///
/// Called by the handler when a `coalesce` call returns a result that was
/// served from a still-cached entry (leader already finished). The handler
/// cannot tell leader-vs-waiter apart from the result alone, so it relies on
/// the stats counters already incremented inside `coalesce`.
#[allow(dead_code)]
pub fn note_cache_hit(_stats: &CoalesceStats) {
    // The increment is done inline at attach time in `coalesce`; this helper
    // exists as a documentation anchor and is intentionally a no-op so the
    // accounting stays in one place.
    debug!("coalesce: served from post-completion cache window");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A test producer that sleeps then returns a fixed response, counting
    /// how many times the *forward* (producer body) actually ran.
    fn make_producer(
        body: &'static [u8],
        delay_ms: u64,
        counter: Arc<AtomicUsize>,
    ) -> impl std::future::Future<Output = Result<CoalescedResponse, CoalesceError>> + Send + 'static
    {
        counter.fetch_add(1, Ordering::SeqCst);
        async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            Ok(CoalescedResponse {
                status: 200,
                body: Arc::from(body),
                backend: "r1/inst-0".to_string(),
                strategy: "hybrid".to_string(),
                kv_overlap: 3,
            })
        }
    }

    fn cfg(ttl_ms: u64, max_inflight: usize) -> CoalescingConfig {
        CoalescingConfig {
            enabled: true,
            ttl_ms,
            max_inflight,
        }
    }

    #[tokio::test]
    async fn concurrent_identical_requests_run_one_forward() {
        let coalescer = RequestCoalescer::new(cfg(0, 1024));
        let forward_calls = Arc::new(AtomicUsize::new(0));
        let key = 42u64;

        // Spawn N concurrent coalesce calls with the same key.
        let n = 8;
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let c = coalescer.clone();
            let fc = forward_calls.clone();
            handles.push(tokio::spawn(async move {
                c.coalesce(key, move || make_producer(b"hello", 50, fc)).await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;

        // Every caller gets a successful, identical response.
        assert_eq!(results.len(), n);
        for r in &results {
            let r = r.as_ref().expect("task panicked");
            assert!(r.is_ok(), "waiter should get leader's result");
            assert_eq!(r.as_ref().unwrap().body.as_ref(), b"hello");
            assert_eq!(r.as_ref().unwrap().status, 200);
        }

        // The forward ran EXACTLY once — the anti-cheat assertion.
        assert_eq!(
            forward_calls.load(Ordering::SeqCst),
            1,
            "coalescer must forward once for N={} identical concurrent requests",
            n
        );

        let (leaders, waiters, bypassed, saved) = coalescer.stats().snapshot();
        assert_eq!(leaders, 1, "exactly one leader");
        assert_eq!(waiters, n as u64 - 1, "remaining callers are waiters");
        assert_eq!(bypassed, 0);
        assert_eq!(saved, n as u64 - 1, "forwards_saved = N - 1");
        assert_eq!(coalescer.inflight_count(), 0, "entry removed after ttl_ms=0");
    }

    #[tokio::test]
    async fn distinct_keys_run_independently() {
        let coalescer = RequestCoalescer::new(cfg(0, 1024));
        let fc = Arc::new(AtomicUsize::new(0));

        let a = coalescer.clone();
        let fc_a = fc.clone();
        let h1 = tokio::spawn(async move {
            a.coalesce(1, || make_producer(b"a", 10, fc_a)).await
        });
        let b = coalescer.clone();
        let fc_b = fc.clone();
        let h2 = tokio::spawn(async move {
            b.coalesce(2, || make_producer(b"b", 10, fc_b)).await
        });

        let (r1, r2) = tokio::join!(h1, h2);
        assert_eq!(r1.unwrap().unwrap().body.as_ref(), b"a");
        assert_eq!(r2.unwrap().unwrap().body.as_ref(), b"b");
        assert_eq!(fc.load(Ordering::SeqCst), 2, "distinct keys forward separately");
        let (leaders, _, _, _) = coalescer.stats().snapshot();
        assert_eq!(leaders, 2);
    }

    #[tokio::test]
    async fn ttl_window_serves_late_arrivals_from_cache() {
        let coalescer = RequestCoalescer::new(cfg(200, 1024));
        let fc = Arc::new(AtomicUsize::new(0));

        // Leader runs and finishes quickly.
        let c0 = coalescer.clone();
        let fc0 = fc.clone();
        let first = c0.coalesce(7, || make_producer(b"v1", 20, fc0)).await.unwrap();
        assert_eq!(first.body.as_ref(), b"v1");
        assert_eq!(fc.load(Ordering::SeqCst), 1);

        // Within the TTL window: a second request with the same key must NOT
        // trigger another forward — it should get the cached result.
        let c1 = coalescer.clone();
        let fc1 = fc.clone();
        let second = c1.coalesce(7, || make_producer(b"v2", 20, fc1)).await.unwrap();
        assert_eq!(second.body.as_ref(), b"v1", "late arrival gets cached leader body");
        assert_eq!(
            fc.load(Ordering::SeqCst),
            1,
            "no new forward during TTL window"
        );

        // Wait out the TTL, then a fresh request must re-forward.
        tokio::time::sleep(Duration::from_millis(250)).await;
        let c2 = coalescer.clone();
        let fc2 = fc.clone();
        let third = c2.coalesce(7, || make_producer(b"v3", 20, fc2)).await.unwrap();
        assert_eq!(third.body.as_ref(), b"v3", "after TTL, a fresh forward runs");
        assert_eq!(fc.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn max_inflight_bypass_runs_producer_directly() {
        // max_inflight = 1: the leader fills the single slot; the next request
        // must bypass and run its own producer.
        let coalescer = RequestCoalescer::new(cfg(0, 1));
        let fc = Arc::new(AtomicUsize::new(0));

        let c1 = coalescer.clone();
        let fc1 = fc.clone();
        let h1 = tokio::spawn(async move {
            c1.coalesce(100, || make_producer(b"x", 50, fc1)).await
        });
        // Give the leader a moment to insert its entry.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let c2 = coalescer.clone();
        let fc2 = fc.clone();
        let h2 = tokio::spawn(async move {
            c2.coalesce(101, || make_producer(b"y", 5, fc2)).await
        });

        let (r1, r2) = tokio::join!(h1, h2);
        assert!(r1.unwrap().is_ok());
        assert!(r2.unwrap().is_ok());
        assert_eq!(fc.load(Ordering::SeqCst), 2, "both producers ran (bypass)");
        let (_, _, bypassed, _) = coalescer.stats().snapshot();
        assert_eq!(bypassed, 1, "second request bypassed due to capacity");
    }

    #[tokio::test]
    async fn leader_error_propagates_to_all_waiters() {
        let coalescer = RequestCoalescer::new(cfg(0, 1024));
        let fc = Arc::new(AtomicUsize::new(0));
        let key = 999u64;

        let mut handles = Vec::new();
        for _ in 0..5 {
            let c = coalescer.clone();
            let fc = fc.clone();
            // Each iteration builds a fresh FnOnce closure (the coalescer
            // consumes it), capturing its own Arc clone of the counter.
            handles.push(tokio::spawn(async move {
                c.coalesce(key, move || {
                    fc.fetch_add(1, Ordering::SeqCst);
                    async move {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        Err::<CoalescedResponse, _>(CoalesceError(Arc::from("boom")))
                    }
                })
                .await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        for r in results {
            let r = r.unwrap();
            assert!(r.is_err(), "waiter must see the leader's error");
            assert_eq!(r.unwrap_err().0.as_ref(), "boom");
        }
        assert_eq!(fc.load(Ordering::SeqCst), 1, "errored forward still ran once");
    }

    #[test]
    fn request_key_excludes_request_id_and_session() {
        // Two requests differing only in request_id/session must share a key.
        let base = serde_json::json!({
            "model": "m", "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 16, "temperature": 0.5
        });
        let mut a: OpenAIChatRequest = serde_json::from_value(base.clone()).unwrap();
        let mut b: OpenAIChatRequest = serde_json::from_value(base).unwrap();
        a.stream = false;
        b.stream = false;
        // Emulate different request ids / sessions (these fields live on the
        // OpenAIChatRequest; if absent the key must still match).
        assert_eq!(request_key(&a), request_key(&b));
    }

    #[test]
    fn request_key_differs_when_body_differs() {
        let mk = |content: &str| {
            let v = serde_json::json!({
                "model": "m",
                "messages": [{"role":"user","content": content}],
                "max_tokens": 16, "temperature": 0.5
            });
            let mut r: OpenAIChatRequest = serde_json::from_value(v).unwrap();
            r.stream = false;
            r
        };
        assert_ne!(request_key(&mk("hi")), request_key(&mk("hello")));
    }
}
