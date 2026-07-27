//! Round-robin routing strategy.
//!
//! The simplest possible baseline: rotate through the candidate set in order,
//! one position per [`evaluate`](RoutingStrategy::evaluate) call. The strategy
//! ignores all request content (KV overlap, load, topology) — it exists as
//!
//! 1. a **baseline** for benchmarking the metadata-driven strategies against,
//! 2. a **fallback-style primary** for deployments where KV metadata is not
//!    available at all and the hybrid strategy has nothing to score on.
//!
//! ## Ranked output doubles as failover order
//!
//! `evaluate` returns *every* candidate, scored so that the candidate at the
//! current cursor position ranks first and the remaining candidates follow in
//! wrap-around rotation order with strictly decreasing scores. With
//! `temperature == 0` the engine's greedy selection therefore picks the cursor
//! candidate, while the ranked list handed to the forwarding loop is exactly
//! the order in which retries should fail over.
//!
//! The cursor is a single [`AtomicUsize`], so `evaluate` is safe to call
//! concurrently and costs one `fetch_add` plus an `O(n)` score materialization
//! — no metadata store access at all.
//!
//! Note: the strategy trusts the engine's candidate pre-filter (model-name
//! based); it does not re-apply capability constraints such as tool-calling
//! support. Deployments relying on those should prefer the hybrid strategy.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::strategy::RoutingStrategy;

/// Round-robin strategy: rotate through candidates in order.
pub struct RoundRobinStrategy {
    /// Monotonic cursor; the effective start position is `cursor % len`.
    cursor: AtomicUsize,
}

impl RoundRobinStrategy {
    /// Create a strategy whose first pick is `candidates[0]`.
    pub fn new() -> Self {
        Self {
            cursor: AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobinStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RoutingStrategy for RoundRobinStrategy {
    fn name(&self) -> &'static str {
        "round_robin"
    }

    async fn evaluate(
        &self,
        _ctx: &RoutingContext,
        candidates: &[BackendId],
        _meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let n = candidates.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % n;
        let mut out = Vec::with_capacity(n);
        for offset in 0..n {
            let idx = (start + offset) % n;
            // Strictly decreasing scores: rank 0 -> n, rank n-1 -> 1. Using
            // `n - offset` (rather than a fractional score) keeps gaps uniform
            // so softmax sampling at temperature > 0 stays well-behaved.
            let score = (n - offset) as f64;
            out.push(ScoredBackend {
                backend_id: candidates[idx].clone(),
                score,
                raw_cost: -score,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, _meta: &MetadataStore) -> bool {
        // No metadata dependency: available whenever there are candidates.
        true
    }

    fn weight(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(names: &[&str]) -> Vec<BackendId> {
        names.iter().map(|n| BackendId::new("r1", *n)).collect()
    }

    fn ctx() -> RoutingContext {
        RoutingContext::default()
    }

    #[tokio::test]
    async fn rotates_through_candidates_in_order() {
        let rr = RoundRobinStrategy::new();
        let meta = MetadataStore::new();
        let cands = candidates(&["a", "b", "c"]);

        for expected_first in ["a", "b", "c", "a"] {
            let scored = rr.evaluate(&ctx(), &cands, &meta).await.unwrap();
            assert_eq!(scored.len(), 3);
            assert_eq!(scored[0].backend_id.instance.as_str(), expected_first);
            // Scores strictly decrease along the ranked list.
            assert!(scored[0].score > scored[1].score);
            assert!(scored[1].score > scored[2].score);
        }
    }

    #[tokio::test]
    async fn ranked_list_is_wraparound_failover_order() {
        let rr = RoundRobinStrategy::new();
        let meta = MetadataStore::new();
        let cands = candidates(&["a", "b", "c"]);

        // First call starts at "a"; second starts at "b" and wraps around.
        let _ = rr.evaluate(&ctx(), &cands, &meta).await.unwrap();
        let scored = rr.evaluate(&ctx(), &cands, &meta).await.unwrap();
        let order: Vec<&str> = scored
            .iter()
            .map(|s| s.backend_id.instance.as_str())
            .collect();
        assert_eq!(order, ["b", "c", "a"]);
    }

    #[tokio::test]
    async fn empty_candidates_yield_empty_scores() {
        let rr = RoundRobinStrategy::new();
        let meta = MetadataStore::new();
        let scored = rr.evaluate(&ctx(), &[], &meta).await.unwrap();
        assert!(scored.is_empty());
    }

    #[tokio::test]
    async fn single_candidate_always_selected() {
        let rr = RoundRobinStrategy::new();
        let meta = MetadataStore::new();
        let cands = candidates(&["only"]);
        for _ in 0..3 {
            let scored = rr.evaluate(&ctx(), &cands, &meta).await.unwrap();
            assert_eq!(scored.len(), 1);
            assert_eq!(scored[0].backend_id.instance.as_str(), "only");
        }
    }

    #[tokio::test]
    async fn cursor_wrap_does_not_panic() {
        let rr = RoundRobinStrategy::new();
        // Push the cursor near overflow; fetch_add wraps on release builds
        // but the modulo keeps the index in range regardless.
        rr.cursor.store(usize::MAX - 1, Ordering::Relaxed);
        let meta = MetadataStore::new();
        let cands = candidates(&["a", "b"]);
        let scored = rr.evaluate(&ctx(), &cands, &meta).await.unwrap();
        assert_eq!(scored.len(), 2);
    }
}
