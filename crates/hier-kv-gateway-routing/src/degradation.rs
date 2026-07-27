//! Degradation routing strategy: lightweight prefix-history-based routing.
//!
//! When the gateway is in degraded mode (KV metadata unavailable or stale,
//! model registry empty, or insufficient data for normal Hybrid routing),
//! this strategy falls back to local prefix reuse history to make a
//! best-effort routing decision.
//!
//! Degradation is triggered when:
//! - KV confidence is 0 (no KV index data), AND
//! - Model registry has no matching backends, OR
//! - Load stats are all stale (>30s)
//!
//! Scoring: for each candidate backend, find the longest prefix match in
//! the dispatch history. score = match_length / total_hash_length.
//! Backends not in history get score 0 but are still included as fallback.

use std::sync::Arc;

use async_trait::async_trait;

use hier_kv_gateway_core::error::Result;
use hier_kv_gateway_core::ids::BackendId;
use hier_kv_gateway_core::request::{RoutingContext, ScoredBackend};
use hier_kv_gateway_metadata::store::MetadataStore;

use crate::prefix_history::PrefixReuseHistory;
use crate::strategy::RoutingStrategy;

/// Load staleness threshold for degradation: if all backends' load data is
/// older than this, consider the gateway degraded.
const LOAD_STALE_THRESHOLD_SECS: u64 = 30;

/// Minimum history entries required for degradation routing to be confident.
/// Below this, degradation still works but scores are lower.
const MIN_CONFIDENT_HISTORY_ENTRIES: usize = 10;

/// Degradation routing strategy.
///
/// This is a fallback strategy used when the normal Hybrid routing is
/// unavailable or untrustworthy. It consults the local
/// [`PrefixReuseHistory`] to replay a previously-successful dispatch
/// decision for the longest matching prefix of the current request.
pub struct DegradationStrategy {
    /// Local prefix reuse history (shared with the routing engine).
    pub history: Arc<PrefixReuseHistory>,
}

impl DegradationStrategy {
    /// Build a new degradation strategy bound to the given history.
    pub fn new(history: Arc<PrefixReuseHistory>) -> Self {
        Self { history }
    }
}

#[async_trait]
impl RoutingStrategy for DegradationStrategy {
    fn name(&self) -> &'static str {
        "degradation"
    }

    async fn evaluate(
        &self,
        ctx: &RoutingContext,
        candidates: &[BackendId],
        _meta: &MetadataStore,
    ) -> Result<Vec<ScoredBackend>> {
        let hashes = ctx.block_hashes.as_slice();
        let total_len = hashes.len().max(1) as f64;

        // Find longest prefix match in history.
        let best_match = self.history.find_longest_match(hashes);

        // Confidence factor: if history is small, lower scores.
        let history_size = self.history.len();
        let confidence = if history_size >= MIN_CONFIDENT_HISTORY_ENTRIES {
            1.0
        } else {
            history_size as f64 / MIN_CONFIDENT_HISTORY_ENTRIES as f64
        };

        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let (score, raw_cost) = if let Some((match_len, ref record)) = best_match {
                if &record.backend == cand {
                    // This backend handled the matching prefix before.
                    let prefix_score = match_len as f64 / total_len;
                    (prefix_score * confidence, -(prefix_score * confidence))
                } else {
                    // Different backend; give small score as fallback.
                    (0.01, 1.0)
                }
            } else {
                // No history match; uniform low score.
                (0.01, 1.0)
            };
            out.push(ScoredBackend {
                backend_id: cand.clone(),
                score,
                raw_cost,
                meta_version: 0,
            });
        }
        Ok(out)
    }

    fn is_available(&self, meta: &MetadataStore) -> bool {
        // Degradation is available when normal routing is NOT:
        // - KV confidence is 0 (no KV data), or
        // - No backends registered, or
        // - All load stats stale.
        let kv_degraded = meta.kv_confidence() == 0.0;
        let no_backends = meta.backends_len() == 0;
        let all_load_stale = {
            let backends = meta.backends_all();
            if backends.is_empty() {
                true
            } else {
                backends.iter().all(|b| {
                    meta.load_freshness(&b.id)
                        .map(|d| d.as_secs() > LOAD_STALE_THRESHOLD_SECS)
                        .unwrap_or(true)
                })
            }
        };
        kv_degraded || no_backends || all_load_stale
    }

    fn weight(&self) -> f64 {
        // Low weight, only used as fallback.
        0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hier_kv_gateway_core::ids::BackendId;
    use hier_kv_gateway_core::request::RoutingContext;
    use hier_kv_gateway_metadata::store::MetadataStore;

    fn backend(name: &str) -> BackendId {
        BackendId::new("r1", name)
    }

    fn ctx_with_hashes(hashes: &[u64]) -> RoutingContext {
        RoutingContext {
            block_hashes: hashes.to_vec(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_evaluate_with_history_match() {
        let hist = Arc::new(PrefixReuseHistory::new(100));
        let a = backend("a");
        let b = backend("b");
        // Pre-populate history so that prefix [1, 2, 3] was routed to `a`.
        // Stuff enough entries to push history past the confidence threshold.
        for i in 0..MIN_CONFIDENT_HISTORY_ENTRIES {
            hist.record_dispatch(&[1, 2, 3, 100 + i as u64], &a);
        }
        // Also record the actual target prefix.
        hist.record_dispatch(&[1, 2, 3], &a);

        let strat = DegradationStrategy::new(hist);
        let meta = MetadataStore::new();
        let ctx = ctx_with_hashes(&[1, 2, 3, 4]);
        let candidates = vec![a.clone(), b.clone()];

        let scored = strat.evaluate(&ctx, &candidates, &meta).await.unwrap();
        // Find a's score; should be the highest since it matched.
        let a_score = scored
            .iter()
            .find(|s| s.backend_id == a)
            .map(|s| s.score)
            .unwrap();
        let b_score = scored
            .iter()
            .find(|s| s.backend_id == b)
            .map(|s| s.score)
            .unwrap();
        assert!(a_score > b_score, "matched backend must outrank fallback");
        // match_len=3, total_len=4, confidence=1.0 -> 0.75
        assert!((a_score - 0.75).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_evaluate_without_history_match() {
        let hist = Arc::new(PrefixReuseHistory::new(100));
        let a = backend("a");
        let b = backend("b");

        let strat = DegradationStrategy::new(hist);
        let meta = MetadataStore::new();
        let ctx = ctx_with_hashes(&[1, 2, 3]);
        let candidates = vec![a.clone(), b.clone()];

        let scored = strat.evaluate(&ctx, &candidates, &meta).await.unwrap();
        // Without any history, every candidate gets the uniform fallback score.
        for s in &scored {
            assert!((s.score - 0.01).abs() < 1e-9);
        }
    }

    #[test]
    fn test_is_available_when_no_backends() {
        let hist = Arc::new(PrefixReuseHistory::new(100));
        let strat = DegradationStrategy::new(hist);
        let meta = MetadataStore::new();
        // Empty metadata store has no backends and kv_confidence == 0.
        assert!(strat.is_available(&meta));
    }

    #[test]
    fn test_low_confidence_lowers_score() {
        let hist = Arc::new(PrefixReuseHistory::new(100));
        let a = backend("a");
        // Record a single prefix [1, 2, 3]; this populates 3 history entries
        // (one per prefix length), still below MIN_CONFIDENT_HISTORY_ENTRIES.
        hist.record_dispatch(&[1, 2, 3], &a);
        let history_size = hist.len();
        assert!(history_size < MIN_CONFIDENT_HISTORY_ENTRIES);

        let strat = DegradationStrategy::new(hist);
        let meta = MetadataStore::new();
        let ctx = ctx_with_hashes(&[1, 2, 3, 4]);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let scored = rt
            .block_on(strat.evaluate(&ctx, &[a.clone()], &meta))
            .unwrap();
        let a_score = scored[0].score;
        // match_len=3, total_len=4 -> prefix_score=0.75
        // confidence = history_size / MIN_CONFIDENT_HISTORY_ENTRIES (< 1.0)
        // expected = prefix_score * confidence
        let expected = 0.75 * (history_size as f64 / MIN_CONFIDENT_HISTORY_ENTRIES as f64);
        assert!(
            (a_score - expected).abs() < 1e-9,
            "expected {expected}, got {a_score}"
        );
    }
}
