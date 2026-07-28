//! Routing decision telemetry events.
//!
//! A [`DecisionEvent`] captures the full profile of one gateway routing
//! decision: the candidate scores the strategies produced, the effective
//! hybrid weights, the failover attempts the forwarding loop made, and the
//! final outcome. Events are emitted once per request through a
//! [`DecisionEventSink`], making the gateway's decision stream consumable by
//! external analysis systems (offline replay, A/B evaluation, weight tuning).
//!
//! The sink abstraction is deliberately synchronous and cheap: hot-path
//! implementations (ring buffer, tracing) must not block the request; the
//! NDJSON file sink ships events over a channel to a background writer for
//! exactly that reason (see the `hier-kv-gateway-api` crate).

use serde::{Deserialize, Serialize};

/// One scored candidate as produced by the routing engine.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateScore {
    /// Candidate backend, in `region/instance` form.
    pub backend: String,
    /// Final score from the selecting strategy (higher is better).
    pub score: f64,
    /// KV prefix overlap (blocks) known at decision time.
    pub kv_overlap: u32,
}

/// One forwarding attempt against a candidate backend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardAttempt {
    /// Backend this attempt targeted, in `region/instance` form.
    pub backend: String,
    /// Whether the attempt succeeded.
    pub success: bool,
    /// The attempt was skipped because the backend's circuit was open.
    pub skipped_open_circuit: bool,
    /// Error message when the attempt failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Terminal outcome of the request the decision served.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcome {
    /// A backend stream was established successfully.
    Success,
    /// The routing engine produced no usable candidate.
    RoutingFailed,
    /// Every candidate failed (or was short-circuited) during forwarding.
    AllCandidatesFailed,
}

/// Snapshot of the effective hybrid weights used for this decision.
///
/// Present only when the hybrid strategy ran; `round_robin`-only or
/// session-affinity decisions leave it `None`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct WeightSnapshot {
    /// Effective KV weight.
    pub kv: f64,
    /// Effective load weight.
    pub load: f64,
    /// Effective topology weight.
    pub topology: f64,
}

/// Full profile of one routing decision, emitted once per request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionEvent {
    /// Unique event identifier (UUIDv4).
    pub event_id: String,
    /// Emission timestamp (Unix milliseconds).
    pub timestamp_unix_ms: i64,
    /// Gateway instance that made the decision.
    pub gateway_instance: String,
    /// Region the gateway belongs to.
    pub gateway_region: String,
    /// Client request identifier.
    pub request_id: String,
    /// Requested model name.
    pub model: String,
    /// Session identifier, when the client supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Strategy that produced the winning decision.
    pub strategy: String,
    /// Effective hybrid weights at decision time (when applicable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<WeightSnapshot>,
    /// Ranked candidate scores considered by the engine.
    pub candidates: Vec<CandidateScore>,
    /// Ordered forwarding attempts (including circuit-skipped ones).
    pub attempts: Vec<ForwardAttempt>,
    /// Backend that ultimately served the request (`None` on failure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_backend: Option<String>,
    /// KV overlap of the winning decision.
    pub kv_overlap: u32,
    /// Prompt size in KV blocks (routing input).
    pub prompt_blocks: u32,
    /// Time spent inside the routing engine (microseconds).
    pub routing_latency_us: u64,
    /// Total gateway handling time until the outcome was known (microseconds).
    pub total_latency_us: u64,
    /// Terminal outcome.
    pub outcome: DecisionOutcome,
}

/// Destination for decision events.
///
/// `emit` is called on the request hot path, so implementations must be
/// non-blocking and allocation-light; anything I/O-heavy must hand the event
/// off to a background task (the NDJSON file sink in the API crate does this
/// with an unbounded channel).
pub trait DecisionEventSink: Send + Sync {
    /// Consume one decision event. Must not panic and must not block.
    fn emit(&self, event: &DecisionEvent);
}

/// Sink that discards every event (the default when telemetry is disabled).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopSink;

impl DecisionEventSink for NoopSink {
    fn emit(&self, _event: &DecisionEvent) {}
}

/// Fan-out sink: forwards each event to every child sink, in order.
pub struct MultiSink {
    /// Child sinks.
    pub sinks: Vec<Box<dyn DecisionEventSink>>,
}

impl MultiSink {
    /// Create a fan-out sink from the given children.
    pub fn new(sinks: Vec<Box<dyn DecisionEventSink>>) -> Self {
        Self { sinks }
    }
}

impl DecisionEventSink for MultiSink {
    fn emit(&self, event: &DecisionEvent) {
        for s in &self.sinks {
            s.emit(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> DecisionEvent {
        DecisionEvent {
            event_id: "evt-1".to_string(),
            timestamp_unix_ms: 1_700_000_000_000,
            gateway_instance: "gw-1".to_string(),
            gateway_region: "cloud-cn-beijing".to_string(),
            request_id: "req-1".to_string(),
            model: "qwen2.5-7b".to_string(),
            session_id: Some("sess-9".to_string()),
            strategy: "hybrid".to_string(),
            weights: Some(WeightSnapshot {
                kv: 0.35,
                load: 0.30,
                topology: 0.20,
            }),
            candidates: vec![
                CandidateScore {
                    backend: "r1/a".to_string(),
                    score: 0.9,
                    kv_overlap: 8,
                },
                CandidateScore {
                    backend: "r1/b".to_string(),
                    score: 0.4,
                    kv_overlap: 0,
                },
            ],
            attempts: vec![
                ForwardAttempt {
                    backend: "r1/a".to_string(),
                    success: false,
                    skipped_open_circuit: false,
                    error: Some("connection refused".to_string()),
                },
                ForwardAttempt {
                    backend: "r1/b".to_string(),
                    success: true,
                    skipped_open_circuit: false,
                    error: None,
                },
            ],
            selected_backend: Some("r1/b".to_string()),
            kv_overlap: 0,
            prompt_blocks: 12,
            routing_latency_us: 320,
            total_latency_us: 4_800,
            outcome: DecisionOutcome::Success,
        }
    }

    #[test]
    fn decision_event_serde_round_trip() {
        let ev = sample_event();
        let s = serde_json::to_string(&ev).unwrap();
        let back: DecisionEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn optional_fields_are_omitted_when_none() {
        let mut ev = sample_event();
        ev.session_id = None;
        ev.weights = None;
        ev.selected_backend = None;
        ev.attempts[0].error = None;
        let s = serde_json::to_string(&ev).unwrap();
        assert!(!s.contains("session_id"));
        assert!(!s.contains("\"weights\""));
        assert!(!s.contains("selected_backend"));
        assert!(!s.contains("\"error\""));
    }

    #[test]
    fn outcome_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&DecisionOutcome::AllCandidatesFailed).unwrap(),
            r#""all_candidates_failed""#
        );
    }

    #[test]
    fn noop_sink_accepts_events() {
        let sink = NoopSink;
        sink.emit(&sample_event());
    }

    #[test]
    fn multi_sink_fans_out_in_order() {
        use std::sync::{Arc, Mutex};

        struct RecordingSink {
            id: u32,
            log: Arc<Mutex<Vec<u32>>>,
        }
        impl DecisionEventSink for RecordingSink {
            fn emit(&self, _event: &DecisionEvent) {
                self.log.lock().unwrap().push(self.id);
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let sink = MultiSink::new(vec![
            Box::new(RecordingSink {
                id: 1,
                log: log.clone(),
            }),
            Box::new(RecordingSink {
                id: 2,
                log: log.clone(),
            }),
        ]);
        sink.emit(&sample_event());
        assert_eq!(*log.lock().unwrap(), vec![1, 2]);
    }
}
