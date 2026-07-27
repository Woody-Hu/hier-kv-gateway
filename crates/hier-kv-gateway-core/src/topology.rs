//! Region topology info and latency matrix.
//!
//! Describes the geographic coordinates, network zones, and external endpoints of
//! each region in the Hier KV Gateway cluster; provides [`haversine_km`] for
//! computing great-circle distance, and [`LatencyMatrix`] falls back to
//! distance-based latency estimation when matrix records are missing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ids::{RegionId, RegionTier};

/// Geographic coordinates (WGS84).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeoCoord {
    /// Latitude, in degrees, in the range [-90, 90].
    pub lat: f64,
    /// Longitude, in degrees, in the range [-180, 180].
    pub lon: f64,
}

/// Region info.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionInfo {
    /// Region identifier.
    pub id: RegionId,
    /// Region tier.
    pub tier: RegionTier,
    /// Geographic coordinates; `None` when not provided.
    pub geo: Option<GeoCoord>,
    /// Network zone label, e.g. `us-east-1a`.
    pub network_zone: String,
    /// List of endpoints exposed by this region.
    pub endpoints: Vec<String>,
}

/// Latency estimate snapshot between two regions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LatencyEstimate {
    /// P50 round-trip latency (milliseconds).
    pub rtt_p50_ms: f64,
    /// P99 round-trip latency (milliseconds).
    pub rtt_p99_ms: f64,
    /// Available inter-region bandwidth (Mbps).
    pub bandwidth_mbps: f64,
    /// Last update time (Unix seconds).
    pub last_updated_unix: i64,
}

/// Pairwise region latency matrix.
///
/// The keys of `entries` are `(source_region, target_region)`. Queries try both
/// the forward and reverse keys, since latency is treated as symmetric in both
/// directions.
pub struct LatencyMatrix {
    /// Latency entry table.
    pub entries: HashMap<(RegionId, RegionId), LatencyEstimate>,
}

impl LatencyMatrix {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Estimate the RTT (milliseconds) between two regions.
    ///
    /// Estimation priority:
    /// 1. Same region: returns 0;
    /// 2. A record exists in the matrix (forward or reverse): returns `rtt_p50_ms`;
    /// 3. The caller provides both geographic coordinates: computed from the
    ///    [`haversine_km`] distance using an empirical conversion;
    /// 4. All of the above are missing: returns `None`.
    pub fn rtt_ms(
        &self,
        a: &RegionId,
        b: &RegionId,
        geo_a: Option<&GeoCoord>,
        geo_b: Option<&GeoCoord>,
    ) -> Option<f64> {
        if a == b {
            return Some(0.0);
        }
        // Try both forward and reverse keys; latency is treated as symmetric
        if let Some(est) = self
            .entries
            .get(&(a.clone(), b.clone()))
            .or_else(|| self.entries.get(&(b.clone(), a.clone())))
        {
            return Some(est.rtt_p50_ms);
        }
        // Fall back to geographic distance estimation
        if let (Some(g_a), Some(g_b)) = (geo_a, geo_b) {
            return Some(Self::estimate_rtt_ms(g_a, g_b));
        }
        None
    }

    /// Estimate RTT (milliseconds) based on geographic coordinates.
    ///
    /// Empirical conversion: 100km ≈ 1ms RTT, approximating the total overhead
    /// of round-trip signal propagation through fiber at roughly 5/6 the speed of
    /// light plus routing hops.
    pub fn estimate_rtt_ms(a: &GeoCoord, b: &GeoCoord) -> f64 {
        haversine_km(*a, *b) / 100.0
    }
}

impl Default for LatencyMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the great-circle distance between two WGS84 coordinates using the
/// Haversine formula, in kilometers.
///
/// Earth radius is taken as 6371 km.
pub fn haversine_km(a: GeoCoord, b: GeoCoord) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;
    let to_rad = |deg: f64| deg * std::f64::consts::PI / 180.0;

    let lat1 = to_rad(a.lat);
    let lat2 = to_rad(b.lat);
    let dlat = to_rad(b.lat - a.lat);
    let dlon = to_rad(b.lon - a.lon);

    let h = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * h.sqrt().asin();
    EARTH_RADIUS_KM * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_zero_for_same_point() {
        let p = GeoCoord { lat: 30.0, lon: 120.0 };
        assert!((haversine_km(p, p)).abs() < 1e-9);
    }

    #[test]
    fn haversine_known_distance_beijing_shanghai() {
        // Beijing is approximately (39.9, 116.4); Shanghai (31.2, 121.5); straight-line distance is about 1067 km.
        let beijing = GeoCoord { lat: 39.9, lon: 116.4 };
        let shanghai = GeoCoord { lat: 31.2, lon: 121.5 };
        let d = haversine_km(beijing, shanghai);
        assert!((d - 1067.0).abs() < 25.0, "actual: {}", d);
    }

    #[test]
    fn rtt_ms_same_region_zero() {
        let m = LatencyMatrix::new();
        let r = RegionId::new("r1");
        assert_eq!(m.rtt_ms(&r, &r, None, None), Some(0.0));
    }

    #[test]
    fn rtt_ms_matrix_hit() {
        let mut m = LatencyMatrix::new();
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        m.entries.insert(
            (a.clone(), b.clone()),
            LatencyEstimate {
                rtt_p50_ms: 12.0,
                rtt_p99_ms: 20.0,
                bandwidth_mbps: 1000.0,
                last_updated_unix: 1,
            },
        );
        // Forward query
        assert_eq!(m.rtt_ms(&a, &b, None, None), Some(12.0));
        // Reverse query should also hit
        assert_eq!(m.rtt_ms(&b, &a, None, None), Some(12.0));
    }

    #[test]
    fn rtt_ms_falls_back_to_geo() {
        let m = LatencyMatrix::new();
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        let g_a = GeoCoord { lat: 0.0, lon: 0.0 };
        // 1 degree of longitude is about 111km, so estimated RTT is about 1.11 ms
        let g_b = GeoCoord { lat: 0.0, lon: 1.0 };
        let rtt = m.rtt_ms(&a, &b, Some(&g_a), Some(&g_b)).unwrap();
        assert!((rtt - 1.11).abs() < 0.1, "actual: {}", rtt);
    }

    #[test]
    fn rtt_ms_no_data_returns_none() {
        let m = LatencyMatrix::new();
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        assert_eq!(m.rtt_ms(&a, &b, None, None), None);
    }
}
