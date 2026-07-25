//! 区域拓扑信息与延迟矩阵。
//!
//! 描述 Aether 集群中各区域的地理坐标、网络区域与对外端点；
//! 提供 [`haversine_km`] 计算球面距离，[`LatencyMatrix`] 在矩阵缺失记录时
//! 退化为基于地理距离的延迟估算。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ids::{RegionId, RegionTier};

/// 地理坐标（WGS84）。
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeoCoord {
    /// 纬度，单位：度，范围 [-90, 90]。
    pub lat: f64,
    /// 经度，单位：度，范围 [-180, 180]。
    pub lon: f64,
}

/// 区域信息。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionInfo {
    /// 区域标识。
    pub id: RegionId,
    /// 区域层级。
    pub tier: RegionTier,
    /// 地理坐标，未提供时为 `None`。
    pub geo: Option<GeoCoord>,
    /// 网络区域标签，例如 `us-east-1a`。
    pub network_zone: String,
    /// 该区域对外暴露的端点列表。
    pub endpoints: Vec<String>,
}

/// 两区域之间的延迟估计快照。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LatencyEstimate {
    /// P50 往返延迟（毫秒）。
    pub rtt_p50_ms: f64,
    /// P99 往返延迟（毫秒）。
    pub rtt_p99_ms: f64,
    /// 区域间可用带宽（Mbps）。
    pub bandwidth_mbps: f64,
    /// 最近一次更新时间（Unix 秒）。
    pub last_updated_unix: i64,
}

/// 区域两两延迟矩阵。
///
/// entries 的键为 `(源区域, 目标区域)`。查询时同时尝试正向与反向键，
/// 因为延迟在两个方向上视作对称。
pub struct LatencyMatrix {
    /// 延迟条目表。
    pub entries: HashMap<(RegionId, RegionId), LatencyEstimate>,
}

impl LatencyMatrix {
    /// 创建空矩阵。
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// 估算两区域之间的 RTT（毫秒）。
    ///
    /// 估算优先级：
    /// 1. 同一区域：返回 0；
    /// 2. 矩阵中存在记录（正向或反向）：返回 `rtt_p50_ms`；
    /// 3. 调用方提供了双方地理坐标：用 [`haversine_km`] 距离按经验值折算；
    /// 4. 上述信息都缺失：返回 `None`。
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
        // 同时尝试正向与反向键，延迟视作对称
        if let Some(est) = self
            .entries
            .get(&(a.clone(), b.clone()))
            .or_else(|| self.entries.get(&(b.clone(), a.clone())))
        {
            return Some(est.rtt_p50_ms);
        }
        // 退化为地理距离估算
        if let (Some(g_a), Some(g_b)) = (geo_a, geo_b) {
            return Some(Self::estimate_rtt_ms(g_a, g_b));
        }
        None
    }

    /// 基于地理坐标估算 RTT（毫秒）。
    ///
    /// 经验折算：100km ≈ 1ms RTT，对应光速往返在大约 5/6 的光速下穿越光纤
    /// 与路由跳数的总开销近似。
    pub fn estimate_rtt_ms(a: &GeoCoord, b: &GeoCoord) -> f64 {
        haversine_km(*a, *b) / 100.0
    }
}

impl Default for LatencyMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// 用 Haversine 公式计算两个 WGS84 坐标之间的球面距离，单位：千米。
///
/// 地球半径取 6371 km。
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
        // 北京约 (39.9, 116.4)；上海约 (31.2, 121.5)；直线距离约 1067 km。
        let beijing = GeoCoord { lat: 39.9, lon: 116.4 };
        let shanghai = GeoCoord { lat: 31.2, lon: 121.5 };
        let d = haversine_km(beijing, shanghai);
        assert!((d - 1067.0).abs() < 25.0, "实际计算: {}", d);
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
        // 正向查询
        assert_eq!(m.rtt_ms(&a, &b, None, None), Some(12.0));
        // 反向查询也应命中
        assert_eq!(m.rtt_ms(&b, &a, None, None), Some(12.0));
    }

    #[test]
    fn rtt_ms_falls_back_to_geo() {
        let m = LatencyMatrix::new();
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        let g_a = GeoCoord { lat: 0.0, lon: 0.0 };
        // 经度 1 度约 111km，估算 RTT 约 1.11 ms
        let g_b = GeoCoord { lat: 0.0, lon: 1.0 };
        let rtt = m.rtt_ms(&a, &b, Some(&g_a), Some(&g_b)).unwrap();
        assert!((rtt - 1.11).abs() < 0.1, "实际: {}", rtt);
    }

    #[test]
    fn rtt_ms_no_data_returns_none() {
        let m = LatencyMatrix::new();
        let a = RegionId::new("a");
        let b = RegionId::new("b");
        assert_eq!(m.rtt_ms(&a, &b, None, None), None);
    }
}
