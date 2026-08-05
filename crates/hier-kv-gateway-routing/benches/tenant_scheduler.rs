//! Benchmarks for the multi-tenant admission scheduler.
//!
//! Run with:
//! ```bash
//! cargo bench -p hier-kv-gateway-routing --bench tenant_scheduler
//! ```

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use hier_kv_gateway_core::config::{TenantConfig, TenantQuotaConfig};
use hier_kv_gateway_core::ids::TenantId;
use hier_kv_gateway_core::tenant::TenantPriority;
use hier_kv_gateway_routing::tenant_scheduler::TenantScheduler;

fn build_config(n_tenants: usize) -> TenantConfig {
    let mut tenants = Vec::with_capacity(n_tenants);
    for i in 0..n_tenants {
        let priority = match i % 3 {
            0 => TenantPriority::Premium,
            1 => TenantPriority::Normal,
            _ => TenantPriority::Background,
        };
        let rps = match priority {
            TenantPriority::Premium => 500.0,
            TenantPriority::Normal => 100.0,
            TenantPriority::Background => 10.0,
        };
        tenants.push(TenantQuotaConfig {
            id: format!("tenant-{}", i),
            priority,
            max_rps: Some(rps),
            max_concurrent: Some(50),
            reserved_capacity_fraction: if priority == TenantPriority::Premium { 0.1 } else { 0.0 },
        });
    }
    TenantConfig {
        enabled: true,
        default_max_rps: Some(100.0),
        default_max_concurrent: Some(10),
        saturation_threshold: 0.8,
        tenants,
    }
}

fn bench_check_admission_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_check_admission");
    group.sample_size(200);

    for n_tenants in [1usize, 10, 100] {
        let config = build_config(n_tenants);
        let scheduler = TenantScheduler::new(config);
        for i in 0..n_tenants {
            let tid = TenantId::from(format!("tenant-{}", i));
            black_box(scheduler.check_admission(&tid));
        }
        let unknown = TenantId::from("unknown");

        group.bench_with_input(
            BenchmarkId::new("known_tenant", n_tenants),
            &n_tenants,
            |b, &_n| {
                let tid = TenantId::from("tenant-0");
                b.iter(|| {
                    let decision = scheduler.check_admission(black_box(&tid));
                    black_box(decision);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("unknown_tenant", n_tenants),
            &n_tenants,
            |b, &_n| {
                b.iter(|| {
                    let decision = scheduler.check_admission(black_box(&unknown));
                    black_box(decision);
                });
            },
        );
    }
    group.finish();
}

fn bench_check_admission_disabled(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_check_admission_disabled");
    group.sample_size(500);

    let config = TenantConfig { enabled: false, ..Default::default() };
    let scheduler = TenantScheduler::new(config);
    let tid = TenantId::from("anyone");

    group.bench_function("disabled_fast_path", |b| {
        b.iter(|| {
            let decision = scheduler.check_admission(black_box(&tid));
            black_box(decision);
        });
    });
    group.finish();
}

fn bench_check_admission_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_check_admission_contended");
    group.sample_size(100);

    for n_threads in [1usize, 4, 8] {
        let config = build_config(10);
        let scheduler = Arc::new(TenantScheduler::new(config));
        let tids: Vec<TenantId> = (0..10).map(|i| TenantId::from(format!("tenant-{}", i))).collect();

        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            &n_threads,
            |b, &_n| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::with_capacity(n_threads);
                    for t in 0..n_threads {
                        let s = scheduler.clone();
                        let tids = tids.clone();
                        handles.push(tokio::task::spawn(async move {
                            let mut count = 0u64;
                            for _ in 0..1000 {
                                let tid = &tids[(t + count as usize) % tids.len()];
                                let d = s.check_admission(tid);
                                if matches!(d, hier_kv_gateway_core::tenant::AdmissionDecision::Admitted) {
                                    count += 1;
                                }
                            }
                            count
                        }));
                    }
                    let mut total = 0u64;
                    for h in handles {
                        total += h.await.unwrap();
                    }
                    black_box(total);
                });
            },
        );
    }
    group.finish();
}

fn bench_token_bucket_consume(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_token_bucket_consume");
    group.sample_size(500);
    group.throughput(Throughput::Elements(1));

    let config = TenantConfig {
        enabled: true,
        default_max_rps: Some(1_000_000.0),
        ..Default::default()
    };
    let scheduler = TenantScheduler::new(config);
    let tid = TenantId::from("fast");

    group.bench_function("consume_unlimited", |b| {
        b.iter(|| {
            let decision = scheduler.check_admission(black_box(&tid));
            black_box(decision);
        });
    });
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_check_admission_hot_path, bench_check_admission_disabled,
              bench_check_admission_contended, bench_token_bucket_consume,
);
criterion_main!(benches);
