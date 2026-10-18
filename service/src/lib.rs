//! Lymph-node metastasis patch classification and tile heatmaps over ONNX Runtime.

pub mod api;
pub mod auth;
pub mod batcher;
pub mod config;
pub mod error;
pub mod imaging;
pub mod model;

use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

/// Install the global Prometheus recorder with latency buckets suited to this service.
pub fn install_metrics() -> anyhow::Result<PrometheusHandle> {
    let seconds = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
    ];
    let batch = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    Ok(PrometheusBuilder::new()
        .set_buckets_for_metric(Matcher::Suffix("_seconds".into()), &seconds)?
        .set_buckets_for_metric(Matcher::Full("pathology_batch_size".into()), &batch)?
        .install_recorder()?)
}
