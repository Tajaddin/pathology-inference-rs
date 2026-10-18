use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use pathology_service::api::{AppState, router};
use pathology_service::auth::Auth;
use pathology_service::batcher::{BatchConfig, Batcher};
use pathology_service::config::Config;
use pathology_service::imaging::PATCH_LEN;
use pathology_service::model::{Classifier, ModelMeta, OnnxClassifier};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn")),
        )
        .init();
    let config = Config::from_env()?;
    let prometheus = pathology_service::install_metrics()?;
    let meta = ModelMeta::load(&config.meta_path)?;
    tracing::info!(version = %meta.version, workers = config.workers, intra_threads = config.intra_threads,
                   max_batch = config.max_batch, max_wait_us = config.max_wait.as_micros() as u64, "starting");

    let env = Arc::new(
        ort::init()
            .with_name("pathology")
            .build()
            .context("ONNX Runtime environment")?,
    );
    let (model_path, model_meta, threads) = (
        config.model_path.clone(),
        meta.clone(),
        config.intra_threads,
    );
    let batcher = Batcher::spawn(
        BatchConfig {
            max_batch: config.max_batch,
            max_wait: config.max_wait,
            workers: config.workers,
            queue: 1024,
        },
        move |_| {
            Ok(Box::new(OnnxClassifier::load(
                &env,
                &model_path,
                model_meta.clone(),
                threads,
            )?) as Box<dyn Classifier>)
        },
    )?;

    let state = Arc::new(AppState {
        batcher: batcher.clone(),
        meta,
        ready: AtomicBool::new(false),
        max_patches: config.max_patches_per_request,
        max_tile_side: config.max_tile_side,
        min_stride: config.min_stride,
        prometheus: Some(prometheus),
    });
    let auth = Arc::new(Auth::new(
        config.api_keys.clone(),
        config.rate_limit_per_minute,
    ));
    if config.api_keys.is_empty() {
        tracing::warn!("API_KEYS is empty: every /v1 request will be refused");
    }

    // Warm up: the first ONNX Runtime run allocates and plans; /ready turns true only after it.
    let warm = Arc::clone(&state);
    tokio::spawn(async move {
        let n = warm.batcher.max_batch();
        match warm.batcher.predict(vec![128.0; n * PATCH_LEN], n).await {
            Ok(_) => {
                warm.ready.store(true, Ordering::Release);
                tracing::info!("model warmed up; ready");
            }
            Err(err) => tracing::error!(error = %err, "warm-up failed; staying not ready"),
        }
    });

    let app = router(state, auth, &config.cors_origins);
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { () = ctrl_c => {}, () = term => {} }
    tracing::info!("shutting down");
}
