//! HTTP API.
//!
//! POST /v1/classify   1-64 base64 PNG/JPEG patches (96x96)  -> calibrated probability per patch
//! POST /v1/heatmap    one tile up to `MAX_TILE_SIDE` pixels    -> sliding-window probabilities (row-major)
//! GET  /v1/model      model version, threshold, temperature
//! GET  /health        liveness;  GET /ready  model loaded and warmed up;  GET /metrics  Prometheus

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Extension, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router, middleware};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::auth::{Auth, Client, require_key};
use crate::batcher::Batcher;
use crate::error::ApiError;
use crate::imaging::{PATCH, PATCH_LEN, decode_base64_rgb, grid, push_patch, windows};
use crate::model::ModelMeta;

const REPLY_TIMEOUT: Duration = Duration::from_secs(20);

pub struct AppState {
    pub batcher: Batcher,
    pub meta: ModelMeta,
    pub ready: AtomicBool,
    pub max_patches: usize,
    pub max_tile_side: u32,
    pub min_stride: u32,
    pub prometheus: Option<PrometheusHandle>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchIn {
    pub id: Option<String>,
    pub image_base64: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifyRequest {
    pub patches: Vec<PatchIn>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PatchOut {
    pub id: String,
    pub probability: f32,
    pub tumor: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ClassifyResponse {
    pub model_version: String,
    pub threshold: f32,
    pub results: Vec<PatchOut>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeatmapRequest {
    pub image_base64: String,
    pub stride: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HeatmapResponse {
    pub model_version: String,
    pub threshold: f32,
    pub patch_size: u32,
    pub stride: u32,
    pub rows: u32,
    pub cols: u32,
    /// Row-major, `rows * cols` values; cell (r, c) covers pixels [r*stride, r*stride+96) x [c*stride, ...).
    pub probabilities: Vec<f32>,
    pub max_probability: f32,
    pub windows_above_threshold: usize,
}

async fn score(state: &AppState, inputs: Vec<f32>, n: usize) -> Result<Vec<f32>, ApiError> {
    if !state.ready.load(Ordering::Acquire) {
        return Err(ApiError::Unavailable("model is still loading".into()));
    }
    match tokio::time::timeout(REPLY_TIMEOUT, state.batcher.predict_many(inputs, n)).await {
        Ok(Ok(p)) => Ok(p),
        Ok(Err(e)) => Err(ApiError::Unavailable(e.to_string())),
        Err(_) => Err(ApiError::Unavailable("inference timed out".into())),
    }
}

async fn classify(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<Arc<Auth>>,
    Extension(Client(client)): Extension<Client>,
    Json(req): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, ApiError> {
    let n = req.patches.len();
    if n == 0 || n > state.max_patches {
        return Err(ApiError::BadRequest(format!(
            "send 1 to {} patches, got {n}",
            state.max_patches
        )));
    }
    auth.check_rate(&client, n as u32)?;
    let started = Instant::now();
    let mut inputs = Vec::with_capacity(n * PATCH_LEN);
    let mut ids = Vec::with_capacity(n);
    for (i, p) in req.patches.iter().enumerate() {
        let img = decode_base64_rgb(&p.image_base64, PATCH)?;
        push_patch(&img, &mut inputs)?;
        ids.push(p.id.clone().unwrap_or_else(|| i.to_string()));
    }
    metrics::histogram!("pathology_decode_seconds", "route" => "classify")
        .record(started.elapsed().as_secs_f64());
    let probs = score(&state, inputs, n).await?;
    let t = state.meta.threshold;
    metrics::counter!("pathology_patches_total", "route" => "classify").increment(n as u64);
    let results = ids
        .into_iter()
        .zip(probs)
        .map(|(id, p)| PatchOut {
            id,
            probability: p,
            tumor: p >= t,
        })
        .collect();
    Ok(Json(ClassifyResponse {
        model_version: state.meta.version.clone(),
        threshold: t,
        results,
    }))
}

async fn heatmap(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<Arc<Auth>>,
    Extension(Client(client)): Extension<Client>,
    Json(req): Json<HeatmapRequest>,
) -> Result<Json<HeatmapResponse>, ApiError> {
    let stride = req.stride.unwrap_or(PATCH);
    if stride < state.min_stride || stride > PATCH {
        return Err(ApiError::BadRequest(format!(
            "stride must be between {} and {PATCH}",
            state.min_stride
        )));
    }
    let started = Instant::now();
    let tile = decode_base64_rgb(&req.image_base64, state.max_tile_side)?;
    let g = grid(tile.width(), tile.height(), stride)?;
    let n = (g.rows * g.cols) as usize;
    auth.check_rate(&client, n as u32)?;
    let inputs = windows(&tile, g);
    metrics::histogram!("pathology_decode_seconds", "route" => "heatmap")
        .record(started.elapsed().as_secs_f64());
    let probs = score(&state, inputs, n).await?;
    metrics::counter!("pathology_patches_total", "route" => "heatmap").increment(n as u64);
    let t = state.meta.threshold;
    Ok(Json(HeatmapResponse {
        model_version: state.meta.version.clone(),
        threshold: t,
        patch_size: PATCH,
        stride,
        rows: g.rows,
        cols: g.cols,
        max_probability: probs.iter().copied().fold(0.0, f32::max),
        windows_above_threshold: probs.iter().filter(|&&p| p >= t).count(),
        probabilities: probs,
    }))
}

async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelMeta> {
    Json(state.meta.clone())
}

async fn health() -> &'static str {
    "ok"
}

async fn ready(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "loading")
    }
}

async fn render_metrics(State(state): State<Arc<AppState>>) -> String {
    state
        .prometheus
        .as_ref()
        .map(PrometheusHandle::render)
        .unwrap_or_default()
}

async fn track(req: axum::extract::Request, next: middleware::Next) -> axum::response::Response {
    let route = req.uri().path().to_string();
    let started = Instant::now();
    let response = next.run(req).await;
    if route.starts_with("/v1/") {
        let status = response.status().as_u16().to_string();
        metrics::histogram!("pathology_request_seconds", "route" => route.clone())
            .record(started.elapsed().as_secs_f64());
        metrics::counter!("pathology_requests_total", "route" => route, "status" => status)
            .increment(1);
    }
    response
}

pub fn router(state: Arc<AppState>, auth: Arc<Auth>, cors_origins: &[String]) -> Router {
    let protected = Router::new()
        .route("/v1/classify", post(classify))
        .route("/v1/heatmap", post(heatmap))
        .route("/v1/model", get(model_info))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&auth),
            require_key,
        ))
        .layer(Extension(auth));
    let origins: Vec<HeaderValue> = cors_origins
        .iter()
        .filter_map(|o| HeaderValue::from_str(o).ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::HeaderName::from_static("x-api-key"),
        ]);
    Router::new()
        .merge(protected)
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(render_metrics))
        // A 2048x2048 PNG tile base64-encoded is at most ~16 MB; anything larger is refused before parsing.
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .layer(middleware::from_fn(track))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
