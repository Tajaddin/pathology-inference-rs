//! Peak heap of the largest heatmaps the API accepts, measured with a counting global allocator.
//!
//! A 2048x2048 tile at stride 32 has 3,844 windows: 425 MB of model inputs if they were laid out at once. The
//! service lays them out one batch at a time under the in-flight budget, so the peak is the decoded tiles plus
//! that budget, however many requests run together.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use http_body_util::BodyExt;
use pathology_service::api::{AppState, HeatmapResponse, router};
use pathology_service::auth::Auth;
use pathology_service::batcher::{BatchConfig, Batcher};
use pathology_service::imaging::PATCH_LEN;
use pathology_service::model::{Classifier, FakeClassifier, ModelMeta};
use tower::ServiceExt;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const KEY: &str = "test-key-viewer-0000001";
const MIB: usize = 1024 * 1024;
const BUDGET: usize = 512;
const MAX_BATCH: usize = 32;
const WORKERS: usize = 2;

fn app() -> axum::Router {
    let batcher = Batcher::spawn(
        BatchConfig {
            max_batch: MAX_BATCH,
            max_wait: Duration::from_micros(500),
            workers: WORKERS,
            max_inflight_patches: BUDGET,
            admit_timeout: Duration::from_secs(1),
        },
        |_| Ok(Box::new(FakeClassifier) as Box<dyn Classifier>),
    )
    .unwrap();
    let state = Arc::new(AppState {
        batcher,
        meta: ModelMeta {
            version: "test".into(),
            input_name: "x".into(),
            output_name: "y".into(),
            temperature: 1.0,
            threshold: 0.5,
        },
        ready: AtomicBool::new(true),
        max_patches: 64,
        max_windows: 4096,
        max_tile_side: 2048,
        min_stride: 32,
        prometheus: None,
    });
    let auth = Arc::new(Auth::new(vec![("viewer".into(), KEY.into())], 1_000_000));
    router(state, auth, &[])
}

async fn heatmap(app: axum::Router, body: String) {
    let req = Request::post("/v1/heatmap")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {KEY}"))
        .body(Body::from(body))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let res: HeatmapResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!((res.rows, res.cols), (62, 62));
}

/// Peak heap above what was live when `run` started, in bytes.
async fn peak_of(run: impl Future<Output = ()>) -> usize {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    run.await;
    PEAK.load(Ordering::SeqCst) - base
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn largest_heatmaps_stay_within_the_inflight_budget() {
    let tile = image::RgbImage::from_fn(2048, 2048, |x, y| {
        image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 256) as u8])
    });
    let mut png = Vec::new();
    tile.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    drop(tile);
    let body = serde_json::json!({"image_base64": STANDARD.encode(&png), "stride": 32}).to_string();
    drop(png);

    let tile_bytes = 2048 * 2048 * 3;
    // The budget's inputs, plus one batch per worker that the worker copies them into.
    let inputs = (BUDGET + WORKERS * MAX_BATCH) * PATCH_LEN * 4;
    let app = app();

    let one = peak_of(heatmap(app.clone(), body.clone())).await;
    eprintln!(
        "one 2048 px heatmap at stride 32: peak {:.1} MiB",
        one as f64 / MIB as f64
    );
    assert!(
        one < tile_bytes + inputs + 16 * MIB,
        "peak {} MiB for one heatmap",
        one / MIB
    );

    let requests = 4;
    let many = peak_of(async {
        let runs: Vec<_> = (0..requests)
            .map(|_| tokio::spawn(heatmap(app.clone(), body.clone())))
            .collect();
        for run in runs {
            run.await.unwrap();
        }
    })
    .await;
    eprintln!(
        "{requests} at once: peak {:.1} MiB",
        many as f64 / MIB as f64
    );
    assert!(
        many < requests * tile_bytes + inputs + 16 * MIB,
        "peak {} MiB for {requests} heatmaps at once",
        many / MIB
    );
}
