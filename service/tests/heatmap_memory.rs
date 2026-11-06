//! Peak heap of the largest heatmaps the API accepts, measured with a counting global allocator.
//!
//! A 2048x2048 tile at stride 32 has 3,844 windows: 425 MB of model inputs if they were laid out at once. The
//! service lays them out one batch at a time under the in-flight budget, so the peak is the decoded tiles plus
//! that budget and the batch each worker copies its jobs into, however many requests run together; and a request
//! waiting for admission has not decoded its tile yet.

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
const TILE: usize = 2048 * 2048 * 3;
const SLACK: usize = 16 * MIB;
const BUDGET: usize = 512;
const MAX_BATCH: usize = 32;
const WORKERS: usize = 2;

/// The fake model, slowed so that (in release builds) windows are laid out faster than they are scored and the
/// budget fills, as it does with the real model.
struct Paced;
impl Classifier for Paced {
    fn predict(&mut self, batch: &[f32], n: usize) -> anyhow::Result<Vec<f32>> {
        std::thread::sleep(Duration::from_millis(2));
        FakeClassifier.predict(batch, n)
    }
}

fn app(budget: usize, max_pending_patches: usize) -> axum::Router {
    let batcher = Batcher::spawn(
        BatchConfig {
            max_batch: MAX_BATCH,
            max_wait: Duration::from_micros(500),
            workers: WORKERS,
            max_inflight_patches: budget,
            max_pending_patches,
            admit_timeout: Duration::from_mins(1),
        },
        |_| Ok(Box::new(Paced) as Box<dyn Classifier>),
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
    });
    let auth = Arc::new(Auth::new(vec![("viewer".into(), KEY.into())], 1_000_000));
    router(state, auth, &[])
}

/// Model inputs a budget allows, plus one batch per worker that the worker copies them into.
fn inputs(budget: usize) -> usize {
    (budget + WORKERS * MAX_BATCH) * PATCH_LEN * 4
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

async fn at_once(app: axum::Router, bodies: Vec<String>) {
    let runs: Vec<_> = bodies
        .into_iter()
        .map(|body| tokio::spawn(heatmap(app.clone(), body)))
        .collect();
    for run in runs {
        run.await.unwrap();
    }
}

/// Peak heap above what was live when `run` started, in bytes.
async fn peak_of(run: impl Future<Output = ()>) -> usize {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    run.await;
    PEAK.load(Ordering::SeqCst) - base
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / MIB as f64
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

    let admits_all = app(BUDGET, 4 * 4096);
    let one = peak_of(heatmap(admits_all.clone(), body.clone())).await;
    eprintln!("one 2048 px heatmap at stride 32: peak {:.1} MiB", mib(one));
    assert!(one < TILE + inputs(BUDGET) + SLACK, "{:.1} MiB", mib(one));

    let many = peak_of(at_once(admits_all, vec![body.clone(); 4])).await;
    eprintln!("4 at once: peak {:.1} MiB", mib(many));
    assert!(
        many < 4 * TILE + inputs(BUDGET) + SLACK,
        "{:.1} MiB",
        mib(many)
    );

    // Room for one tile's windows and one batch of inputs at a time: the others wait for admission without a
    // decoded tile. At most two tiles are alive at once, as one request finishes its answer while the next starts.
    let queued = peak_of(at_once(app(MAX_BATCH, 4096), vec![body; 6])).await;
    eprintln!(
        "6 at once, admitted one at a time: peak {:.1} MiB",
        mib(queued)
    );
    assert!(
        queued < 2 * TILE + inputs(MAX_BATCH) + SLACK,
        "{:.1} MiB",
        mib(queued)
    );
}
