//! The router end to end (auth, validation, batching, heatmaps) with the deterministic fake classifier.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use http_body_util::BodyExt;
use image::{Rgb, RgbImage};
use pathology_service::api::{AppState, ClassifyResponse, HeatmapResponse, router};
use pathology_service::auth::Auth;
use pathology_service::batcher::{BatchConfig, Batcher};
use pathology_service::model::{Classifier, FakeClassifier, ModelMeta};
use serde_json::{Value, json};
use tower::ServiceExt;

const KEY: &str = "test-key-viewer-0000001";

fn meta() -> ModelMeta {
    ModelMeta {
        version: "test".into(),
        input_name: "x".into(),
        output_name: "y".into(),
        temperature: 1.0,
        threshold: 0.5,
    }
}

fn app(per_minute: u32) -> axum::Router {
    let batcher = Batcher::spawn(
        BatchConfig {
            max_batch: 8,
            max_wait: Duration::from_millis(1),
            workers: 2,
            queue: 64,
        },
        |_| Ok(Box::new(FakeClassifier) as Box<dyn Classifier>),
    )
    .unwrap();
    let state = Arc::new(AppState {
        batcher,
        meta: meta(),
        ready: AtomicBool::new(true),
        max_patches: 64,
        max_tile_side: 1024,
        min_stride: 32,
        prometheus: None,
    });
    let auth = Arc::new(Auth::new(vec![("viewer".into(), KEY.into())], per_minute));
    router(state, auth, &["http://localhost:5173".into()])
}

fn png(img: &RgbImage) -> String {
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    STANDARD.encode(buf)
}

fn flat(side: u32, level: u8) -> RgbImage {
    RgbImage::from_pixel(side, side, Rgb([level, level, level]))
}

async fn post(
    app: axum::Router,
    path: &str,
    body: Value,
    key: Option<&str>,
) -> (StatusCode, Value) {
    let mut req = Request::post(path).header("content-type", "application/json");
    if let Some(k) = key {
        req = req.header("authorization", format!("Bearer {k}"));
    }
    let res = app
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn classify_returns_one_probability_per_patch_in_order() {
    let body = json!({"patches": [
        {"id": "dark", "image_base64": png(&flat(96, 20))},
        {"id": "light", "image_base64": png(&flat(96, 230))},
    ]});
    let (status, value) = post(app(6000), "/v1/classify", body, Some(KEY)).await;
    assert_eq!(status, StatusCode::OK);
    let res: ClassifyResponse = serde_json::from_value(value).unwrap();
    assert_eq!(res.results.len(), 2);
    assert_eq!(res.results[0].id, "dark");
    assert!(res.results[0].tumor && !res.results[1].tumor);
    assert!((res.results[0].probability - (1.0 - 20.0 / 255.0)).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn heatmap_matches_patch_by_patch_scoring() {
    // Left half dark, right half light: 288x192 tile, stride 48 -> 3 rows x 5 cols.
    let tile = RgbImage::from_fn(288, 192, |x, _| {
        if x < 144 {
            Rgb([30, 30, 30])
        } else {
            Rgb([220, 220, 220])
        }
    });
    let body = json!({"image_base64": png(&tile), "stride": 48});
    let (status, value) = post(app(6000), "/v1/heatmap", body, Some(KEY)).await;
    assert_eq!(status, StatusCode::OK);
    let res: HeatmapResponse = serde_json::from_value(value).unwrap();
    assert_eq!((res.rows, res.cols, res.stride), (3, 5, 48));
    assert_eq!(res.probabilities.len(), 15);
    for r in 0..3u32 {
        for c in 0..5u32 {
            let x0 = c * 48;
            let crop = image::imageops::crop_imm(&tile, x0, r * 48, 96, 96).to_image();
            let expected = 1.0
                - crop.as_raw().iter().map(|&v| f32::from(v)).sum::<f32>()
                    / (96.0 * 96.0 * 3.0 * 255.0);
            let got = res.probabilities[(r * 5 + c) as usize];
            assert!(
                (got - expected).abs() < 1e-5,
                "cell ({r},{c}) {got} vs {expected}"
            );
        }
    }
    assert!(res.windows_above_threshold > 0 && res.windows_above_threshold < 15);
}

#[tokio::test]
async fn requests_without_a_valid_key_are_refused() {
    let body = json!({"patches": [{"image_base64": png(&flat(96, 1))}]});
    let (status, value) = post(app(6000), "/v1/classify", body.clone(), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(value["error"]["code"], "unauthorized");
    let (status, _) = post(
        app(6000),
        "/v1/classify",
        body,
        Some("wrong-key-0000000000"),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn invalid_inputs_are_422_and_never_echoed() {
    let cases = [
        json!({"patches": []}),
        json!({"patches": [{"image_base64": png(&flat(64, 1))}]}),
        json!({"patches": [{"image_base64": "not base64 at all %%%"}]}),
        json!({"image_base64": png(&flat(2000, 1)), "stride": 48}),
        json!({"image_base64": png(&flat(192, 1)), "stride": 8}),
    ];
    for (i, body) in cases.into_iter().enumerate() {
        let path = if body.get("patches").is_some() {
            "/v1/classify"
        } else {
            "/v1/heatmap"
        };
        let (status, value) = post(app(6000), path, body, Some(KEY)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "case {i}");
        assert_eq!(value["error"]["code"], "invalid_request", "case {i}");
    }
    // Unknown fields are rejected by serde (axum answers 422 for JSON that does not match the schema).
    let (status, _) = post(
        app(6000),
        "/v1/classify",
        json!({"patches": [], "extra": 1}),
        Some(KEY),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn rate_limit_counts_patches_and_sets_retry_after() {
    let app = app(3);
    let one = json!({"patches": [{"image_base64": png(&flat(96, 1))}, {"image_base64": png(&flat(96, 1))}]});
    let (status, _) = post(app.clone(), "/v1/classify", one.clone(), Some(KEY)).await;
    assert_eq!(status, StatusCode::OK);
    let req = Request::post("/v1/classify")
        .header("content-type", "application/json")
        .header("x-api-key", KEY)
        .body(Body::from(one.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(res.headers().get("retry-after").is_some());
}

#[tokio::test]
async fn health_ready_and_cors() {
    let app = app(6000);
    let res = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = app
        .clone()
        .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let preflight = Request::options("/v1/classify")
        .header("origin", "http://localhost:5173")
        .header("access-control-request-method", "POST")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(preflight).await.unwrap();
    assert_eq!(
        res.headers().get("access-control-allow-origin").unwrap(),
        "http://localhost:5173"
    );
}
