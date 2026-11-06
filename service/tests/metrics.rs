//! Request metrics are labelled by route template, and `/metrics` is served only by the internal metrics router.
//! (Its own test binary: it installs the process-wide Prometheus recorder.)

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use pathology_service::api::{AppState, metrics_router, router};
use pathology_service::auth::Auth;
use pathology_service::batcher::{BatchConfig, Batcher};
use pathology_service::model::{Classifier, FakeClassifier, ModelMeta};
use tower::ServiceExt;

const KEY: &str = "test-key-viewer-0000001";

fn app() -> axum::Router {
    let batcher = Batcher::spawn(
        BatchConfig {
            max_batch: 8,
            max_wait: Duration::from_millis(1),
            workers: 1,
            max_inflight_patches: 8,
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
        max_windows: 400,
        max_tile_side: 1024,
        min_stride: 32,
    });
    let auth = Arc::new(Auth::new(vec![("viewer".into(), KEY.into())], 6000));
    router(state, auth, &[])
}

async fn get(app: axum::Router, path: &str, key: Option<&str>) -> (StatusCode, String) {
    let mut req = Request::get(path);
    if let Some(k) = key {
        req = req.header("x-api-key", k);
    }
    let res = app.oneshot(req.body(Body::empty()).unwrap()).await.unwrap();
    let status = res.status();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&body).into_owned())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arbitrary_paths_add_no_series_and_metrics_are_not_public() {
    let prometheus = pathology_service::install_metrics().unwrap();
    let app = app();
    for i in 0..200 {
        get(app.clone(), &format!("/v1/x{i}"), None).await;
        get(app.clone(), &format!("/v1/model/{i}"), Some(KEY)).await;
    }
    assert_eq!(
        get(app.clone(), "/v1/model", Some(KEY)).await.0,
        StatusCode::OK
    );

    let text = prometheus.render();
    let routes: BTreeSet<&str> = text
        .lines()
        .filter(|l| l.starts_with("pathology_requests_total{"))
        .filter_map(|l| l.split("route=\"").nth(1)?.split('"').next())
        .collect();
    assert_eq!(routes, BTreeSet::from(["/v1/model", "unmatched"]));

    let (status, body) = get(app, "/metrics", None).await;
    assert_ne!(status, StatusCode::OK);
    assert!(!body.contains("pathology_"));
    let (status, body) = get(metrics_router(prometheus), "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains(r#"pathology_requests_total{route="/v1/model",status="200"} 1"#));
}
