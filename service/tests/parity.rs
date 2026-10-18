//! The real ONNX model through the real service path must reproduce the probabilities Python's `onnxruntime` gave
//! for the same 16 `PCam` test patches (fixtures written by `python -m pcam.demo_assets`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pathology_service::batcher::{BatchConfig, Batcher};
use pathology_service::imaging::{decode_rgb, push_patch};
use pathology_service::model::{Classifier, ModelMeta, OnnxClassifier};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    file: String,
    probability: f32,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rust_service_matches_python_onnxruntime() {
    let model_dir = root().join("models/pcam/1");
    let fixtures_dir = root().join("service/tests/fixtures");
    // Both are committed (models/pcam/1, written by `make train`; fixtures by `make assets`): a missing file is a
    // failure, not a skip, so CI can never pass without running the real model.
    assert!(
        model_dir.join("model.onnx").exists(),
        "models/pcam/1/model.onnx is missing"
    );
    assert!(
        fixtures_dir.join("parity.json").exists(),
        "service/tests/fixtures/parity.json is missing"
    );
    let meta = ModelMeta::load(&model_dir.join("model.json")).unwrap();
    let fixtures: Vec<Fixture> =
        serde_json::from_str(&std::fs::read_to_string(fixtures_dir.join("parity.json")).unwrap())
            .unwrap();

    let env = Arc::new(ort::init().with_name("parity-test").build().unwrap());
    let path = model_dir.join("model.onnx");
    let batcher = Batcher::spawn(
        BatchConfig {
            max_batch: 8,
            max_wait: Duration::from_millis(1),
            workers: 1,
            queue: 32,
        },
        move |_| {
            Ok(
                Box::new(OnnxClassifier::load(&env, &path, meta.clone(), 2)?)
                    as Box<dyn Classifier>,
            )
        },
    )
    .unwrap();

    let mut inputs = Vec::new();
    for f in &fixtures {
        let bytes = std::fs::read(fixtures_dir.join(&f.file)).unwrap();
        push_patch(&decode_rgb(&bytes, 96).unwrap(), &mut inputs).unwrap();
    }
    let got = batcher.predict_many(inputs, fixtures.len()).await.unwrap();
    let max_diff = fixtures
        .iter()
        .zip(&got)
        .map(|(f, p)| (f.probability - p).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "parity: {} patches, max |rust - python| = {max_diff:e}",
        fixtures.len()
    );
    for (f, p) in fixtures.iter().zip(&got) {
        assert!(
            (f.probability - p).abs() < 1e-4,
            "{}: python {} rust {}",
            f.file,
            f.probability,
            p
        );
    }
}
