//! The classifier behind the batcher: ONNX Runtime in production, a deterministic fake in tests.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};

use crate::imaging::{PATCH, PATCH_LEN};

/// What the training job wrote next to the ONNX file (`model.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub version: String,
    pub input_name: String,
    pub output_name: String,
    /// Temperature fitted on the validation split: probability = sigmoid(logit / temperature).
    pub temperature: f32,
    /// Operating threshold on the calibrated probability.
    pub threshold: f32,
}

impl ModelMeta {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Scores `n` patches laid out back to back in `batch` (NHWC, 0..255). Returns calibrated probabilities.
pub trait Classifier: Send + 'static {
    fn predict(&mut self, batch: &[f32], n: usize) -> Result<Vec<f32>>;
}

#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct OnnxClassifier {
    session: Session,
    meta: ModelMeta,
}

impl OnnxClassifier {
    pub fn load(
        env: &ort::environment::Environment,
        model: &Path,
        meta: ModelMeta,
        intra_threads: usize,
    ) -> Result<Self> {
        // Builder errors carry the builder back and are not Send, so they become plain messages here.
        let session = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("optimization level: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow!("intra-op threads: {e}"))?
            .commit_from_file(model)
            .with_context(|| format!("loading {}", model.display()))?;
        Ok(Self { session, meta })
    }
}

impl Classifier for OnnxClassifier {
    fn predict(&mut self, batch: &[f32], n: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            batch.len() == n * PATCH_LEN,
            "batch has {} values for {n} patches",
            batch.len()
        );
        let side = PATCH as usize;
        let view = ndarray::ArrayView4::from_shape((n, side, side, 3), batch)?;
        let outputs = self.session.run(
            ort::inputs![self.meta.input_name.as_str() => TensorRef::from_array_view(view)?],
        )?;
        let logits = outputs[self.meta.output_name.as_str()].try_extract_array::<f32>()?;
        let t = self.meta.temperature;
        Ok(logits.iter().map(|&z| sigmoid(z / t)).collect())
    }
}

/// Deterministic stand-in for tests: darker (more hematoxylin-stained) patches score higher.
pub struct FakeClassifier;

impl Classifier for FakeClassifier {
    fn predict(&mut self, batch: &[f32], n: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            batch.len() == n * PATCH_LEN,
            "batch has {} values for {n} patches",
            batch.len()
        );
        Ok(batch
            .chunks(PATCH_LEN)
            .map(|p| 1.0 - p.iter().sum::<f32>() / (PATCH_LEN as f32 * 255.0))
            .collect())
    }
}

/// Reshape helper used by the benchmarks.
#[must_use]
pub fn zeros(n: usize) -> Array4<f32> {
    Array4::zeros((n, PATCH as usize, PATCH as usize, 3))
}
