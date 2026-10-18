# 0002. Serve the model from Rust with ONNX Runtime

Date: 2026-10-18. Status: accepted.

## Context

Pathology viewers score thousands of patches per slide region; the service should use CPUs well, start fast,
ship as a small image, and avoid the Python GIL and per-process model copies.

## Decision

- Export the Keras model to ONNX (tf2onnx, opset 17, through a float32 SavedModel: [ADR 0005](0005-onnx-export-through-a-float32-savedmodel.md)) and check parity with TensorFlow on 2,048 test patches.
- Serve with Rust: axum 0.8 on tokio, ONNX Runtime through the `ort` crate (2.0.0-rc.14, the version its docs
  recommend), images decoded with the `image` crate. Preprocessing in Rust is only decoding and pixel layout,
  because normalisation lives in the graph.
- A Python service (FastAPI + onnxruntime, same API) is kept as a baseline and load-tested the same way, so the
  language choice is measured, not assumed.

## Alternatives

- TF Serving: proven, but a 500 MB+ image, no request-level control over batching and image decoding, and a
  second protocol hop for decoding.
- `tract` (pure Rust ONNX): no native dependency, simpler builds, but fewer operators and no production track
  record for this kind of model.

## Consequences

- The build downloads ONNX Runtime binaries (`download-binaries`, copied next to the executable); CI caches them.
- `Session::run` needs `&mut self` in ort 2.0, which shapes the batching design (ADR 0003).
