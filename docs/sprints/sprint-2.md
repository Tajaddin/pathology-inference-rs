# Sprint 2 retro: Rust inference service

Dates: 2026-10-18. Milestone: [Sprint 2 - Rust inference service](https://github.com/Tajaddin/pathology-inference-rs/milestone/2).

**Goal:** an authenticated Rust service (axum + ONNX Runtime) with dynamic batching, patch and tile endpoints,
metrics and readiness, and a Python baseline with the same contract. **Met.**

## Delivered

| Issue | Points | Result |
|---|---|---|
| #6 classify endpoint | 5 | 1-64 base64 PNG/JPEG patches, constant-time API keys, per-client rate limit counted in patches, errors that never echo input |
| #7 dynamic batching | 5 | bounded queue, batch up to `MAX_BATCH` or `MAX_WAIT_US`, one ONNX Runtime session per worker thread |
| #8 heatmap endpoint | 3 | one tile up to 2048 px, sliding windows at stride 32+, split across batches |
| #9 observability | 3 | Prometheus histograms (request, queue wait, inference, decode, batch size), JSON logs, `/health` and `/ready` after a warm-up batch |
| #10 Python baseline | 3 | FastAPI + onnxruntime with the same `/v1/classify` contract, 7 tests |

19 of 19 points done.

## Measured

- 10 unit tests (batching, imaging, auth, config) and 6 API tests with a deterministic fake model.
- **Parity: the Rust service reproduces the probabilities Python's `onnxruntime` gave for 16 real test patches to
  within 1.5e-8**, through its own PNG decoding and batching; CI runs this against the committed model.
- Service image 180 MB (Debian trixie, non-root); the Python baseline image is 977 MB.

## What went well

- Worker-owned sessions kept the batcher free of locks: `Session::run` needs `&mut self` in `ort` 2.0, and each
  worker owns one, so the only shared structure is the queue.

## What did not

- The first compile failed with 15 errors: `ort` 2.0.0-rc.14's session-builder errors carry the builder and are
  not `Send`, so `?` into `anyhow` failed; and `predict_many` spawned futures that borrowed the batcher. Fixed with
  `map_err` per builder call and a cloned handle per spawned job; clippy (pedantic, warnings as errors) then found
  ten smaller issues.
- The Docker build failed twice at link time: the slim Rust image has no `libstdc++`, and the static ONNX Runtime
  that `ort` downloads needs glibc 2.38+ and a newer `libstdc++` than Debian bookworm has. Both stages moved to
  Debian trixie.

## Carried into Sprint 3

- Nothing open; the load comparison with the baseline is #14 in Sprint 4.
