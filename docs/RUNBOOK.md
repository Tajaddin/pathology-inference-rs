# Runbook

Alerts in `observability/alert_rules.yml` (Prometheus, local stack) and `infra/terraform/cicd_and_cost.tf` (Cloud
Monitoring, GCP) link here. Metrics come from the service's `/metrics`.

## Errors

**PathologyErrorRate** (more than 1% of requests answer 5xx for 5 minutes) or **PathologyInferenceFailures**
(requests get 503 `unavailable`).

1. `GET /ready`: false means the model did not load or the warm-up batch failed. The service log says which file
   (`MODEL_PATH`, `MODEL_META_PATH`) and why.
2. `pathology_errors_total` by `code`: `unavailable` is inference (ONNX Runtime returned an error or a worker
   stopped), `internal` is a bug; neither should happen in normal operation.
3. A new model version: check `models/pcam/<version>/model.json` names the ONNX graph's input and output
   (`input_name`, `output_name`); a mismatch fails every batch. Roll back by pointing `MODEL_PATH` at the previous
   version (Cloud Storage keeps 5 old versions).

## Latency

**PathologyClassifyLatency**: `/v1/classify` p99 above 200 ms for 10 minutes.

- If `pathology_queue_wait_seconds` is high too, the workers are saturated: see Queueing.
- If queue wait is low but `pathology_inference_seconds` grew, batches got larger (`pathology_batch_size`) or the
  instance has fewer CPUs than `WORKERS x INTRA_THREADS` assumes. Keep `WORKERS x INTRA_THREADS` at or below the
  CPU count.
- `pathology_decode_seconds` high means clients send large or slow-to-decode images (JPEG at 96x96 is cheap).

## Queueing

**PathologyQueueing**: p95 wait for a worker above 50 ms. Throughput is capped by CPU: add workers (when CPUs
are idle), add replicas (Cloud Run scales on concurrency), or lower `MAX_WAIT_US` if traffic is light and the
wait is the batching window itself.

## Rate limiting

`429` with `Retry-After` is the per-client limit (`RATE_LIMIT_PER_MINUTE`, counted in patches, so a 64-patch
request costs 64). `pathology_rate_limited_total` by client shows who hits it.

## Rolling out a model

1. `make train` (or a new run), `make assets` (new parity fixtures), `cargo test` (the Rust parity test must pass
   against the new ONNX file), `make model-card`.
2. Upload `models/pcam/<n>/` to the models bucket and set `model_version` in Terraform (or `MODEL_PATH`).
3. Watch `/ready` and the error rate for the first minutes.
