# Runbook

Alerts in `observability/alert_rules.yml` (Prometheus, local stack) and `infra/terraform/cicd_and_cost.tf` (Cloud
Monitoring, GCP) link here. Metrics come from the service's `/metrics` on the internal port `METRICS_PORT` (9464),
which is not published: Cloud Run routes only port 8080, and compose leaves it on the internal network.

## Errors

**PathologyErrorRate** (more than 1% of requests answer 5xx for 5 minutes) or **PathologyInferenceFailures**
(requests get 503 `unavailable`).

1. `GET /ready`: false means the model did not load or the warm-up batch failed. The service log says which file
   (`MODEL_PATH`, `MODEL_META_PATH`) and why. `/health` and `/ready` both answer 503 `inference worker stopped`
   once a worker thread has died (a bug: the log has the panic), so the liveness probe restarts the instance.
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

## Overload

`503` with code `overloaded` and `Retry-After: 1`: the admitted work (`MAX_PENDING_PATCHES`) stayed full for
`ADMIT_TIMEOUT_MS`, so the request was refused before any work was done. It shows in `pathology_errors_total`
with `code="overloaded"` and in the 5xx rate. Add replicas or workers as for Queueing. Under overload an admitted
request waits about `MAX_PENDING_PATCHES` divided by the patches scored per second; lower it (with
`MAX_WINDOWS_PER_REQUEST`, which it must not be below) to answer sooner. `MAX_INFLIGHT_PATCHES` bounds memory
instead: each patch is 108 KiB of model inputs.

## Rate limiting

`429` with `Retry-After` is the per-client limit (`RATE_LIMIT_PER_MINUTE`, counted in patches, so a 64-patch
request costs 64). Clients in `CLIENT_LIMITS` (the viewer) have their own quota and heatmap window cap instead.
`pathology_rate_limited_total` by client shows who hits it.

A heatmap over the window cap is charged the cap before it is refused with 422. The quota is charged before
admission, so a request refused with 503 `overloaded` has been charged too; charging later would let a client over
its quota wait in the admission queue. Request bodies (up to 16 MB) are read and base64-decoded before the charge,
and a malformed one is refused without it: the body limit and the instance's request concurrency (64 on Cloud
Run), not the quota, bound what those cost.

## Rolling out a model

1. `make train` (or a new run), `make assets` (new parity fixtures), `cargo test` (the Rust parity test must pass
   against the new ONNX file), `make model-card`.
2. Upload `models/pcam/<n>/` to the models bucket and set `model_version` in Terraform (or `MODEL_PATH`).
3. Watch `/ready` and the error rate for the first minutes.
