# 0003. Dynamic batching with worker-owned sessions

Date: 2026-10-18. Status: accepted.

## Context

A convolutional network on CPU does far more work per second on a batch than on single patches, but clients send
one or a few patches per request. ONNX Runtime sessions in ort 2.0 need exclusive access to run.

## Decision

- Requests enqueue jobs on a bounded tokio channel. A collector takes the first job, then more until the batch
  holds `MAX_BATCH` patches or `MAX_WAIT_US` has passed, and hands the batch to a pool of `WORKERS` OS threads.
- Each worker owns one session (no lock during inference) with `INTRA_THREADS` threads; workers x intra threads
  is sized to the CPU.
- A job never splits across batches; a tile larger than `MAX_BATCH` windows is split by the caller into jobs that
  run in parallel.
- Back-pressure: when every worker is busy the collector blocks, the queue fills, and callers wait (and time out
  at 20 s with a 503) instead of the process growing without bound.
- Metrics expose batch size, queue wait and model time, and an alert fires when queue wait grows.

## Consequences

- At low load a request waits at most `MAX_WAIT_US` (2 ms by default) for company; the load test measures the
  throughput gained against that latency cost.
- Tests check batching actually happens under concurrency, that each caller gets its own results, and that model
  errors reach every waiting caller.
