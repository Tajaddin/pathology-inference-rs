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

- At low load a request waits at most `MAX_WAIT_US` for company. The load test measured that cost
  (`reports/k6/summary.json`, 4 CPUs each): with the first setting (2 workers x 2 threads, 2 ms) single-patch p99
  was 10-34 ms against the Python baseline's 6-9 ms. With 4 workers x 1 thread and 0.5 ms (now the default) it is
  8-12 ms up to 400 requests/s, and at 800 requests/s the service keeps up (p99 154 ms, nothing dropped) where the
  Python baseline saturates at 486 requests/s with a p99 of 9.2 s. One intra-op thread per worker beats two:
  ONNX Runtime's intra-op parallelism gains little on batches this small.
- Tests check batching actually happens under concurrency, that each caller gets its own results, and that model
  errors reach every waiting caller.
