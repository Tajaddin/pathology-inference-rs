# 0003. Dynamic batching with worker-owned sessions

Date: 2026-10-18. Status: accepted; back-pressure amended 2026-11-06.

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
- Back-pressure: when every worker is busy the collector blocks and jobs wait in the queue. The queue alone did
  not bound memory (a 2048 px heatmap laid out all 3,844 windows, 425 MB, before queueing), so every job now
  takes permits from an in-flight budget counted in patches (`MAX_INFLIGHT_PATCHES`) before its inputs are laid
  out, and gives them back when it is done. Callers did not wait a bounded time either, so a request now first
  reserves all of its patches among `MAX_PENDING_PATCHES` (work admitted but not yet answered); one that gets no
  room within `ADMIT_TIMEOUT_MS` is refused with 503 `overloaded` and `Retry-After: 1`, and an admitted request
  waits behind at most that much work. One that times out at 20 s or whose client disconnects has its queued
  jobs dropped unscored.
- Metrics expose batch size, model time and the three waits (for admission, for budget, for a worker), and an
  alert fires when the wait for a worker grows.

## Consequences

- At low load a request waits at most `MAX_WAIT_US` for company. The load test measured that cost
  (`reports/k6/summary.json` as of da50c09, 4 CPUs each): with the first setting (2 workers x 2 threads, 2 ms) single-patch p99
  was 10-34 ms against the Python baseline's 6-9 ms. With 4 workers x 1 thread and 0.5 ms (now the default) it
  was 8-12 ms up to 400 requests/s, and at 800 requests/s the service kept up (p99 154 ms, nothing dropped) where
  the Python baseline saturated at 486 requests/s with a p99 of 9.2 s. One intra-op thread per worker beats two:
  ONNX Runtime's intra-op parallelism gains little on batches this small. These runs (2026-10-18) predate the
  in-flight budget and admission amended above; `git show da50c09:reports/k6/summary.json` holds their rows. The
  file itself now holds the 2026-11-07 repeat on the amended code, which the README describes: 795/s served at
  800 requests/s with p99 405 ms and 128 dropped, and within 2.5 ms of these figures up to 400/s.
- Admission is first come, first served: tokio's semaphore queues waiters in order, and the one at the head
  takes whatever room is free while it waits for the rest. A large request that does not fit therefore holds
  back every request behind it, even ones that would fit, until it is admitted or refused at
  `ADMIT_TIMEOUT_MS`. At the default 4,096 pending patches, a 3,844-window tile does this whenever more than 252
  are pending when it arrives. Letting small requests pass instead would starve large tiles under steady traffic.
  Only trusted keys can send tiles that large (the viewer's is capped at 256 windows), and a
  `MAX_PENDING_PATCHES` well above `MAX_WINDOWS_PER_REQUEST` makes it rarer, at the cost of a longer wait for
  admitted work under overload.
- Tests check batching actually happens under concurrency, that each caller gets its own results, that model
  errors reach every waiting caller, and that admission keeps arrival order.
