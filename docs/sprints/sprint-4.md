# Sprint 4 retro: release

Dates: 2026-10-18. Milestone: [Sprint 4 - Release](https://github.com/Tajaddin/pathology-inference-rs/milestone/4).

**Goal:** one-command stack, load tests, Terraform for Cloud Run, CI green, README and model card. **Met**,
except applying the Terraform (no GCP account).

## Delivered

| Issue | Points | Result |
|---|---|---|
| #13 one command | 2 | `make up`: service, viewer, Prometheus and Grafana with health checks, up in about 20 s from cached images |
| #14 load tests | 2 | k6 at fixed arrival rates, Rust against the Python baseline under the same 4-CPU limit (`reports/k6/`) |
| #15 Terraform | 3 | Cloud Run with the model mounted from Cloud Storage, keys from Secret Manager, keyless deploys, budget and alerts; validated, tflint and trivy clean |
| #16 docs | 3 | README, architecture, model card generated from the reports, runbook, deploy guide, ADR 0005 |

10 of 10 points done.

## Measured (`reports/k6/summary.json`, one machine, each service limited to 4 CPUs, 30 s per step)

| One patch per request | Rust, first setting (2 x 2, 2 ms) | Rust, tuned (4 x 1, 0.5 ms) | Python baseline |
|---|---|---|---|
| 100 req/s, p99 | 9.8 ms | 7.6 ms | 6.3 ms |
| 400 req/s, p99 | 34.3 ms | 12.3 ms | 9.2 ms |
| 800 req/s offered | 772/s served, p99 863 ms, 395 dropped | **799/s served, p99 154 ms, 0 dropped** | 486/s served, p99 9.2 s, 8,657 dropped |

With 16 patches per request, overload (1,600 patches/s offered) gave 1,016 patches/s from the tuned Rust service
and 653 from Python; at 800 patches/s Python's p99 was lower (79 ms against 108 ms).

## What went well

- Measuring before concluding: the first configuration lost to Python at every load below saturation, and the
  dashboard (queue wait vs model time, batch size) showed why. One intra-op thread per worker and a shorter
  batching window fixed most of it; both runs are committed.

## What did not

- The first k6 run produced nothing: the fixtures were mounted inside the read-only scripts mount, which Docker
  cannot do. They have their own mount now.
- The Rust image needed Debian trixie and `g++` to link ONNX Runtime (Sprint 2), found only when the image was
  first built here; CI builds all three images on every push from now on.
- Python wins at light load by about a millisecond; the Rust service's case is throughput and tail latency under
  load, plus a 180 MB image against 977 MB. The README says so rather than claiming it is faster across the board.

## Still open (Backlog)

- #17 deploy to Cloud Run once a GCP project exists, then measure cold start and steady state there.
- #18 the GitHub Projects board (needs a token with the `project` scope).
