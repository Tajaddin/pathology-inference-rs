#!/usr/bin/env bash
# Rust service vs Python baseline under the same CPU budget (bench/compose.bench.yml), k6 at fixed arrival rates.
#
#   docker compose -f docker-compose.yml -f bench/compose.bench.yml --profile compare up -d --build --wait service baseline
#   bash bench/run_compare.sh            results in reports/k6/<label>-r<rate>-p<patches>.json, one line each on stdout
#
# Environment: DURATION (30s), PLAN (rate:patches pairs, space-separated), TARGETS (label:host pairs).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
duration=${DURATION:-30s}
plan=${PLAN:-"100:1 200:1 400:1 800:1 25:16 50:16 100:16"}
targets=${TARGETS:-"rust_4x1:service python:baseline"}
mkdir -p "$root/reports/k6"

for target in $targets; do
  label=${target%%:*}
  host=${target#*:}
  for step in $plan; do
    rate=${step%%:*}
    patches=${step#*:}
    docker run --rm --network pathology_default \
      -v "$root/bench/k6:/scripts:ro" -v "$root/service/tests/fixtures:/fixtures:ro" \
      -v "$root/reports/k6:/out" \
      -e FIXTURES_DIR=/fixtures -e TARGET="http://$host:8080" -e LABEL="$label" -e RATE="$rate" -e PATCHES="$patches" -e DURATION="$duration" \
      grafana/k6:1.3.0 run --quiet /scripts/classify.js
    sleep 5 # let queues drain between steps
  done
done
