.PHONY: data prepare select train evaluate-onnx assets model-card test test-rust test-python test-web e2e lint check secret-scan \
        up down compare bench alerts-test tf-validate dashboard backlog

RAW ?= data/raw
PCAM ?= $(HOME)/data/pcam
LR ?= 1e-3
VENV ?= $(HOME)/venvs/pcam
# tensorflow[and-cuda] ships the CUDA libraries as wheels; TensorFlow only finds all of them on this path.
GPU_ENV = LD_LIBRARY_PATH=$$(ls -d $(VENV)/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':') TF_CPP_MIN_LOG_LEVEL=2
PY = cd training && $(GPU_ENV) $(VENV)/bin/python

data:               ## the nine PCam files from Zenodo, resumable, MD5-checked (~8.1 GB)
	bash scripts/download_pcam.sh $(RAW)

prepare:            ## checksums again, then memory-mapped arrays in $(PCAM) (~9 GB)
	$(PY) -c "from pathlib import Path; from pcam import data; data.prepare(Path('../$(RAW)'), Path('$(PCAM)'))"
	cp $(PCAM)/splits.json reports/splits.json

select:             ## learning-rate selection on the validation split only (writes reports/selection-*.json)
	for lr in 1e-3 3e-4 1e-4; do $(PY) -m pcam.train --data $(PCAM) --out /tmp/pcam-select --reports ../reports \
	  --lr $$lr --select-only; done

train:              ## train with the selected LR, score the test split once, export ONNX to models/pcam/1
	$(PY) -m pcam.train --data $(PCAM) --out ../models/pcam/1 --reports ../reports --lr $(LR)

evaluate-onnx:      ## the test split through the exported ONNX model the service runs (reports/onnx-test.json)
	$(PY) -m pcam.evaluate_onnx --data $(PCAM) --model ../models/pcam/1 --reports ../reports

assets:             ## demo tiles for the viewer and parity fixtures for the Rust tests (from the test split)
	$(PY) -m pcam.demo_assets --data $(PCAM) --model ../models/pcam/1 --viewer ../web/viewer/public/demo \
	  --fixtures ../service/tests/fixtures

model-card:
	$(PY) -m pcam.model_card --reports ../reports --splits ../reports/splits.json --out ../docs/MODEL-CARD.md \
	  --hardware "$$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) (WSL2)"

test-rust:
	cargo fmt --all --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked

test-python:
	cd training && uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest

test-web:
	cd web && npm run lint && npm run typecheck && npm test && npm run build

test: test-rust test-python test-web
	shellcheck scripts/*.sh && bats tests/bats

e2e:                ## Playwright against `make up`
	cd web && npm run e2e

secret-scan:        ## run before every push
	bash scripts/secret_scan.sh

up:                 ## the Rust service, the viewer, Prometheus and Grafana
	docker compose up -d --build --wait --wait-timeout 600 service viewer prometheus grafana

down:
	docker compose down -v

compare:            ## the service and the Python baseline (:8090), each limited to 4 CPUs (bench/compose.bench.yml)
	docker compose -f docker-compose.yml -f bench/compose.bench.yml --profile compare up -d --build --wait service baseline

bench:              ## k6 at fixed rates against both (after `make compare`); reports/k6/ and a summary table
	bash bench/run_compare.sh && python3 scripts/k6_table.py

alerts-test:
	docker run --rm -v "$(CURDIR)/observability:/w" -w /w --entrypoint promtool prom/prometheus:v3.5.0 test rules alerts_test.yml

tf-validate:
	terraform -chdir=infra/terraform fmt -check -recursive && terraform -chdir=infra/terraform init -backend=false \
	  -input=false && terraform -chdir=infra/terraform validate

dashboard:
	python scripts/make_dashboard.py

backlog:            ## labels, milestones and issues on GitHub from docs/agile/backlog.json
	python scripts/agile/sync_backlog.py
