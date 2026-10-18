"""Smoke test a deployed service: health, model info, and the 16 parity patches against their recorded probabilities.

    python scripts/cloud_smoke.py https://<service>.run.app --key-file ~/.config/pathology/api-keys --label warm

The first request's latency is the cold start only if the service had no instances; check that in Cloud Monitoring
(run.googleapis.com/container/instance_count) before running with --label cold.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "service" / "tests" / "fixtures"
OUT = Path(__file__).resolve().parents[1] / "reports" / "cloud-run"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--key-file", type=Path, required=True, help="client:key pairs, comma separated")
    parser.add_argument("--client", default="viewer")
    parser.add_argument("--label", choices=["cold", "warm"], default="warm")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    pairs = dict(p.split(":", 1) for p in args.key_file.expanduser().read_text().strip().split(","))
    url, key = args.url.rstrip("/"), pairs[args.client]

    def call(method: str, path: str, body: object = None) -> tuple[int, float, object]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, method=method, data=data)  # noqa: S310 (https URL from the operator)
        req.add_header("content-type", "application/json")
        req.add_header("x-api-key", key)
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            raw = r.read()
            ms = round((time.perf_counter() - start) * 1000, 1)
            return r.status, ms, json.loads(raw) if raw[:1] in (b"{", b"[") else None

    expected = json.loads((FIXTURES / "parity.json").read_text())
    patches = [
        {"id": e["file"], "image_base64": base64.b64encode((FIXTURES / e["file"]).read_bytes()).decode()}
        for e in expected
    ]

    first_status, first_ms, _ = call("GET", "/health")
    _, _, model = call("GET", "/v1/model")
    timings = []
    for _ in range(args.repeats):
        status, ms, out = call("POST", "/v1/classify", {"patches": patches})
        assert status == 200
        timings.append(ms)
    got = {r["id"]: r["probability"] for r in out["results"]}
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": url,
        "label": args.label,
        "first_request": {"path": "/health", "status": first_status, "ms": first_ms},
        "model": model,
        "classify_16_patches_ms": timings,
        "max_abs_diff_vs_local": max(abs(got[e["file"]] - e["probability"]) for e in expected),
        "patches_matching_label": sum((got[e["file"]] >= out["threshold"]) == bool(e["label"]) for e in expected),
        "patches": len(expected),
        "note": "client on a home connection in Tampa, FL; service in us-central1; latencies include the network",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"smoke-{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
