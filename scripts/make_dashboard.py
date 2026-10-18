"""Generate observability/grafana/dashboards/pathology.json and its provisioning files."""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
ROOT = Path(__file__).resolve().parents[1] / "observability" / "grafana"


def panel(pid: int, title: str, exprs: list[tuple[str, str]], x: int, y: int, w: int = 12, h: int = 8,
          unit: str = "short", kind: str = "timeseries") -> dict:
    return {
        "id": pid, "title": title, "type": kind, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": legend}
                    for i, (e, legend) in enumerate(exprs)],
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
    }


def q(metric: str, quantile: float, by: str = "") -> str:
    group = f"le{', ' + by if by else ''}"
    return f"histogram_quantile({quantile}, sum by ({group}) (rate({metric}_bucket[2m])))"


def main() -> None:
    panels = [
        panel(1, "Requests per second by route and status",
              [("sum by (route, status) (rate(pathology_requests_total[1m]))", "{{route}} {{status}}")], 0, 0, unit="reqps"),
        panel(2, "Request latency p50 / p99 by route",
              [(q("pathology_request_seconds", 0.5, "route"), "p50 {{route}}"),
               (q("pathology_request_seconds", 0.99, "route"), "p99 {{route}}")], 12, 0, unit="s"),
        panel(3, "Patches scored per second", [("sum by (route) (rate(pathology_patches_total[1m]))", "{{route}}")],
              0, 8, unit="short"),
        panel(4, "Batch size (p50 / p95): dynamic batching at work",
              [(q("pathology_batch_size", 0.5), "p50"), (q("pathology_batch_size", 0.95), "p95")], 12, 8),
        panel(5, "Model time per batch vs queue wait (p95)",
              [(q("pathology_inference_seconds", 0.95), "inference p95"),
               (q("pathology_queue_wait_seconds", 0.95), "queue wait p95")], 0, 16, unit="s"),
        panel(6, "Errors by code", [("sum by (code) (rate(pathology_errors_total[5m]))", "{{code}}")], 12, 16, unit="reqps"),
    ]
    dashboard = {"uid": "pathology", "title": "Pathology inference service", "schemaVersion": 41, "version": 1,
                 "refresh": "10s", "time": {"from": "now-30m", "to": "now"}, "panels": panels, "timezone": "utc"}
    (ROOT / "dashboards").mkdir(parents=True, exist_ok=True)
    (ROOT / "dashboards" / "pathology.json").write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8",
                                                         newline="\n")
    (ROOT / "provisioning" / "datasources").mkdir(parents=True, exist_ok=True)
    (ROOT / "provisioning" / "dashboards").mkdir(parents=True, exist_ok=True)
    (ROOT / "provisioning" / "datasources" / "prometheus.yml").write_text(
        "apiVersion: 1\ndatasources:\n  - name: Prometheus\n    uid: prometheus\n    type: prometheus\n"
        "    access: proxy\n    url: http://prometheus:9090\n    isDefault: true\n", encoding="utf-8", newline="\n")
    (ROOT / "provisioning" / "dashboards" / "dashboards.yml").write_text(
        "apiVersion: 1\nproviders:\n  - name: pathology\n    type: file\n    options:\n"
        "      path: /var/lib/grafana/dashboards\n", encoding="utf-8", newline="\n")
    print("dashboard written")


if __name__ == "__main__":
    main()
