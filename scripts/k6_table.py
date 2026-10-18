"""Summarise reports/k6/*.json (k6 handleSummary output) as a Markdown table and reports/k6/summary.json.

    python scripts/k6_table.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

K6 = Path(__file__).resolve().parents[1] / "reports" / "k6"


def main() -> None:
    rows = []
    for f in sorted(K6.glob("*-r*-p*.json")):
        m = re.fullmatch(r"(\w+)-r(\d+)-p(\d+)\.json", f.name)
        if not m:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))["metrics"]
        dur = d["http_req_duration"]["values"]
        rows.append({
            "service": m.group(1), "rate": int(m.group(2)), "patches": int(m.group(3)),
            "requests": int(d["http_reqs"]["values"]["count"]),
            "achieved_rps": round(d["http_reqs"]["values"]["rate"], 1),
            "patches_per_s": round(d["http_reqs"]["values"]["rate"] * int(m.group(3)), 1),
            "failed_pct": round(100 * d["http_req_failed"]["values"]["rate"], 2),
            "dropped_iterations": int(d.get("dropped_iterations", {}).get("values", {}).get("count", 0)),
            "p50_ms": round(dur["med"], 1), "p95_ms": round(dur["p(95)"], 1), "p99_ms": round(dur["p(99)"], 1),
        })
    rows.sort(key=lambda r: (r["patches"], r["rate"], r["service"]))
    (K6 / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("| Patches per request | Offered rate | Service | Achieved | Patches/s | Failed | Dropped | p50 | p95 | p99 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['patches']} | {r['rate']}/s | {r['service']} | {r['achieved_rps']}/s | {r['patches_per_s']:,.0f} | "
              f"{r['failed_pct']}% | {r['dropped_iterations']} | {r['p50_ms']} ms | {r['p95_ms']} ms | {r['p99_ms']} ms |")


if __name__ == "__main__":
    main()
