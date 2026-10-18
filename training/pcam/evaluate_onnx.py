"""Score the test split with the exported ONNX model, the one the service runs, and compare it with TensorFlow.

    python -m pcam.evaluate_onnx --data ~/data/pcam --model ../models/pcam/1 --reports ../reports

The training run reports test metrics from the TensorFlow model (trained in mixed precision); the service runs the
float32 ONNX export. This scores all 32,768 test patches through ONNX Runtime on CPU with the same temperature and
thresholds, and writes reports/onnx-test.json. It reads the test split again but changes no choice: the model,
temperature and thresholds were fixed before.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pcam import data, metrics


def score(model_dir: Path, x: np.ndarray, batch: int = 256) -> tuple[np.ndarray, float]:
    import onnxruntime as ort

    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count() or 4
    session = ort.InferenceSession(str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    logits = np.concatenate(
        [
            session.run([meta["output_name"]], {meta["input_name"]: np.asarray(x[i : i + batch], dtype=np.float32)})[
                0
            ].reshape(-1)
            for i in range(0, len(x), batch)
        ]
    )
    return metrics.sigmoid(logits.astype(np.float64) / meta["temperature"]), time.perf_counter() - start


def compare(y: np.ndarray, p_onnx: np.ndarray, p_tf: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    diff = np.abs(p_onnx - p_tf)
    return {
        "patches": len(y),
        "auc": metrics.auc(y, p_onnx),
        "auc_ci95": list(metrics.bootstrap_ci(y, p_onnx)),
        "ece": metrics.ece(y, p_onnx),
        "operating_points": {k: metrics.operating_point(y, p_onnx, t) for k, t in thresholds.items()},
        "vs_tensorflow": {
            "auc_tensorflow": metrics.auc(y, p_tf),
            "max_abs_probability_diff": float(diff.max()),
            "mean_abs_probability_diff": float(diff.mean()),
            "decisions_changed_at_threshold": {
                k: int(np.sum((p_onnx >= t) != (p_tf >= t))) for k, t in thresholds.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()

    x, y = data.load(args.data, "test")
    run = json.loads((args.reports / "metrics.json").read_text(encoding="utf-8"))
    p_tf = np.load(args.reports / "test_probabilities.npy").astype(np.float64)
    p_onnx, seconds = score(args.model, x)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime": f"onnxruntime CPU, {os.cpu_count()} threads, batches of 256",
        "seconds": round(seconds, 1),
        "patches_per_second": round(len(y) / seconds, 1),
        **compare(y, p_onnx, p_tf, run["thresholds"]),
    }
    (args.reports / "onnx-test.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("auc", "ece", "patches_per_second")} | report["vs_tensorflow"]))


if __name__ == "__main__":
    main()
