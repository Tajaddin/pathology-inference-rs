"""Demo tiles for the viewer and fixtures for the Rust parity test, all from the PCam *test* split.

    python -m pcam.demo_assets --data ~/data/pcam --model ../models/pcam/1 --viewer ../web/viewer/public/demo \
        --fixtures ../service/tests/fixtures

- Three 768x768 PNG tiles, each an 8x8 grid of real test patches, with their labels (tiles.json). The mix of
  tumour patches differs per tile (mostly normal, mixed, mostly tumour). Patches are placed side by side, so a
  tile is a mosaic, not a contiguous slide region; the labels say exactly what each cell is.
- 16 individual test patches as PNG plus the probabilities onnxruntime gives for them (parity.json), which the
  Rust integration test reproduces through the service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pcam import data
from pcam.metrics import sigmoid

GRID = 8
MIXES = {"Mostly normal (6 of 64 tumour)": 6, "Mixed (32 of 64 tumour)": 32, "Mostly tumour (58 of 64 tumour)": 58}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()

    x, y = data.load(args.data, "test")
    rng = np.random.default_rng(96)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    args.viewer.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i, (name, n_pos) in enumerate(MIXES.items()):
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, GRID * GRID - n_pos, replace=False)]
        )
        rng.shuffle(idx)
        mosaic = np.zeros((GRID * 96, GRID * 96, 3), dtype=np.uint8)
        for k, j in enumerate(idx):
            r, c = divmod(k, GRID)
            mosaic[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96] = x[j]
        file = f"tile-{i + 1}.png"
        Image.fromarray(mosaic).save(args.viewer / file, optimize=True)
        tiles.append(
            {
                "name": name,
                "file": file,
                "grid": GRID,
                "labels": [int(y[j]) for j in idx],
                "test_indices": [int(j) for j in idx],
            }
        )
    (args.viewer / "tiles.json").write_text(json.dumps(tiles) + "\n", encoding="utf-8")

    import onnxruntime as ort

    meta = json.loads((args.model / "model.json").read_text())
    session = ort.InferenceSession(str(args.model / "model.onnx"), providers=["CPUExecutionProvider"])
    args.fixtures.mkdir(parents=True, exist_ok=True)
    picks = np.concatenate([rng.choice(pos, 8, replace=False), rng.choice(neg, 8, replace=False)])
    batch = np.asarray(x[np.sort(picks)]).astype(np.float32)
    logits = session.run([meta["output_name"]], {meta["input_name"]: batch})[0].reshape(-1)
    probs = sigmoid(logits / meta["temperature"])
    fixtures = []
    for k, (j, p) in enumerate(zip(np.sort(picks), probs, strict=True)):
        file = f"patch-{k:02d}.png"
        Image.fromarray(np.asarray(x[j])).save(args.fixtures / file)
        fixtures.append({"file": file, "test_index": int(j), "label": int(y[j]), "probability": float(p)})
    (args.fixtures / "parity.json").write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
    print(f"{len(tiles)} tiles, {len(fixtures)} parity fixtures")


if __name__ == "__main__":
    main()
