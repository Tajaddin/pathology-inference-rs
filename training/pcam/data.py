"""PatchCamelyon files: decompress, verify and convert to memory-mapped arrays.

PCam (Veeling et al., 2018; CC0, https://github.com/basveeling/pcam) has fixed splits: 262,144 training,
32,768 validation and 32,768 test patches of 96x96 RGB from Camelyon16 lymph-node slides. The label is whether
the centre 32x32 region contains tumour. The splits are used exactly as published; the test split is read only
by the final evaluation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")
EXPECTED = {"train": 262_144, "valid": 32_768, "test": 32_768}
URL = "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_{split}_{part}.h5.gz/content"
# MD5 checksums published by Zenodo for record 2546921 (https://zenodo.org/api/records/2546921).
MD5 = {
    "camelyonpatch_level_2_split_train_x.h5.gz": "1571f514728f59376b705fc836ff4b63",
    "camelyonpatch_level_2_split_train_y.h5.gz": "35c2d7259d906cfc8143347bb8e05be7",
    "camelyonpatch_level_2_split_train_meta.csv": "5a3dd671e465cfd74b5b822125e65b0a",
    "camelyonpatch_level_2_split_valid_x.h5.gz": "d5b63470df7cfa627aeec8b9dc0c066e",
    "camelyonpatch_level_2_split_valid_y.h5.gz": "2b85f58b927af9964a4c15b8f7e8f179",
    "camelyonpatch_level_2_split_valid_meta.csv": "67589e00a4a37ec317f2d1932c7502ca",
    "camelyonpatch_level_2_split_test_x.h5.gz": "d8c2d60d490dbd479f8199bdfa0cf6ec",
    "camelyonpatch_level_2_split_test_y.h5.gz": "60a7035772fbdb7f34eb86d4420cf66a",
    "camelyonpatch_level_2_split_test_meta.csv": "3455fd69135b66734e1008f3af684566",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching the publisher's checksum, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(raw_dir: Path, checksums: dict[str, str] = MD5) -> None:
    """Every raw file present with the published checksum, or a ValueError naming the ones that are not."""
    bad = [name for name, want in checksums.items() if not (raw_dir / name).exists() or md5sum(raw_dir / name) != want]
    if bad:
        raise ValueError(f"missing or corrupt PCam files in {raw_dir}: {', '.join(sorted(bad))}")


def raw_name(split: str, part: str) -> str:
    return f"camelyonpatch_level_2_split_{split}_{part}.h5.gz"


@dataclass
class SplitInfo:
    split: str
    patches: int
    positive_share: float
    slides: int


def _gunzip(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    tmp = dst.with_suffix(".partial")
    with gzip.open(src, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    tmp.rename(dst)


def prepare(raw_dir: Path, out_dir: Path, checksums: dict[str, str] = MD5) -> list[SplitInfo]:
    """raw .h5.gz files -> out_dir/<split>_x.npy (uint8 NHWC) and <split>_y.npy (uint8), plus meta CSVs.
    The raw files are checked against the published checksums first."""
    import h5py

    verify(raw_dir, checksums)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    for split in SPLITS:
        x_npy, y_npy = out_dir / f"{split}_x.npy", out_dir / f"{split}_y.npy"
        if not (x_npy.exists() and y_npy.exists()):
            for part in ("x", "y"):
                _gunzip(raw_dir / raw_name(split, part), out_dir / raw_name(split, part)[:-3])
            with h5py.File(out_dir / raw_name(split, "y")[:-3], "r") as f:
                y = np.asarray(f["y"]).reshape(-1).astype(np.uint8)
            with h5py.File(out_dir / raw_name(split, "x")[:-3], "r") as f:
                ds = f["x"]
                x = np.lib.format.open_memmap(x_npy, mode="w+", dtype=np.uint8, shape=ds.shape)
                for start in range(0, ds.shape[0], 8192):
                    x[start : start + 8192] = ds[start : start + 8192]
                x.flush()
            np.save(y_npy, y)
            for part in ("x", "y"):
                (out_dir / raw_name(split, part)[:-3]).unlink(missing_ok=True)
        shutil.copy(raw_dir / f"camelyonpatch_level_2_split_{split}_meta.csv", out_dir / f"{split}_meta.csv")
        x = np.load(x_npy, mmap_mode="r")
        y = np.load(y_npy)
        if x.shape != (EXPECTED[split], 96, 96, 3) or y.shape != (EXPECTED[split],):
            raise ValueError(f"{split}: got {x.shape} / {y.shape}, expected {EXPECTED[split]} patches of 96x96x3")
        slides = {row["wsi"] for row in read_meta(out_dir, split)}
        infos.append(SplitInfo(split, int(y.shape[0]), float(y.mean()), len(slides)))
    (out_dir / "splits.json").write_text(json.dumps([asdict(i) for i in infos], indent=2) + "\n", encoding="utf-8")
    return infos


def read_meta(data_dir: Path, split: str) -> list[dict[str, str]]:
    with (data_dir / f"{split}_meta.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(data_dir / f"{split}_x.npy", mmap_mode="r"), np.load(data_dir / f"{split}_y.npy")
