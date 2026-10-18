"""PCam preparation on a tiny synthetic copy with the real file layout: checksums, shapes, labels, metadata."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pcam import data

SIZES = {"train": 12, "valid": 4, "test": 4}


def write_split(raw: Path, split: str, n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y = (np.arange(n) % 2).astype(np.uint8).reshape(n, 1, 1, 1)
    for part, arr in (("x", x), ("y", y)):
        plain = raw / data.raw_name(split, part)[:-3]
        with h5py.File(plain, "w") as f:
            f.create_dataset(part, data=arr)
        with plain.open("rb") as src, gzip.open(raw / data.raw_name(split, part), "wb") as dst:
            dst.write(src.read())
        plain.unlink()
    with (raw / f"camelyonpatch_level_2_split_{split}_meta.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["", "coord_y", "coord_x", "tumor_patch", "center_tumor_patch", "wsi"])
        for i in range(n):
            w.writerow([i, 0, i * 96, str(bool(i % 2)), str(bool(i % 2)), f"{split}_slide_{i % 3}"])
    return x


@pytest.fixture
def raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, np.ndarray]]:
    monkeypatch.setattr(data, "EXPECTED", SIZES)
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    xs = {split: write_split(raw, split, n, rng) for split, n in SIZES.items()}
    return raw, xs


def checksums(raw: Path) -> dict[str, str]:
    return {p.name: data.md5sum(p) for p in raw.iterdir()}


def test_prepare_writes_arrays_labels_and_split_summary(
    raw: tuple[Path, dict[str, np.ndarray]], tmp_path: Path
) -> None:
    raw_dir, xs = raw
    out = tmp_path / "prepared"
    infos = data.prepare(raw_dir, out, checksums(raw_dir))
    assert [(i.split, i.patches, i.positive_share, i.slides) for i in infos] == [
        ("train", 12, 0.5, 3),
        ("valid", 4, 0.5, 3),
        ("test", 4, 0.5, 3),
    ]
    x, y = data.load(out, "test")
    assert x.shape == (4, 96, 96, 3)
    assert x.dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(x), xs["test"])
    np.testing.assert_array_equal(y, [0, 1, 0, 1])
    assert json.loads((out / "splits.json").read_text())[0]["split"] == "train"
    assert not list(out.glob("*.h5")), "decompressed HDF5 files are removed after conversion"


def test_prepare_refuses_a_corrupt_download(raw: tuple[Path, dict[str, np.ndarray]], tmp_path: Path) -> None:
    raw_dir, _ = raw
    sums = checksums(raw_dir)
    with (raw_dir / data.raw_name("valid", "y")).open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ValueError, match="valid_y"):
        data.prepare(raw_dir, tmp_path / "prepared", sums)


def test_verify_names_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test_meta"):
        data.verify(tmp_path, {"camelyonpatch_level_2_split_test_meta.csv": "0" * 32})


def test_published_checksums_cover_all_nine_files() -> None:
    names = {data.raw_name(s, p) for s in data.SPLITS for p in ("x", "y")}
    names |= {f"camelyonpatch_level_2_split_{s}_meta.csv" for s in data.SPLITS}
    assert set(data.MD5) == names
    assert all(len(v) == 32 for v in data.MD5.values())
