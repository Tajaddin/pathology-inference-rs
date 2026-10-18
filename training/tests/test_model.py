"""The network, its augmentation, the training helpers and the ONNX export (TensorFlow on CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tf

tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    from pcam.model import build

    return build(None)


def test_one_logit_per_patch(model) -> None:  # type: ignore[no-untyped-def]
    out = model(np.zeros((3, 96, 96, 3), np.float32), training=False)
    assert tuple(out.shape) == (3, 1)


def test_augment_keeps_shape_and_pixel_range() -> None:
    from pcam.model import augment

    images = tf.constant(np.random.default_rng(0).integers(0, 256, (8, 96, 96, 3)), tf.uint8)
    out = augment(images)
    assert out.shape == (8, 96, 96, 3)
    assert out.dtype == tf.float32
    assert float(tf.reduce_min(out)) >= 0.0
    assert float(tf.reduce_max(out)) <= 255.0


def test_dihedral_gives_the_eight_symmetries() -> None:
    from pcam.model import dihedral

    image = tf.constant(np.arange(96 * 96 * 3, dtype=np.float32).reshape(1, 96, 96, 3))
    views = [dihedral(image, k).numpy().tobytes() for k in range(8)]
    assert len(set(views)) == 8
    np.testing.assert_array_equal(dihedral(image, 0).numpy(), image.numpy())


def test_batches_cover_every_patch_once_in_sorted_reads() -> None:
    from pcam.train import batches

    x = np.arange(103 * 2).reshape(103, 2)
    y = np.arange(103)
    seen = []
    for xb, yb in batches(x, y, 16, shuffle=True, seed=3):
        assert np.all(np.diff(yb) > 0), "indices are sorted inside a batch"
        np.testing.assert_array_equal(xb[:, 0] // 2, yb)
        seen.extend(yb.tolist())
    assert sorted(seen) == list(range(103))


def test_slide_level_uses_the_highest_patch_per_slide() -> None:
    from pcam.train import slide_level

    meta = [
        {"wsi": "a", "tumor_patch": "False"},
        {"wsi": "a", "tumor_patch": "True"},
        {"wsi": "b", "tumor_patch": "False"},
        {"wsi": "b", "tumor_patch": "False"},
        {"wsi": "c", "tumor_patch": "True"},
    ]
    out = slide_level(meta, np.array([0.1, 0.9, 0.3, 0.2, 0.8]))
    assert (out["slides"], out["tumour_slides"], out["auc_max_patch"]) == (3, 2, 1.0)


def test_onnx_export_matches_tensorflow(model, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from pcam.train import export_onnx

    sample = np.random.default_rng(1).integers(0, 256, (300, 96, 96, 3)).astype(np.uint8)
    report = export_onnx(model, tmp_path, sample)
    assert report["input_name"] == "image"
    assert report["parity_patches"] == 300
    assert report["max_abs_probability_diff"] < 1e-4
    assert (tmp_path / "model.onnx").stat().st_size > 1_000_000
