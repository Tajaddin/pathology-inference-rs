"""Train on the PCam training split, select on validation, evaluate once on test, export to ONNX.

    python -m pcam.train --data ~/data/pcam --out ../models/pcam/1 --reports ../reports

Split use: training patches fit the network; validation patches choose the epoch (early stopping on AUC), fit the
temperature and fix the operating thresholds; the test split is scored once, after all of that is frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pcam import data, metrics
from pcam.model import PATCH, augment, build, dihedral


def batches(
    x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch):
        idx = np.sort(order[start : start + batch])  # sorted reads are kinder to the memory map
        yield np.asarray(x[idx]), y[idx].astype(np.float32)


def dataset(x: np.ndarray, y: np.ndarray, batch: int, train: bool, seed: int = 0) -> Any:
    import tensorflow as tf

    epoch = {"n": 0}

    def gen() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        epoch["n"] += 1
        yield from batches(x, y, batch, shuffle=train, seed=seed + epoch["n"])

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec((None, PATCH, PATCH, 3), tf.uint8), tf.TensorSpec((None,), tf.float32)),
    )
    steps = (len(y) + batch - 1) // batch
    ds = ds.apply(tf.data.experimental.assert_cardinality(steps))
    if train:
        ds = ds.map(lambda a, b: (augment(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(4)


def predict_logits(model: Any, x: np.ndarray, batch: int = 512, tta: bool = False) -> np.ndarray:
    import tensorflow as tf

    out = []
    for start in range(0, len(x), batch):
        b = tf.cast(np.asarray(x[start : start + batch]), tf.float32)
        if tta:
            logits = [model(dihedral(b, k), training=False)[:, 0] for k in range(8)]
            out.append(tf.reduce_mean(tf.stack(logits), axis=0).numpy())
        else:
            out.append(model(b, training=False)[:, 0].numpy())
    return np.concatenate(out).astype(np.float64)


def evaluate_split(y: np.ndarray, p: np.ndarray, thresholds: dict[str, float], ci: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "patches": len(y),
        "auc": metrics.auc(y, p),
        "ece": metrics.ece(y, p),
        "nll": metrics.nll(y, p),
        "operating_points": {k: metrics.operating_point(y, p, t) for k, t in thresholds.items()},
    }
    if ci:
        out["auc_ci95"] = list(metrics.bootstrap_ci(y, p))
    return out


def slide_level(meta: list[dict[str, str]], p: np.ndarray) -> dict[str, Any]:
    """Slide = has any tumour patch; score = its highest patch probability. 129 test slides."""
    by_slide: dict[str, list[int]] = {}
    for i, row in enumerate(meta):
        by_slide.setdefault(row["wsi"], []).append(i)
    names = sorted(by_slide)
    y = np.array([any(meta[i]["tumor_patch"] == "True" for i in by_slide[s]) for s in names], dtype=np.int64)
    s = np.array([p[by_slide[name]].max() for name in names])
    out: dict[str, Any] = {"slides": len(names), "tumour_slides": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auc_max_patch"] = metrics.auc(y, s)
        out["auc_ci95"] = list(metrics.bootstrap_ci(y, s))
    return out


def float32_copy(model: Any) -> Any:
    """The same weights in a float32 graph: the service runs on CPU, where float16 casts only cost time."""
    import keras

    policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("float32")
    try:
        fp32 = build(None)
        fp32.set_weights(model.get_weights())
    finally:
        keras.mixed_precision.set_global_policy(policy)
    return fp32


def export_onnx(model: Any, out_dir: Path, sample: np.ndarray) -> dict[str, Any]:
    """ONNX (opset 17) of the float32 copy, through a SavedModel; checked against TensorFlow on real patches.

    The SavedModel step matters: converting the Keras model or a traced function directly (tf2onnx.from_keras,
    from_function, Keras' own ONNX export) turns the constants of EfficientNetV2's Normalization layer into extra
    graph inputs, and the ONNX model then cannot run on images alone.
    """
    import subprocess
    import sys
    import tempfile

    import onnxruntime as ort

    fp32 = float32_copy(model)
    path = out_dir / "model.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        fp32.export(str(Path(tmp) / "saved"), format="tf_saved_model", verbose=False)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(Path(tmp) / "saved"),
                "--opset",
                "17",
                "--output",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    x = sample.astype(np.float32)
    input_name = session.get_inputs()[0].name
    ort_logits = np.concatenate(
        [session.run([output_name], {input_name: x[i : i + 256]})[0].reshape(-1) for i in range(0, len(x), 256)]
    )
    tf_logits = predict_logits(model, x)  # batched: one 2,048-patch call ran the 8 GB GPU out of memory
    return {
        "onnx_path": str(path.name),
        "input_name": session.get_inputs()[0].name,
        "output_name": output_name,
        "opset": 17,
        "parity_patches": len(x),
        "max_abs_logit_diff": float(np.max(np.abs(ort_logits - tf_logits))),
        "max_abs_probability_diff": float(np.max(np.abs(metrics.sigmoid(ort_logits) - metrics.sigmoid(tf_logits)))),
        "onnx_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N training patches (smoke runs)")
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="train and score on validation only (model selection); never reads the test split, exports nothing",
    )
    args = parser.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(42)
    keras.mixed_precision.set_global_policy("mixed_float16" if tf.config.list_physical_devices("GPU") else "float32")
    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    x_tr, y_tr = data.load(args.data, "train")
    x_va, y_va = data.load(args.data, "valid")
    x_te, y_te = data.load(args.data, "test")
    if args.limit:
        x_tr, y_tr = x_tr[: args.limit], y_tr[: args.limit]
        x_va, y_va, x_te, y_te = (
            x_va[: args.limit // 4],
            y_va[: args.limit // 4],
            x_te[: args.limit // 4],
            y_te[: args.limit // 4],
        )

    model = build(None if args.weights == "none" else args.weights)
    steps = (len(y_tr) + args.batch - 1) // args.batch
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps * args.epochs, alpha=0.02)
    model.compile(
        optimizer=keras.optimizers.AdamW(schedule, weight_decay=1e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.AUC(name="auc", from_logits=True), keras.metrics.BinaryAccuracy(threshold=0.0)],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    best = args.out / "best.weights.h5"
    start = time.perf_counter()
    history = model.fit(
        dataset(x_tr, y_tr, args.batch, train=True),
        validation_data=dataset(x_va, y_va, 512, train=False),
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(best), monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True
            ),
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        ],
        verbose=2,
    )
    train_seconds = time.perf_counter() - start
    model.load_weights(best)
    best.unlink()

    # Everything below is chosen on validation only.
    z_va = predict_logits(model, x_va)
    if args.select_only:
        val_auc = [float(v) for v in history.history["val_auc"]]
        selection = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "lr": args.lr,
            "batch": args.batch,
            "weights": args.weights,
            "train_patches": len(y_tr),
            "epochs_run": len(val_auc),
            "best_epoch": int(np.argmax(val_auc)) + 1,
            "valid_auc_best_weights": metrics.auc(y_va, z_va),
            "train_seconds": round(train_seconds, 1),
            "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        }
        args.reports.mkdir(parents=True, exist_ok=True)
        (args.reports / f"selection-lr{args.lr:g}.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: selection[k] for k in ("lr", "best_epoch", "valid_auc_best_weights", "train_seconds")}))
        return
    temperature = metrics.fit_temperature(z_va, y_va)
    p_va = metrics.sigmoid(z_va / temperature)
    thresholds = {
        "balanced_0.5": 0.5,
        "specificity_95_on_valid": metrics.threshold_for_specificity(y_va, p_va, 0.95),
    }

    # The one look at the test split.
    z_te = predict_logits(model, x_te)
    p_te = metrics.sigmoid(z_te / temperature)
    p_te_uncal = metrics.sigmoid(z_te)
    z_te_tta = predict_logits(model, x_te, tta=True)
    z_va_tta = predict_logits(model, x_va, tta=True)
    t_tta = metrics.fit_temperature(z_va_tta, y_va)
    p_te_tta = metrics.sigmoid(z_te_tta / t_tta)
    test_meta = data.read_meta(args.data, "test")[: len(y_te)]

    parity = export_onnx(model, args.out, np.asarray(x_te[:2048]))
    model_json = {
        "version": "1",
        "input_name": parity["input_name"],
        "output_name": parity["output_name"],
        "temperature": temperature,
        "threshold": 0.5,
    }
    (args.out / "model.json").write_text(json.dumps(model_json, indent=2) + "\n", encoding="utf-8")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "EfficientNetV2-B0 (ImageNet weights) fine-tuned, one logit, mixed precision",
        "gpus": gpus,
        "train_patches": len(y_tr),
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(train_seconds, 1),
        "batch": args.batch,
        "lr": args.lr,
        "temperature": temperature,
        "thresholds": thresholds,
        "valid": evaluate_split(y_va, p_va, thresholds, ci=False),
        "test": evaluate_split(y_te, p_te, thresholds, ci=True),
        "test_uncalibrated": {"ece": metrics.ece(y_te, p_te_uncal), "nll": metrics.nll(y_te, p_te_uncal)},
        "test_tta8": {**evaluate_split(y_te, p_te_tta, thresholds, ci=True), "temperature": t_tta},
        "test_slide_level": slide_level(test_meta, p_te),
        "onnx": parity,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(args.reports / "test_probabilities.npy", p_te.astype(np.float32))
    print(json.dumps({k: report[k] for k in ("epochs_run", "train_seconds", "temperature")}))
    print(
        "test AUC",
        round(report["test"]["auc"], 4),
        "TTA",
        round(report["test_tta8"]["auc"], 4),
        "slides",
        report["test_slide_level"].get("auc_max_patch"),
        "onnx diff",
        parity["max_abs_probability_diff"],
    )


if __name__ == "__main__":
    main()
