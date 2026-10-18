"""The patch classifier: EfficientNetV2-B0 (ImageNet weights) fine-tuned on PCam, one logit out.

The network takes raw RGB in 0..255: Keras' EfficientNetV2 includes its own rescaling and normalisation, so the
ONNX export carries them too and the Rust service only lays out pixels.
"""

from __future__ import annotations

from typing import Any

PATCH = 96


def build(weights: str | None = "imagenet", dropout: float = 0.25) -> Any:
    import keras

    inputs = keras.Input(shape=(PATCH, PATCH, 3), name="image", dtype="float32")
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(PATCH, PATCH, 3),
        pooling="avg",
        include_preprocessing=True,
    )
    x = backbone(inputs)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    logit = keras.layers.Dense(1, name="logit", dtype="float32")(x)
    return keras.Model(inputs, logit, name="pcam_effnetv2b0")


def augment(images: Any, seed: Any = None) -> Any:
    """Geometric (the 8 dihedral symmetries) and stain/colour jitter, per image, on a uint8 batch."""
    import tensorflow as tf

    x = tf.cast(images, tf.float32) / 255.0
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    transpose = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.5
    x = tf.where(transpose, tf.transpose(x, [0, 2, 1, 3]), x)
    # H&E stain differs between labs and scanners: small hue/saturation/brightness/contrast shifts.
    x = tf.image.random_brightness(x, 0.10)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    x = tf.image.random_saturation(x, 0.85, 1.15)
    x = tf.image.random_hue(x, 0.04)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def dihedral(images: Any, k: int) -> Any:
    """The k-th of the 8 rotations/reflections (k in 0..7), for test-time augmentation."""
    import tensorflow as tf

    x = tf.image.rot90(images, k % 4)
    return tf.image.flip_left_right(x) if k >= 4 else x
