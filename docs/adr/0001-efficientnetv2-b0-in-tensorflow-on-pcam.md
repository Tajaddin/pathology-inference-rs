# 0001. EfficientNetV2-B0 in TensorFlow, fine-tuned on PatchCamelyon

Date: 2026-10-18. Status: accepted.

## Context

The task is patch-level metastasis detection on lymph-node sections (PatchCamelyon: 96x96 patches, published
train/validation/test splits, CC0). The service must run on CPU at interactive latency, so the model has to be
small; it must also be trainable on one 8 GB consumer GPU.

## Decision

- Keras 3 on TensorFlow 2.21, `EfficientNetV2B0` with ImageNet weights, `include_preprocessing=True` so the
  network takes raw 0..255 RGB and the ONNX export carries the normalisation.
- One logit out; temperature scaling fitted on the validation split, applied by the service.
- Augmentation for histology: the 8 dihedral symmetries (tissue has no orientation) and small hue, saturation,
  brightness and contrast jitter (stain varies between labs and scanners).
- Mixed precision on the GPU; AdamW with cosine decay; early stopping on validation AUC.

## Alternatives

- A pathology foundation model (UNI, Virchow, CONCH): stronger features, but their licences restrict use and the
  weights are hundreds of MB to GBs; not a CPU service.
- Rotation-equivariant CNNs (the PCam paper's G-DenseNet): better published AUC (0.963), no maintained Keras
  implementation.

## Consequences

- The model is about 6M parameters; ONNX size and CPU latency are measured in the reports.
- PCam's test split comes from different slides than training; the gap between validation and test numbers is
  reported, not tuned away.
