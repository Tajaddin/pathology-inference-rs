# Sprint 1 retro: data and model

Dates: 2026-10-18. Milestone: [Sprint 1 - Data and model](https://github.com/Tajaddin/pathology-inference-rs/milestone/1).

**Goal:** PCam prepared reproducibly, a TensorFlow classifier trained on the GPU, evaluated once on the held-out
test split, exported to ONNX with a parity check. **Met.**

## Delivered

| Issue | Points | Result |
|---|---|---|
| #1 scaffold | 2 | Cargo workspace, Python training project with a lock file, JS packages, CI |
| #2 data | 3 | resumable download with MD5 checks against Zenodo (bats-tested); preparation refuses a corrupt file |
| #3 training | 5 | EfficientNetV2-B0 with ImageNet weights, mixed precision, stain and dihedral augmentation, early stopping |
| #4 evaluation | 3 | test split scored once: AUC with bootstrap CI, operating points, calibration, 8-way TTA, slide level |
| #5 ONNX export | 2 | float32 export through a SavedModel; parity checked on real test patches |
| #19 bug: export | 2 | the first three export routes made EfficientNet's normalization constants extra graph inputs ([ADR 0005](../adr/0005-onnx-export-through-a-float32-savedmodel.md)) |
| #20 selection | 2 | learning rate chosen on validation only; the candidates' results are committed |

19 points done (15 planned, 4 for work found during the sprint).

## Measured (`reports/`, `docs/MODEL-CARD.md`)

- All nine PCam files (8.1 GB) match Zenodo's MD5 checksums; preparation into memory-mapped arrays took 219 s.
- Selection on validation: learning rate 1e-3 reached AUC 0.9625, 3e-4 0.9584, 1e-4 0.9578.
- Final run (RTX 4060, 8.7 min, 5 epochs): **test AUC 0.9409 (95% CI 0.9386-0.9435)**, 0.9475 with 8-way TTA;
  the exported ONNX model, which the service runs, scores **0.9410** on the same 32,768 patches; slide-level AUC
  0.975 on 129 test slides. Validation AUC was 0.9629.

## What went well

- A one-epoch smoke run on 2,048 patches before the real training exercised every step to the end and found the
  export bug in two minutes instead of after a full run.

## What did not

- The first full run trained for 11 minutes and then ran the 8 GB GPU out of memory in the last step: the parity
  check pushed 2,048 patches through TensorFlow in one call (the smoke run had only used 512). Nothing was saved,
  because the report is written after the export. The check is batched now.
- The first full run also overfitted after one epoch (validation AUC 0.963 to 0.914 by epoch 4), which is why the
  learning rate was then chosen on validation (#20).
- **Calibration did not transfer to the test split** (ECE 0.110 after temperature scaling, 0.125 before) and at
  0.5 the model under-calls tumour on test (sensitivity 69.6%, specificity 96.8%). The model card says so, and the
  viewer no longer calls its scores calibrated.
- During the final run the WSL VM ran out of memory (swap full, WSL stopped answering for a few minutes): two
  Docker stacks from other projects were still running in the same VM. They were stopped; the run finished.

## Carried into Sprint 2

- The served model is the float32 export; its own test numbers are in `reports/onnx-test.json`.
