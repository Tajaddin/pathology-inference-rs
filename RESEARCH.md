# Research behind this project

Written before the code (2026-10-18). Sources are linked.

## What employers in the target list ask for

From 25 saved postings: **computer vision** (Moffitt Cancer Center's AI/ML engineer role in precision oncology
asks for multimodal work across "clinical, imaging, pathology, molecular" data and for "APIs, software services,
and user-facing applications that integrate AI"), **Rust** (SpruceID; 1Password asks for "Rust and/or Golang"),
**JavaScript** (Affirm, Clerkie, Pinterest, Tampa General), **TensorFlow** (Artera, Mosaic, Quora, ReliaQuest).
A pathology model behind a Rust service with a JavaScript viewer is one system a precision-oncology team would
recognise, and each layer proves one of those skills.

## Trends that shaped it (2026)

- Edge and CPU inference with ONNX Runtime: the `ort` crate is the established Rust wrapper; its docs call
  2.0.0-rc.14 production-ready, and it is used by Hugging Face Text Embeddings Inference, Google's Magika and
  SurrealDB ([ort](https://ort.pyke.io/), [overview](https://ai-tldr.dev/tools/ort-rust/),
  [2026 article](https://dasroot.net/posts/2026/03/onnx-runtime-rust-ml-inference-optimization/)).
  `tract` is the pure-Rust alternative, better suited to WASM/embedded than server throughput.
- Rust in ML and data infrastructure (DataFusion, Polars, Iceberg-rust) keeps growing
  ([Apache Data Lakehouse Weekly](https://amdatalakehouse.substack.com/p/apache-data-lakehouse-weekly-may-746)).
- Portfolios that show the whole lifecycle (data, held-out evaluation, deployment, monitoring) and state results
  near the top are what reviewers look for
  ([letsdatascience](https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026)).

## Free tools used

| Skill | Production use | Free or local mode used |
|---|---|---|
| TensorFlow | Keras 3 training, export to ONNX for serving | TF 2.21 with CUDA wheels on an RTX 4060 (WSL2); tf2onnx 1.17 |
| Computer vision | Histopathology patch classification, tumour heatmaps | PatchCamelyon, CC0 |
| Rust | axum services, ONNX Runtime inference | Rust 1.95, crates from crates.io |
| JavaScript | Browser viewer, client SDK, k6 load scripts | Node 24, Vite, Playwright, k6 (all open source) |
| GCP | Cloud Run for the service, Cloud Storage for the model | Terraform validated; Cloud Run free tier (2M requests/month) once an account exists |

## Dataset

**PatchCamelyon (PCam)**: 327,680 RGB patches of 96x96 px from the Camelyon16 lymph-node whole-slide images,
binary label = tumour in the centre 32x32 region; published splits of 262,144 / 32,768 / 32,768 patches; the
test split has 32,768 patches from 129 test slides (checked from the metadata CSV). Licence CC0 per the
official repository ([basveeling/pcam](https://github.com/basveeling/pcam), mirrored on
[Hugging Face](https://huggingface.co/datasets/1aurent/PatchCamelyon)); files from
[Zenodo record 2546921](https://zenodo.org/records/2546921). Reference result from the PCam paper: a
rotation-equivariant G-DenseNet at 0.963 AUC
([Veeling et al., 2018](https://arxiv.org/abs/1806.03962)). The original PCam split contains duplicate patches
from probabilistic sampling; Kaggle's version removes them
([note](https://www.tbench.ai/registry/mlebench-lite/head/histopathologic-cancer-detection)). This project uses the
original splits unchanged so results are comparable with the paper.

Why not other datasets: MedMNIST PathMNIST (28 px; too small to be realistic), Camelyon16/17 whole slides
(hundreds of GB), TCGA (controlled-access for some data), industrial defect sets such as MVTec AD (non-commercial
licence).
