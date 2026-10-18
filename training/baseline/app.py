"""FastAPI + onnxruntime baseline with the Rust service's /v1/classify contract.

Written the way a Python team would ship it: one ONNX Runtime session per worker process, the request's patches
scored as one batch, no cross-request batching. `uvicorn baseline.app:app --workers N`.
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/pcam/1"))
META = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
KEYS = [item.split(":", 1)[1] for item in os.environ.get("API_KEYS", "").split(",") if ":" in item]
options = ort.SessionOptions()
options.intra_op_num_threads = int(os.environ.get("INTRA_THREADS", "4"))
SESSION = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), options, providers=["CPUExecutionProvider"])

app = FastAPI(title="PCam baseline (Python)")


class PatchIn(BaseModel):
    id: str | None = None
    image_base64: str


class ClassifyRequest(BaseModel):
    patches: list[PatchIn] = Field(min_length=1, max_length=64)


def decode(b64: str) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="unreadable image") from exc
    if img.size != (96, 96):
        raise HTTPException(status_code=422, detail="patches must be 96x96")
    return np.asarray(img, dtype=np.float32)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/classify")
def classify(req: ClassifyRequest, authorization: str = Header(default="")) -> dict[str, object]:
    key = authorization.removeprefix("Bearer ")
    if not any(hmac.compare_digest(key.encode(), k.encode()) for k in KEYS):
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    batch = np.stack([decode(p.image_base64) for p in req.patches])
    logits = SESSION.run([META["output_name"]], {META["input_name"]: batch})[0].reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits / META["temperature"]))
    t = META["threshold"]
    return {
        "model_version": META["version"],
        "threshold": t,
        "results": [
            {"id": p.id or str(i), "probability": float(v), "tumor": bool(v >= t)}
            for i, (p, v) in enumerate(zip(req.patches, probs, strict=True))
        ],
    }
