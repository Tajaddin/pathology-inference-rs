"""The Python baseline's /v1/classify contract, against a tiny ONNX model built here (no trained weights needed)."""

from __future__ import annotations

import base64
import importlib
import io
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

KEY = "local-dev-key-tests-000001"


def tiny_model(path: Path) -> None:
    """logit = (127.5 - mean pixel) / 10: darker patches score higher, like the Rust service's fake."""
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Sub", ["mid", "mean"], ["centred"]),
        helper.make_node("Div", ["centred", "scale"], ["scaled"]),
        helper.make_node("Reshape", ["scaled", "shape"], ["output_0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 96, 96, 3])],
        [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["n", 1])],
        [
            helper.make_tensor("mid", TensorProto.FLOAT, [], [127.5]),
            helper.make_tensor("scale", TensorProto.FLOAT, [], [10.0]),
            helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def png(level: int, size: int = 96) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), level, np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    tiny_model(tmp_path / "model.onnx")
    (tmp_path / "model.json").write_text(
        json.dumps(
            {"version": "test", "input_name": "image", "output_name": "output_0", "temperature": 1.0, "threshold": 0.5}
        )
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("API_KEYS", f"tests:{KEY}")
    from baseline import app as module

    return TestClient(importlib.reload(module).app)


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {KEY}"}


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_scores_patches_in_order_with_their_ids(client: TestClient) -> None:
    body = {"patches": [{"id": "dark", "image_base64": png(20)}, {"id": "light", "image_base64": png(230)}]}
    out = client.post("/v1/classify", json=body, headers=auth()).json()
    assert [r["id"] for r in out["results"]] == ["dark", "light"]
    dark, light = (r["probability"] for r in out["results"])
    assert dark == pytest.approx(1 / (1 + np.exp(-(127.5 - 20) / 10)), rel=1e-5)
    assert dark > 0.5 > light
    assert [r["tumor"] for r in out["results"]] == [True, False]


def test_refuses_requests_without_a_valid_key(client: TestClient) -> None:
    body = {"patches": [{"image_base64": png(100)}]}
    assert client.post("/v1/classify", json=body).status_code == 401
    assert client.post("/v1/classify", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.parametrize(
    "patches",
    [
        [{"image_base64": png(100, size=64)}],  # wrong size
        [{"image_base64": "not base64!"}],
        [{"image_base64": png(100)}] * 65,  # over the 64-patch limit
        [],
    ],
)
def test_rejects_bad_input(client: TestClient, patches: list[dict[str, str]]) -> None:
    assert client.post("/v1/classify", json={"patches": patches}, headers=auth()).status_code == 422
