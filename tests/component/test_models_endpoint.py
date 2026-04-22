"""Component tests for the /models endpoint — quality scores + shape."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router


def _minimal_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_models_endpoint_lists_all_backends():
    client = TestClient(_minimal_app())
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    names = {m["backend"] for m in body["models"]}
    assert names == {"mistral", "nllb-local", "mistral-embed", "labse-local"}


def test_models_have_quality_scores_in_range():
    client = TestClient(_minimal_app())
    body = client.get("/models").json()
    for m in body["models"]:
        assert 0.0 <= m["quality_score"] <= 1.0
        assert m["kind"] in {"translation", "embedding"}
        assert m["cost_tier"] in {"paid", "free"}


def test_mistral_higher_quality_than_local():
    client = TestClient(_minimal_app())
    body = client.get("/models").json()
    by_backend = {m["backend"]: m for m in body["models"]}
    assert by_backend["mistral"]["quality_score"] > by_backend["nllb-local"]["quality_score"]
    assert by_backend["mistral-embed"]["quality_score"] > by_backend["labse-local"]["quality_score"]


def test_embedding_backends_expose_dim():
    client = TestClient(_minimal_app())
    body = client.get("/models").json()
    mistral_e = next(m for m in body["models"] if m["backend"] == "mistral-embed")
    labse = next(m for m in body["models"] if m["backend"] == "labse-local")
    assert mistral_e["dim"] == 1024
    assert labse["dim"] == 768


def test_translation_backends_have_null_dim():
    client = TestClient(_minimal_app())
    body = client.get("/models").json()
    for m in body["models"]:
        if m["kind"] == "translation":
            assert m["dim"] is None
