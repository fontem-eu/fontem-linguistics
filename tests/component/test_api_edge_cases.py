"""Extra route coverage: readyz, batch circuit-open, embed spend-cap, IDEMP overflow."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import Services
from src.api.routes import router as api_router, _IDEMPOTENCY_STORE
from src.backends.mistral import MistralBackend
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.spend_cap import SpendCap
from src.services.embedding import EmbeddingService
from src.services.translation import TranslationService
from tests.fixtures.mistral_stub import make_stub


@dataclass
class InMemoryCache:
    pool: object = None
    translations: dict = field(default_factory=dict)
    embeddings: dict = field(default_factory=dict)

    async def get_translations(self, text, source_lang, targets, backend):
        return {t: self.translations[(text, source_lang, t, backend)]
                for t in targets
                if (text, source_lang, t, backend) in self.translations}

    async def put_translations(self, text, source_lang, backend, translations):
        for t, v in translations.items():
            self.translations[(text, source_lang, t, backend)] = v

    async def get_embedding(self, text, backend):
        if (text, backend) not in self.embeddings:
            return None
        return list(self.embeddings[(text, backend)])

    async def put_embedding(self, text, backend, vector):
        self.embeddings[(text, backend)] = list(vector)


class _FakeCon:
    async def fetchval(self, _):
        return 1


class _FakePool:
    def __init__(self, ok: bool = True):
        self.ok = ok

    def acquire(self):
        outer = self

        class _Ctx:
            async def __aenter__(self):
                if not outer.ok:
                    raise RuntimeError("db down")
                return _FakeCon()

            async def __aexit__(self, *a):
                return False

        return _Ctx()


def _build_app(breaker=None, cap=None, pool_ok: bool = True):
    _IDEMPOTENCY_STORE.clear()
    stub_app, stub_state = make_stub()
    transport = httpx.ASGITransport(app=stub_app)
    client = httpx.AsyncClient(
        transport=transport, base_url="http://stub/v1",
        headers={"Authorization": "Bearer test-key-never-used"},
    )
    mistral = MistralBackend(
        api_url="http://stub/v1", api_key="test-key-never-used",
        chat_model="m", embed_model="me", timeout_s=5.0, max_retries=2,
        price_input_per_mtok=0.2, price_output_per_mtok=0.6, price_embed_per_mtok=0.1,
        client=client,
    )
    cache = InMemoryCache(pool=_FakePool(ok=pool_ok))
    b = breaker or CircuitBreaker(failure_threshold=1.0, min_requests=1000)
    c = cap or SpendCap(daily_cap_usd=1.0)
    translation = TranslationService(
        cache=cache, mistral=mistral, nllb=None,
        mistral_breaker=b, mistral_spend_cap=c,
    )
    embedding = EmbeddingService(
        cache=cache, mistral=mistral, labse=None, minilm=None,
        mistral_breaker=b, mistral_spend_cap=c,
    )
    app = FastAPI()
    app.include_router(api_router)
    app.state.services = Services(translation=translation, embedding=embedding)
    app.state.cache = cache
    return app, stub_state


def test_readyz_ok():
    app, _ = _build_app()
    r = TestClient(app).get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_readyz_fails_when_db_down():
    app, _ = _build_app(pool_ok=False)
    r = TestClient(app).get("/readyz")
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_batch_circuit_open_returns_empty_item():
    breaker = CircuitBreaker(failure_threshold=0.0, min_requests=1)
    await breaker.record_failure()   # trip it
    app, _ = _build_app(breaker=breaker)
    r = TestClient(app).post("/translate/batch", json={
        "items": [{"text": "a", "source_lang": "en"}],
        "targets": ["fr"], "backend": "mistral",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["translations"] == {}


def test_embed_spend_cap_429():
    app, _ = _build_app(cap=SpendCap(daily_cap_usd=0.0))
    r = TestClient(app).post("/embed", json={"text": "hi", "backend": "mistral-embed"})
    assert r.status_code == 429
    assert r.headers.get("X-Backend-State") == "spend-cap-exceeded"


def test_embed_circuit_open_503():
    # asyncio.run, not get_event_loop().run_until_complete — Python 3.14
    # removed implicit loop creation in threads without a running loop.
    breaker = CircuitBreaker(failure_threshold=0.0, min_requests=1)
    asyncio.run(breaker.record_failure())
    app, _ = _build_app(breaker=breaker)
    r = TestClient(app).post("/embed", json={"text": "hi", "backend": "mistral-embed"})
    assert r.status_code == 503
    assert r.headers.get("X-Backend-State") == "circuit-open"


def test_embed_502_on_malformed_mistral():
    app, stub = _build_app()
    stub.embed_responses.append((200, {"data": [{"embedding": []}]}))
    r = TestClient(app).post("/embed", json={"text": "hi", "backend": "mistral-embed"})
    assert r.status_code == 502
    assert "mistral error" in r.json()["detail"]


def test_embed_502_on_transient_exhausted():
    app, stub = _build_app()
    stub.embed_responses.extend([(503, {"error": "down"})] * 5)
    r = TestClient(app).post("/embed", json={"text": "hi", "backend": "mistral-embed"})
    assert r.status_code == 502
    assert "transient" in r.json()["detail"]


def test_idempotency_store_evicts_oldest_over_cap(monkeypatch):
    _IDEMPOTENCY_STORE.clear()
    app, _ = _build_app()
    # Shrink the cap for the test by monkeypatching the module constant.
    from src.api import routes
    monkeypatch.setattr(routes, "_IDEMPOTENCY_MAX", 3)

    client = TestClient(app)
    payload = {
        "items": [{"text": "x", "source_lang": "en"}],
        "targets": ["fr"], "backend": "mistral",
    }
    for k in ["a", "b", "c", "d"]:
        client.post("/translate/batch", json=payload,
                    headers={"Idempotency-Key": f"key-{k}"})
    # oldest evicted
    assert "key-a" not in _IDEMPOTENCY_STORE
    assert "key-d" in _IDEMPOTENCY_STORE
