"""Component tests: real FastAPI app wired with in-memory cache + Mistral stub."""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.deps import Services
from src.api.routes import router as api_router, _IDEMPOTENCY_STORE
from src.backends.mistral import MistralBackend
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.spend_cap import SpendCap
from src.services.embedding import EmbeddingService
from src.services.translation import TranslationService
from tests.fixtures.mistral_stub import make_stub


# ── In-memory cache substitute (PG-free) ──────────────────────────

@dataclass
class InMemoryCache:
    pool: object = None     # not used but app.state.cache expects it for readyz
    translations: dict = field(default_factory=dict)
    embeddings: dict = field(default_factory=dict)

    async def get_translations(self, text, source_lang, targets, backend):
        return {
            t: self.translations[(text, source_lang, t, backend)]
            for t in targets
            if (text, source_lang, t, backend) in self.translations
        }

    async def put_translations(self, text, source_lang, backend, translations):
        for t, v in translations.items():
            self.translations[(text, source_lang, t, backend)] = v

    async def get_embedding(self, text, backend):
        if (text, backend) not in self.embeddings:
            return None
        return list(self.embeddings[(text, backend)])

    async def put_embedding(self, text, backend, vector):
        self.embeddings[(text, backend)] = list(vector)


def _build_mistral_against_stub(stub_app) -> MistralBackend:
    """Give the MistralBackend an httpx client that routes to the in-process stub."""
    transport = httpx.ASGITransport(app=stub_app)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="http://stub/v1",
        headers={"Authorization": "Bearer test-key-never-used"},
    )
    return MistralBackend(
        api_url="http://stub/v1",
        api_key="test-key-never-used",
        chat_model="mistral-small-latest",
        embed_model="mistral-embed",
        timeout_s=5.0,
        max_retries=2,
        price_input_per_mtok=0.20,
        price_output_per_mtok=0.60,
        price_embed_per_mtok=0.10,
        client=client,
    )


@pytest.fixture
def app_and_state():
    """A FastAPI instance with services + stub wired in, no lifespan."""
    from fastapi import FastAPI
    _IDEMPOTENCY_STORE.clear()
    stub_app, stub_state = make_stub()
    mistral = _build_mistral_against_stub(stub_app)
    cache = InMemoryCache()
    breaker = CircuitBreaker(failure_threshold=1.0, min_requests=1000)
    cap = SpendCap(daily_cap_usd=1.0)
    translation = TranslationService(
        cache=cache, mistral=mistral, nllb=None,
        mistral_breaker=breaker, mistral_spend_cap=cap,
    )
    embedding = EmbeddingService(
        cache=cache, mistral=mistral, labse=None,
        mistral_breaker=breaker, mistral_spend_cap=cap,
    )
    app = FastAPI()
    app.include_router(api_router)
    app.state.services = Services(translation=translation, embedding=embedding)
    app.state.cache = cache
    return app, stub_state, breaker, cap


def test_translate_endpoint_happy_path(app_and_state):
    app, _stub, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/translate", json={
        "text": "Ministero della Difesa",
        "source_lang": "it",
        "targets": ["en", "fr"],
        "backend": "mistral",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "mistral"
    assert body["cached"] is False
    assert body["translations"] == {"en": "[en]stub", "fr": "[fr]stub"}


def test_translate_cache_hit_second_call(app_and_state):
    app, stub, *_ = app_and_state
    client = TestClient(app)
    payload = {
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    }
    assert client.post("/translate", json=payload).status_code == 200
    r2 = client.post("/translate", json=payload)
    assert r2.json()["cached"] is True
    # Only one chat call made
    assert sum(1 for c in stub.calls if c[0] == "chat") == 1


def test_translate_partial_cache(app_and_state):
    app, _stub, *_ = app_and_state
    client = TestClient(app)
    client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    })
    # Second call adds de; fr should be served from cache
    r = client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr", "de"], "backend": "mistral",
    })
    body = r.json()
    assert set(body["translations"].keys()) == {"fr", "de"}
    assert body["partial_cached_targets"] == ["fr"]


def test_translate_400_on_missing_targets(app_and_state):
    app, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": [], "backend": "mistral",
    })
    assert r.status_code == 422


def test_translate_returns_503_when_breaker_open(app_and_state):
    app, _stub, breaker, _cap = app_and_state
    client = TestClient(app)
    # Force the breaker open. asyncio.run, not get_event_loop() — Python
    # 3.14 removed implicit loop creation in threads without a running loop.
    breaker.failure_threshold = 0.0
    breaker.min_requests = 1
    import asyncio
    asyncio.run(breaker.record_failure())

    r = client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    })
    assert r.status_code == 503
    assert r.headers.get("X-Backend-State") == "circuit-open"


def test_translate_returns_429_when_spend_cap_exceeded(app_and_state):
    app, _stub, _breaker, cap = app_and_state
    client = TestClient(app)
    cap.daily_cap_usd = 0.0   # any reservation is rejected

    r = client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    })
    assert r.status_code == 429
    assert r.headers.get("X-Backend-State") == "spend-cap-exceeded"


def test_translate_502_on_mistral_malformed(app_and_state):
    app, stub, *_ = app_and_state
    stub.chat_responses.append((200, {
        "choices": [{"message": {"content": "not-json"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }))
    client = TestClient(app)
    r = client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    })
    assert r.status_code == 502
    assert "mistral error" in r.json()["detail"]


def test_translate_502_on_mistral_transient_exhausted(app_and_state):
    app, stub, *_ = app_and_state
    # Three 500s — max_retries is 2 plus the initial attempt = 3 total.
    stub.chat_responses.extend([(500, {"error": "boom"})] * 5)
    client = TestClient(app)
    r = client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    })
    assert r.status_code == 502
    assert "transient" in r.json()["detail"]


def test_batch_endpoint_happy(app_and_state):
    app, _stub, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/translate/batch", json={
        "items": [
            {"text": "hello", "source_lang": "en"},
            {"text": "goodbye", "source_lang": "en"},
        ],
        "targets": ["fr"],
        "backend": "mistral",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert all("fr" in r["translations"] for r in body["results"])


def test_batch_endpoint_idempotency_key(app_and_state):
    app, stub, *_ = app_and_state
    client = TestClient(app)
    headers = {"Idempotency-Key": "key-abc-123"}
    payload = {
        "items": [{"text": "x", "source_lang": "en"}],
        "targets": ["fr"],
        "backend": "mistral",
    }
    r1 = client.post("/translate/batch", json=payload, headers=headers)
    r2 = client.post("/translate/batch", json=payload, headers=headers)
    assert r1.status_code == 200 and r2.status_code == 200
    # Second call returns cached response — only one chat invocation to the stub.
    assert sum(1 for c in stub.calls if c[0] == "chat") == 1


def test_embed_endpoint_happy(app_and_state):
    app, _stub, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/embed", json={"text": "hello", "backend": "mistral-embed"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "mistral-embed"
    assert body["dim"] == 1024
    assert len(body["vector"]) == 1024
    assert body["cached"] is False


def test_embed_cache_hit(app_and_state):
    app, stub, *_ = app_and_state
    client = TestClient(app)
    client.post("/embed", json={"text": "hi", "backend": "mistral-embed"})
    r2 = client.post("/embed", json={"text": "hi", "backend": "mistral-embed"})
    assert r2.json()["cached"] is True
    assert sum(1 for c in stub.calls if c[0] == "embed") == 1


def test_healthz_always_ok(app_and_state):
    app, *_ = app_and_state
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_metrics_exposes_prometheus(app_and_state):
    app, *_ = app_and_state
    client = TestClient(app)
    client.post("/translate", json={
        "text": "x", "source_lang": "en",
        "targets": ["fr"], "backend": "mistral",
    })
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "translations_total" in r.text


def test_keywords_endpoint_happy_path(app_and_state):
    app, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/keywords", json={
        "text": "the directive on combating violence against women",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lang"] == "en"
    assert body["keywords"] == ["directive", "combating", "violence", "women"]
    assert "the" in body["removed"]


def test_keywords_endpoint_rejects_blank_text(app_and_state):
    app, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/keywords", json={"text": "   "})
    assert r.status_code == 400


def test_keywords_endpoint_explicit_lang(app_and_state):
    app, *_ = app_and_state
    client = TestClient(app)
    r = client.post("/keywords", json={"text": "die Verordnung", "lang": "de"})
    assert r.status_code == 200
    assert r.json()["keywords"] == ["verordnung"]
