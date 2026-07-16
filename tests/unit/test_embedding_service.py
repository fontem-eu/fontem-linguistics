"""EmbeddingService composition: cache hit vs miss, breaker + cap."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    EmbeddingBackend,
)
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.spend_cap import SpendCap
from src.services.embedding import EmbeddingService


pytestmark = pytest.mark.asyncio


@dataclass
class FakeCache:
    embeddings: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    async def get_embedding(self, text, backend):
        return self.embeddings.get((text, backend))

    async def put_embedding(self, text, backend, vector):
        self.embeddings[(text, backend)] = list(vector)


@dataclass
class FakeMistral:
    vector: list[float] = field(default_factory=lambda: [0.5] * 1024)
    raises: Exception | None = None
    call_count: int = 0
    embed_encoder_id: str = "mistral-embed@api-mistral-embed"

    def estimate_embed_usd(self, text_chars):
        return 0.00001

    async def embed(self, text):
        self.call_count += 1
        if self.raises:
            raise self.raises
        return list(self.vector)


class _FakeLabse:
    def __init__(self, vec=None, encoder_id="labse@1.0.0-test000"):
        self.vec = vec or [0.1] * 768
        self.call_count = 0
        self.encoder_id = encoder_id

    async def embed(self, text):
        self.call_count += 1
        return list(self.vec)


def _mk(cache=None, mistral=None, labse=None, minilm=None) -> EmbeddingService:
    return EmbeddingService(
        cache=cache or FakeCache(),
        mistral=mistral,
        labse=labse,
        minilm=minilm,
        mistral_breaker=CircuitBreaker(),
        mistral_spend_cap=SpendCap(daily_cap_usd=10.0),
    )


async def test_cache_hit_skips_backend():
    cache = FakeCache()
    cache.embeddings[("hi", "mistral-embed")] = [0.7] * 1024
    mistral = FakeMistral()
    svc = _mk(cache, mistral)

    result = await svc.embed("hi", EmbeddingBackend.MISTRAL_EMBED)

    assert result.cached is True
    assert result.dim == 1024
    assert result.vector == [0.7] * 1024
    assert mistral.call_count == 0


async def test_cache_miss_calls_mistral_and_stores():
    cache = FakeCache()
    mistral = FakeMistral(vector=[0.3] * 1024)
    svc = _mk(cache, mistral)

    result = await svc.embed("hi", EmbeddingBackend.MISTRAL_EMBED)

    assert result.cached is False
    assert result.vector == [0.3] * 1024
    assert mistral.call_count == 1
    assert ("hi", "mistral-embed") in cache.embeddings


async def test_cache_miss_calls_labse():
    labse = _FakeLabse([0.9] * 768)
    svc = _mk(labse=labse)
    result = await svc.embed("hi", EmbeddingBackend.LABSE_LOCAL)
    assert result.dim == 768
    assert labse.call_count == 1
    # Encoder identity on every embed — cache miss path.
    assert result.encoder_id == "labse@1.0.0-test000"


async def test_labse_encoder_id_on_cache_hit():
    cache = FakeCache()
    cache.embeddings[("hi", "labse-local")] = [0.4] * 768
    labse = _FakeLabse(encoder_id="labse@2.3.0-abc1234")
    svc = _mk(cache, labse=labse)
    result = await svc.embed("hi", EmbeddingBackend.LABSE_LOCAL)
    assert result.cached is True
    assert result.encoder_id == "labse@2.3.0-abc1234"
    assert labse.call_count == 0


async def test_mistral_encoder_id_surfaces():
    mistral = FakeMistral()
    mistral.embed_encoder_id = "mistral-embed@api-mistral-embed"
    svc = _mk(mistral=mistral)
    result = await svc.embed("hi", EmbeddingBackend.MISTRAL_EMBED)
    assert result.encoder_id == "mistral-embed@api-mistral-embed"


async def test_empty_text_raises():
    svc = _mk(mistral=FakeMistral())
    with pytest.raises(ValueError):
        await svc.embed("", EmbeddingBackend.MISTRAL_EMBED)


async def test_missing_mistral_raises():
    svc = _mk(mistral=None)
    with pytest.raises(BackendUnavailable):
        await svc.embed("x", EmbeddingBackend.MISTRAL_EMBED)


async def test_missing_labse_raises():
    svc = _mk(labse=None)
    with pytest.raises(BackendUnavailable):
        await svc.embed("x", EmbeddingBackend.LABSE_LOCAL)


async def test_open_breaker_blocks_mistral():
    mistral = FakeMistral()
    svc = _mk(mistral=mistral)
    svc.mistral_breaker.failure_threshold = 0.0
    svc.mistral_breaker.min_requests = 1
    await svc.mistral_breaker.record_failure()

    with pytest.raises(CircuitOpen):
        await svc.embed("x", EmbeddingBackend.MISTRAL_EMBED)
    assert mistral.call_count == 0


async def test_transient_failure_releases_reservation():
    from src.backends.mistral import MistralTransientError
    mistral = FakeMistral(raises=MistralTransientError("x"))
    cap = SpendCap(daily_cap_usd=10.0)
    svc = EmbeddingService(
        cache=FakeCache(), mistral=mistral, labse=None, minilm=None,
        mistral_breaker=CircuitBreaker(failure_threshold=1.0, min_requests=1000),
        mistral_spend_cap=cap,
    )
    with pytest.raises(MistralTransientError):
        await svc.embed("x", EmbeddingBackend.MISTRAL_EMBED)
    assert await cap.spent() == pytest.approx(0.0)
