"""EmbeddingService — cache → breaker → backend → cache, same pattern as translation."""
from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from src.backends.labse_local import LabseLocalBackend
from src.backends.mistral import MistralBackend, MistralError, MistralTransientError
from src.cache.postgres import PostgresCache
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    EmbeddingBackend,
    EmbeddingResult,
)
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.metrics import EMBEDDING_LATENCY, EMBEDDINGS_TOTAL
from src.infra.spend_cap import SpendCap


@dataclass
class EmbeddingService:
    cache: PostgresCache
    mistral: MistralBackend | None
    labse: LabseLocalBackend | None
    mistral_breaker: CircuitBreaker
    mistral_spend_cap: SpendCap

    async def embed(self, text: str, backend: EmbeddingBackend) -> EmbeddingResult:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        backend_str = backend.value
        start = time.perf_counter()

        cached = await self.cache.get_embedding(text, backend_str)
        if cached is not None:
            EMBEDDINGS_TOTAL.labels(backend=backend_str, cached="true").inc()
            EMBEDDING_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
            return EmbeddingResult(vector=cached, dim=len(cached), backend=backend, cached=True)

        vec = await self._call_backend(text, backend)
        await self.cache.put_embedding(text, backend_str, vec)
        EMBEDDINGS_TOTAL.labels(backend=backend_str, cached="false").inc()
        EMBEDDING_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
        return EmbeddingResult(vector=vec, dim=len(vec), backend=backend, cached=False)

    async def _call_backend(self, text: str, backend: EmbeddingBackend) -> list[float]:
        if backend is EmbeddingBackend.MISTRAL_EMBED:
            return await self._call_mistral(text)
        return await self._call_labse(text)

    async def _call_mistral(self, text: str) -> list[float]:
        if self.mistral is None:
            raise BackendUnavailable("mistral backend not configured")
        if not await self.mistral_breaker.allow():
            raise CircuitOpen("mistral circuit breaker is open")

        estimate = self.mistral.estimate_embed_usd(len(text))
        await self.mistral_spend_cap.reserve(estimate)

        try:
            vec = await self.mistral.embed(text)
        except (MistralTransientError, MistralError) as exc:
            await self.mistral_breaker.record_failure()
            await self.mistral_spend_cap.release(estimate)
            logger.warning("mistral embed failure: {}", exc)
            raise
        await self.mistral_breaker.record_success()
        return vec

    async def _call_labse(self, text: str) -> list[float]:
        if self.labse is None:
            raise BackendUnavailable("labse-local backend not configured")
        return await self.labse.embed(text)
