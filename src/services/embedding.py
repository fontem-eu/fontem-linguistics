"""EmbeddingService — cache → breaker → backend → cache, same pattern as translation."""
from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from src.backends.labse_local import LabseLocalBackend
from src.backends.minilm_local import MinilmLocalBackend
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
    minilm: MinilmLocalBackend | None
    mistral_breaker: CircuitBreaker
    mistral_spend_cap: SpendCap

    async def embed(self, text: str, backend: EmbeddingBackend) -> EmbeddingResult:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        backend_str = backend.value
        start = time.perf_counter()
        encoder_id = self._encoder_id(backend)

        cached = await self.cache.get_embedding(text, backend_str)
        if cached is not None:
            EMBEDDINGS_TOTAL.labels(backend=backend_str, cached="true").inc()
            EMBEDDING_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
            return EmbeddingResult(
                vector=cached, dim=len(cached), backend=backend,
                cached=True, encoder_id=encoder_id,
            )

        vec = await self._call_backend(text, backend)
        await self.cache.put_embedding(text, backend_str, vec)
        EMBEDDINGS_TOTAL.labels(backend=backend_str, cached="false").inc()
        EMBEDDING_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
        return EmbeddingResult(
            vector=vec, dim=len(vec), backend=backend,
            cached=False, encoder_id=encoder_id,
        )

    async def embed_batch(
        self, texts: list[str], backend: EmbeddingBackend,
    ) -> list[EmbeddingResult]:
        """Batched /embed. Cache hits skipped; misses batched into one
        model.encode() call on the backend for BLAS-parallel throughput.
        Only supported for local (SentenceTransformer) backends today;
        mistral falls back to per-text calls (still saves HTTP hops)."""
        if not texts:
            return []
        backend_str = backend.value
        encoder_id = self._encoder_id(backend)

        # Fetch cached vectors up front.
        cached_vecs: list[list[float] | None] = []
        for t in texts:
            if not t or not t.strip():
                raise ValueError("text must be non-empty")
            cached_vecs.append(await self.cache.get_embedding(t, backend_str))

        # Batch-encode the misses on the backend.
        miss_idx = [i for i, v in enumerate(cached_vecs) if v is None]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            miss_vecs = await self._call_backend_batch(miss_texts, backend)
            for i, v in zip(miss_idx, miss_vecs):
                cached_vecs[i] = v
                await self.cache.put_embedding(texts[i], backend_str, v)
                EMBEDDINGS_TOTAL.labels(backend=backend_str, cached="false").inc()
        for i in range(len(texts)):
            if i not in miss_idx:
                EMBEDDINGS_TOTAL.labels(backend=backend_str, cached="true").inc()

        return [
            EmbeddingResult(
                vector=v, dim=len(v), backend=backend,
                cached=(i not in miss_idx), encoder_id=encoder_id,
            )
            for i, v in enumerate(cached_vecs)
        ]

    async def _call_backend_batch(
        self, texts: list[str], backend: EmbeddingBackend,
    ) -> list[list[float]]:
        if backend is EmbeddingBackend.LABSE_LOCAL:
            if self.labse is None:
                raise BackendUnavailable("labse-local backend not configured")
            return await self.labse.embed_batch(texts)
        if backend is EmbeddingBackend.MINILM_LOCAL:
            if self.minilm is None:
                raise BackendUnavailable("minilm-local backend not configured")
            return await self.minilm.embed_batch(texts)
        # Mistral: no server-side batch; loop with per-text /embed. Still
        # saves the HTTP overhead on the linguistics ↔ client hop.
        return [await self._call_backend(t, backend) for t in texts]

    def _encoder_id(self, backend: EmbeddingBackend) -> str:
        """Resolve the signed-mirror identity of the active encoder.

        Cached rows from a previous encoder version get re-stamped with
        the current encoder_id — the cache keys by (text, backend), not
        by version. That's acceptable today because we never bump the
        encoder within a backend without also wiping the cache; see the
        migration notes in the versioned-encoder plan.
        """
        if backend is EmbeddingBackend.MISTRAL_EMBED:
            if self.mistral is None:
                raise BackendUnavailable("mistral backend not configured")
            return self.mistral.embed_encoder_id
        if backend is EmbeddingBackend.LABSE_LOCAL:
            if self.labse is None:
                raise BackendUnavailable("labse-local backend not configured")
            return self.labse.encoder_id
        if backend is EmbeddingBackend.MINILM_LOCAL:
            if self.minilm is None:
                raise BackendUnavailable("minilm-local backend not configured")
            return self.minilm.encoder_id
        raise BackendUnavailable(f"unknown embedding backend: {backend!r}")

    async def _call_backend(self, text: str, backend: EmbeddingBackend) -> list[float]:
        if backend is EmbeddingBackend.MISTRAL_EMBED:
            return await self._call_mistral(text)
        if backend is EmbeddingBackend.LABSE_LOCAL:
            return await self._call_labse(text)
        if backend is EmbeddingBackend.MINILM_LOCAL:
            return await self._call_minilm(text)
        raise BackendUnavailable(f"unknown embedding backend: {backend!r}")

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

    async def _call_minilm(self, text: str) -> list[float]:
        if self.minilm is None:
            raise BackendUnavailable("minilm-local backend not configured")
        return await self.minilm.embed(text)
