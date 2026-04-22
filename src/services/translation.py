"""TranslationService — the composition point: cache → breaker → backend → cache."""
from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from src.backends.mistral import MistralBackend, MistralError, MistralTransientError
from src.backends.nllb_local import NllbLocalBackend
from src.cache.postgres import PostgresCache
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    TranslationBackend,
    TranslationResult,
)
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.metrics import TRANSLATION_LATENCY, TRANSLATIONS_TOTAL
from src.infra.spend_cap import SpendCap


@dataclass
class TranslationService:
    cache: PostgresCache
    mistral: MistralBackend | None
    nllb: NllbLocalBackend | None
    mistral_breaker: CircuitBreaker
    mistral_spend_cap: SpendCap

    async def translate(
        self,
        text: str,
        source_lang: str,
        targets: list[str],
        backend: TranslationBackend,
    ) -> TranslationResult:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        if not targets:
            raise ValueError("targets must be non-empty")

        backend_str = backend.value
        start = time.perf_counter()

        cached = await self.cache.get_translations(text, source_lang, targets, backend_str)
        missing = [t for t in targets if t not in cached]

        if not missing:
            for t in targets:
                TRANSLATIONS_TOTAL.labels(
                    backend=backend_str, cached="true", source_lang=source_lang,
                ).inc()
            TRANSLATION_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
            return TranslationResult(
                translations=cached,
                backend=backend,
                cached_targets=frozenset(cached.keys()),
            )

        fresh = await self._call_backend(text, source_lang, missing, backend)
        await self.cache.put_translations(text, source_lang, backend_str, fresh)

        merged = {**cached, **fresh}
        for t in cached:
            TRANSLATIONS_TOTAL.labels(
                backend=backend_str, cached="true", source_lang=source_lang,
            ).inc()
        for t in fresh:
            TRANSLATIONS_TOTAL.labels(
                backend=backend_str, cached="false", source_lang=source_lang,
            ).inc()
        TRANSLATION_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
        return TranslationResult(
            translations=merged,
            backend=backend,
            cached_targets=frozenset(cached.keys()),
        )

    async def _call_backend(
        self,
        text: str,
        source_lang: str,
        missing: list[str],
        backend: TranslationBackend,
    ) -> dict[str, str]:
        if backend is TranslationBackend.MISTRAL:
            return await self._call_mistral(text, source_lang, missing)
        return await self._call_nllb(text, source_lang, missing)

    async def _call_mistral(
        self, text: str, source_lang: str, missing: list[str]
    ) -> dict[str, str]:
        if self.mistral is None:
            raise BackendUnavailable("mistral backend not configured")
        if not await self.mistral_breaker.allow():
            raise CircuitOpen("mistral circuit breaker is open")

        estimate = self.mistral.estimate_chat_usd(len(text), len(missing))
        await self.mistral_spend_cap.reserve(estimate)

        try:
            result = await self.mistral.translate(text, source_lang, missing)
        except (MistralTransientError, MistralError) as exc:
            await self.mistral_breaker.record_failure()
            await self.mistral_spend_cap.release(estimate)
            logger.warning("mistral translate failure: {}", exc)
            raise
        await self.mistral_breaker.record_success()
        return result

    async def _call_nllb(
        self, text: str, source_lang: str, missing: list[str]
    ) -> dict[str, str]:
        if self.nllb is None:
            raise BackendUnavailable("nllb-local backend not configured")
        return await self.nllb.translate(text, source_lang, missing)
