"""TranslationService — the composition point: cache → breaker → backend → cache."""
from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from src.backends.mistral import MistralBackend, MistralError, MistralTransientError
from src.backends.nebius import NebiusBackend, NebiusError, NebiusTransientError
from src.backends.nllb_local import NllbLocalBackend
from src.backends.openai_chat import LANG_FULLNAMES, UNDETERMINED
from src.cache.postgres import PostgresCache
from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    DetectionResult,
    SummaryResult,
    TranslationBackend,
    TranslationResult,
)
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.metrics import TRANSLATION_LATENCY, TRANSLATIONS_TOTAL
from src.infra.spend_cap import SpendCap


#: Cache key for results of the identify-the-language-yourself prompt. Its
#: first wording cached the source text as every target's "translation", and
#: the cache key does not include the prompt: keying the revised prompt's
#: output under its own revision means a retry can never be answered with
#: those copies. Bump the suffix whenever that prompt's wording changes.
UNDETERMINED_CACHE_KEY = f"{UNDETERMINED}#2"


#: The model every Nebius translation cached before the model became part of
#: the cache key was made by: its entries keep the bare "nebius" key, so a
#: switch back to it is served from them instead of paid for again.
LEGACY_NEBIUS_MODEL = "google/gemma-3-27b-it"


def cache_source(source_lang: str) -> str:
    """The source-language component of the cache key."""
    return UNDETERMINED_CACHE_KEY if source_lang == UNDETERMINED else source_lang


#: Bump when the summary prompt's wording changes: summaries are cached in
#: the translations table under their own source key, and an old wording's
#: summaries must not answer for the new one.
SUMMARY_CACHE_TAG = "summary#1"


def summary_cache_source(lang: str, max_chars: int, about: str | None) -> str:
    """The source-language component of a summary's cache key: what the
    summary was asked for (language, length, subject) rather than a source
    language, so a summary never answers for a translation or another ask."""
    return f"{SUMMARY_CACHE_TAG}:{lang}:{max_chars}:{about or ''}"


def summary_language(source_lang: str) -> str:
    """The language a summary is written in: the source's, or English when
    the source's is undetermined or not one of the EU's."""
    return source_lang if source_lang in LANG_FULLNAMES else "en"


# The composition point for every translation path: one cache, and per
# provider a backend, a breaker and a budget. Splitting them into
# per-provider structs would hide that they are wired identically.
@dataclass
class TranslationService:  # pylint: disable=too-many-instance-attributes
    cache: PostgresCache
    mistral: MistralBackend | None
    nllb: NllbLocalBackend | None
    mistral_breaker: CircuitBreaker
    mistral_spend_cap: SpendCap
    # Nebius arrives with its own breaker and cap: one provider degrading
    # must not trip the other, and the budgets are not shared.
    nebius: NebiusBackend | None = None
    nebius_breaker: CircuitBreaker | None = None
    nebius_spend_cap: SpendCap | None = None

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
        cache_key = self.cache_backend(backend)
        start = time.perf_counter()

        cached = await self.cache.get_translations(
            text, cache_source(source_lang), targets, cache_key)
        missing = [t for t in targets if t not in cached]

        if not missing:
            for _ in targets:
                TRANSLATIONS_TOTAL.labels(
                    backend=backend_str, cached="true", source_lang=source_lang,
                ).inc()
            TRANSLATION_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
            return TranslationResult(
                translations=cached,
                backend=backend,
                cached_targets=frozenset(cached.keys()),
            )

        fresh, cost_usd = await self._call_backend(text, source_lang, missing, backend)
        await self.cache.put_translations(text, cache_source(source_lang), cache_key, fresh)

        merged = {**cached, **fresh}
        for _ in cached:
            TRANSLATIONS_TOTAL.labels(
                backend=backend_str, cached="true", source_lang=source_lang,
            ).inc()
        for _ in fresh:
            TRANSLATIONS_TOTAL.labels(
                backend=backend_str, cached="false", source_lang=source_lang,
            ).inc()
        TRANSLATION_LATENCY.labels(backend=backend_str).observe(time.perf_counter() - start)
        return TranslationResult(
            translations=merged,
            backend=backend,
            cached_targets=frozenset(cached.keys()),
            cost_usd=cost_usd,
        )

    async def summarize(  # pylint: disable=too-many-arguments
        self,
        text: str,
        source_lang: str,
        targets: list[str],
        backend: TranslationBackend,
        *,
        max_chars: int = 280,
        about: str | None = None,
    ) -> SummaryResult:
        """A summary of at most ``max_chars`` characters in the source's
        language, then machine-translated into ``targets`` the way any text
        is (cache, breaker, budget; JSON with the tagged fallback). The
        summary is cached too, so a second environment asking for the same
        text pays nothing."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        if backend is not TranslationBackend.NEBIUS:
            raise ValueError("summaries are written by the nebius backend only")
        lang = summary_language(source_lang)
        summary, cost, cached = await self._summary(text, lang, max_chars, about, backend)
        summaries = {lang: summary}
        others = [t for t in targets if t != lang]
        if others:
            translated = await self.translate(summary, lang, others, backend)
            summaries.update(translated.translations)
            cost += translated.cost_usd
        return SummaryResult(summaries=summaries, lang=lang, backend=backend,
                             cached=cached, cost_usd=cost)

    async def _summary(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, text: str, lang: str, max_chars: int, about: str | None,
        backend: TranslationBackend,
    ) -> tuple[str, float, bool]:
        """(the summary in ``lang``, what writing it cost, whether the cache had it)."""
        key, cache_key = summary_cache_source(lang, max_chars, about), self.cache_backend(backend)
        hit = await self.cache.get_translations(text, key, [lang], cache_key)
        if lang in hit:
            return hit[lang], 0.0, True
        summary, cost = await self._call_nebius_summary(text, lang, max_chars, about)
        await self.cache.put_translations(text, key, cache_key, {lang: summary})
        return summary, cost, False

    async def _call_nebius_summary(self, text: str, lang: str, max_chars: int,
                                   about: str | None) -> tuple[str, float]:
        """Breaker, reserve, call, settle — as for a translation."""
        await self._nebius_ready()
        estimate = self.nebius.estimate_chat_usd(len(text), 2)
        await self.nebius_spend_cap.reserve(estimate)
        try:
            summary, actual = await self.nebius.summarize_with_cost(text, lang, max_chars, about)
        except (NebiusTransientError, NebiusError) as exc:
            await self.nebius_breaker.record_failure()
            await self.nebius_spend_cap.release(estimate)
            logger.warning("nebius summarize failure: {}", exc)
            raise
        await self.nebius_spend_cap.finalize(estimate, actual)
        await self.nebius_breaker.record_success()
        return summary, actual

    async def _nebius_ready(self) -> None:
        """Raise unless Nebius is configured and its breaker lets a call through."""
        if self.nebius is None or self.nebius_breaker is None or self.nebius_spend_cap is None:
            raise BackendUnavailable("nebius backend not configured")
        if not await self.nebius_breaker.allow():
            raise CircuitOpen("nebius circuit breaker is open")

    def cache_backend(self, backend: TranslationBackend) -> str:
        """The backend component of the cache key. For Nebius it names the
        model, so a translation is only reused while the model that made it
        is the one configured; the legacy model keeps the bare key."""
        if (backend is TranslationBackend.NEBIUS and self.nebius is not None
                and self.nebius.chat_model != LEGACY_NEBIUS_MODEL):
            return f"{backend.value}:{self.nebius.chat_model}"
        return backend.value

    async def detect(self, texts: list[str], backend: TranslationBackend) -> DetectionResult:
        """Identify each text's language in one call. Nebius only.

        Not cached: a caller that needs the answer again keeps it, and the
        call costs a fraction of a translation. Same breaker and budget as
        Nebius translation, so detection cannot outspend the daily cap.
        """
        if not texts or any(not t or not t.strip() for t in texts):
            raise ValueError("every text must be non-empty")
        if backend is not TranslationBackend.NEBIUS:
            raise ValueError("language detection is served by the nebius backend only")
        await self._nebius_ready()

        estimate = self.nebius.estimate_detect_usd(sum(len(t) for t in texts), len(texts))
        await self.nebius_spend_cap.reserve(estimate)
        try:
            langs, actual = await self.nebius.detect_with_cost(texts)
        except (NebiusTransientError, NebiusError) as exc:
            await self.nebius_breaker.record_failure()
            await self.nebius_spend_cap.release(estimate)
            logger.warning("nebius detect failure: {}", exc)
            raise
        await self.nebius_spend_cap.finalize(estimate, actual)
        await self.nebius_breaker.record_success()
        return DetectionResult(langs=langs, model=f"nebius:{self.nebius.chat_model}",
                               cost_usd=actual)

    async def _call_backend(
        self,
        text: str,
        source_lang: str,
        missing: list[str],
        backend: TranslationBackend,
    ) -> tuple[dict[str, str], float]:
        """``(translations, what it cost)``. Cost is zero where nothing was
        billed — the local backends, and any path that only read cache."""
        if backend is TranslationBackend.MISTRAL:
            return await self._call_mistral(text, source_lang, missing), 0.0
        if backend is TranslationBackend.NEBIUS:
            return await self._call_nebius(text, source_lang, missing)
        return await self._call_nllb(text, source_lang, missing), 0.0

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

    async def _call_nebius(
        self, text: str, source_lang: str, missing: list[str]
    ) -> tuple[dict[str, str], float]:
        """Same shape as Mistral: breaker, reserve, call, settle.

        The reservation is an estimate; `finalize` replaces it with what the
        provider's usage block actually charged, so a title that translates
        longer than guessed cannot walk past the cap unnoticed.
        """
        await self._nebius_ready()

        estimate = self.nebius.estimate_chat_usd(len(text), len(missing))
        await self.nebius_spend_cap.reserve(estimate)

        try:
            result, actual = await self.nebius.translate_with_cost(text, source_lang, missing)
        except (NebiusTransientError, NebiusError) as exc:
            await self.nebius_breaker.record_failure()
            await self.nebius_spend_cap.release(estimate)
            logger.warning("nebius translate failure: {}", exc)
            raise
        await self.nebius_spend_cap.finalize(estimate, actual)
        await self.nebius_breaker.record_success()
        return result, actual

    async def _call_nllb(
        self, text: str, source_lang: str, missing: list[str]
    ) -> dict[str, str]:
        if self.nllb is None:
            raise BackendUnavailable("nllb-local backend not configured")
        return await self.nllb.translate(text, source_lang, missing)
