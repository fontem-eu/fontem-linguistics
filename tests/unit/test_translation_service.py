"""TranslationService: cache/breaker/cap/backend composition. Fake backends."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.domain.models import (
    BackendUnavailable,
    CircuitOpen,
    SpendCapExceeded,
    TranslationBackend,
)
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.spend_cap import SpendCap
from src.services.translation import TranslationService


pytestmark = pytest.mark.asyncio


# ── Fake cache ────────────────────────────────────────────────────

@dataclass
class FakeCache:
    translations: dict[tuple[str, str, str, str], str] = field(default_factory=dict)
    put_calls: list[tuple[str, str, str, dict[str, str]]] = field(default_factory=list)
    embeddings: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    async def get_translations(self, text, source_lang, targets, backend):
        return {
            t: self.translations[(text, source_lang, t, backend)]
            for t in targets
            if (text, source_lang, t, backend) in self.translations
        }

    async def put_translations(self, text, source_lang, backend, translations):
        self.put_calls.append((text, source_lang, backend, dict(translations)))
        for t, v in translations.items():
            self.translations[(text, source_lang, t, backend)] = v

    async def get_embedding(self, text, backend):
        return self.embeddings.get((text, backend))

    async def put_embedding(self, text, backend, vector):
        self.embeddings[(text, backend)] = list(vector)


# ── Fake Mistral backend ──────────────────────────────────────────

@dataclass
class FakeMistral:
    translations: dict[str, str] = field(default_factory=dict)
    raises: Exception | None = None
    call_count: int = 0
    last_targets: list[str] | None = None

    def estimate_chat_usd(self, text_chars, n_targets):
        return 0.0001

    def estimate_embed_usd(self, text_chars):
        return 0.00005

    async def translate(self, text, source_lang, targets):
        self.call_count += 1
        self.last_targets = list(targets)
        if self.raises:
            raise self.raises
        return {t: self.translations.get(t, f"{t}:{text}") for t in targets}

    async def embed(self, text):
        self.call_count += 1
        if self.raises:
            raise self.raises
        return [0.1] * 1024


def _mk(cache=None, mistral=None, nllb=None) -> TranslationService:
    return TranslationService(
        cache=cache or FakeCache(),
        mistral=mistral,
        nllb=nllb,
        mistral_breaker=CircuitBreaker(),
        mistral_spend_cap=SpendCap(daily_cap_usd=10.0),
    )


async def test_full_cache_hit_skips_backend():
    cache = FakeCache()
    cache.translations[("hi", "en", "fr", "mistral")] = "salut"
    mistral = FakeMistral()
    svc = _mk(cache, mistral)

    result = await svc.translate("hi", "en", ["fr"], TranslationBackend.MISTRAL)

    assert result.translations == {"fr": "salut"}
    assert result.fully_cached is True
    assert mistral.call_count == 0
    assert not cache.put_calls


async def test_partial_cache_only_fetches_missing():
    cache = FakeCache()
    cache.translations[("hi", "en", "fr", "mistral")] = "salut"
    # en and de not cached
    mistral = FakeMistral(translations={"en": "hi", "de": "hallo"})
    svc = _mk(cache, mistral)

    result = await svc.translate("hi", "en", ["fr", "en", "de"], TranslationBackend.MISTRAL)

    assert result.translations == {"fr": "salut", "en": "hi", "de": "hallo"}
    assert result.fully_cached is False
    assert mistral.last_targets == ["en", "de"]      # only missing targets
    # put_calls must only have en+de, not fr (already cached)
    assert cache.put_calls == [("hi", "en", "mistral", {"en": "hi", "de": "hallo"})]


async def test_full_miss_calls_backend_and_writes_all():
    cache = FakeCache()
    mistral = FakeMistral(translations={"fr": "salut", "de": "hallo"})
    svc = _mk(cache, mistral)

    result = await svc.translate("hi", "en", ["fr", "de"], TranslationBackend.MISTRAL)

    assert result.translations == {"fr": "salut", "de": "hallo"}
    assert result.fully_cached is False
    assert mistral.last_targets == ["fr", "de"]


async def test_empty_text_raises():
    svc = _mk()
    with pytest.raises(ValueError):
        await svc.translate("", "en", ["fr"], TranslationBackend.MISTRAL)
    with pytest.raises(ValueError):
        await svc.translate("   ", "en", ["fr"], TranslationBackend.MISTRAL)


async def test_empty_targets_raises():
    svc = _mk()
    with pytest.raises(ValueError):
        await svc.translate("hi", "en", [], TranslationBackend.MISTRAL)


async def test_missing_mistral_raises_backend_unavailable():
    svc = _mk(mistral=None)
    with pytest.raises(BackendUnavailable):
        await svc.translate("hi", "en", ["fr"], TranslationBackend.MISTRAL)


async def test_missing_nllb_raises_backend_unavailable():
    svc = _mk(nllb=None)
    with pytest.raises(BackendUnavailable):
        await svc.translate("hi", "en", ["fr"], TranslationBackend.NLLB_LOCAL)


async def test_open_breaker_raises_circuit_open():
    mistral = FakeMistral(translations={"fr": "salut"})
    svc = _mk(mistral=mistral)
    # Trip the breaker directly.
    svc.mistral_breaker.failure_threshold = 0.0
    svc.mistral_breaker.min_requests = 1
    await svc.mistral_breaker.record_failure()
    from src.domain.models import CircuitState
    assert await svc.mistral_breaker.state() is CircuitState.OPEN

    with pytest.raises(CircuitOpen):
        await svc.translate("hi", "en", ["fr"], TranslationBackend.MISTRAL)
    assert mistral.call_count == 0


async def test_spend_cap_exceeded_raises_and_releases():
    from src.backends.mistral import MistralTransientError
    mistral = FakeMistral(raises=MistralTransientError("boom"))
    cache = FakeCache()
    cap = SpendCap(daily_cap_usd=0.00005)    # very tight: estimate 0.0001 can't fit
    svc = TranslationService(
        cache=cache, mistral=mistral, nllb=None,
        mistral_breaker=CircuitBreaker(),
        mistral_spend_cap=cap,
    )
    with pytest.raises(SpendCapExceeded):
        await svc.translate("hi", "en", ["fr"], TranslationBackend.MISTRAL)
    assert mistral.call_count == 0


async def test_transient_failure_records_breaker_and_releases_cap():
    from src.backends.mistral import MistralTransientError
    mistral = FakeMistral(raises=MistralTransientError("timeout"))
    cache = FakeCache()
    cap = SpendCap(daily_cap_usd=10.0)
    svc = TranslationService(
        cache=cache, mistral=mistral, nllb=None,
        mistral_breaker=CircuitBreaker(failure_threshold=1.0, min_requests=1000),
        mistral_spend_cap=cap,
    )
    with pytest.raises(MistralTransientError):
        await svc.translate("hi", "en", ["fr"], TranslationBackend.MISTRAL)
    # Reservation must have been released on failure.
    assert await cap.spent() == pytest.approx(0.0)


async def test_hard_error_records_breaker_and_releases_cap():
    from src.backends.mistral import MistralError
    mistral = FakeMistral(raises=MistralError("malformed"))
    cache = FakeCache()
    cap = SpendCap(daily_cap_usd=10.0)
    svc = TranslationService(
        cache=cache, mistral=mistral, nllb=None,
        mistral_breaker=CircuitBreaker(failure_threshold=1.0, min_requests=1000),
        mistral_spend_cap=cap,
    )
    with pytest.raises(MistralError):
        await svc.translate("hi", "en", ["fr"], TranslationBackend.MISTRAL)
    assert await cap.spent() == pytest.approx(0.0)


class _FakeNllb:
    async def translate(self, text, source_lang, targets):
        return {t: f"[{t}]{text}" for t in targets}


async def test_nllb_path_happy():
    svc = _mk(nllb=_FakeNllb())
    result = await svc.translate("hi", "en", ["fr"], TranslationBackend.NLLB_LOCAL)
    assert result.translations == {"fr": "[fr]hi"}
