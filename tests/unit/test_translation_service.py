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
from src.backends.nebius import NebiusTransientError
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.spend_cap import SpendCap
from src.services.translation import TranslationService, UNDETERMINED_CACHE_KEY, cache_source


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


# ── Nebius routing ────────────────────────────────────────────────


@dataclass
class FakeNebius:
    """Reports a cost that differs from the estimate, like the real one."""

    cost_usd: float = 0.00035
    raises: Exception | None = None
    call_count: int = 0

    def estimate_chat_usd(self, text_chars, n_targets):
        return 0.0001

    async def translate_with_cost(self, text, source_lang, targets):
        self.call_count += 1
        if self.raises:
            raise self.raises
        return {t: f"{t}:{text}" for t in targets}, self.cost_usd

    async def translate(self, text, source_lang, targets):
        got, _cost = await self.translate_with_cost(text, source_lang, targets)
        return got


def _mk_nebius(nebius, cap_usd: float = 10.0, cache=None) -> TranslationService:
    return TranslationService(
        cache=cache or FakeCache(),
        mistral=None,
        nllb=None,
        mistral_breaker=CircuitBreaker(),
        mistral_spend_cap=SpendCap(daily_cap_usd=10.0),
        nebius=nebius,
        nebius_breaker=CircuitBreaker(),
        nebius_spend_cap=SpendCap(daily_cap_usd=cap_usd),
    )


async def test_nebius_backend_is_routed_to_and_cached():
    nebius = FakeNebius()
    svc = _mk_nebius(nebius)
    result = await svc.translate("Roboty", "pl", ["mt", "ga"], TranslationBackend.NEBIUS)
    assert result.translations == {"mt": "mt:Roboty", "ga": "ga:Roboty"}
    assert nebius.call_count == 1
    # Cached under its own backend key: a Nebius translation must not be
    # served as a Mistral one, or vice versa.
    again = await svc.translate("Roboty", "pl", ["mt", "ga"], TranslationBackend.NEBIUS)
    assert nebius.call_count == 1 and not again.cached_targets ^ {"mt", "ga"}


async def test_the_cap_settles_on_the_real_cost_not_the_estimate():
    """The estimate under-reserves — measured 0.00035 actual against a
    0.0001 guess. Without finalize a budget drifts by that ratio and a
    "EUR 5 run" is not one."""
    nebius = FakeNebius(cost_usd=0.00035)
    svc = _mk_nebius(nebius)
    await svc.translate("Roboty", "pl", ["mt"], TranslationBackend.NEBIUS)
    assert svc.nebius_spend_cap.spent_usd == pytest.approx(0.00035)


async def test_a_full_cap_refuses_before_spending_more():
    nebius = FakeNebius(cost_usd=0.5)
    svc = _mk_nebius(nebius, cap_usd=0.4)
    await svc.translate("a", "pl", ["mt"], TranslationBackend.NEBIUS)
    with pytest.raises(SpendCapExceeded):
        await svc.translate("b", "pl", ["mt"], TranslationBackend.NEBIUS)
    assert nebius.call_count == 1


async def test_a_failed_call_releases_its_reservation():
    nebius = FakeNebius(raises=NebiusTransientError("503"))
    svc = _mk_nebius(nebius)
    with pytest.raises(NebiusTransientError):
        await svc.translate("Roboty", "pl", ["mt"], TranslationBackend.NEBIUS)
    assert svc.nebius_spend_cap.spent_usd == 0.0


async def test_nebius_unconfigured_is_a_clear_refusal():
    svc = _mk_nebius(None)
    with pytest.raises(BackendUnavailable, match="nebius"):
        await svc.translate("Roboty", "pl", ["mt"], TranslationBackend.NEBIUS)


async def test_the_result_reports_what_the_call_cost():
    """A caller working to a budget accumulates this rather than estimating
    from its own token arithmetic — which is what makes a "EUR 5 run" a
    measurement instead of a hope."""
    svc = _mk_nebius(FakeNebius(cost_usd=0.00035))
    result = await svc.translate("Roboty", "pl", ["mt"], TranslationBackend.NEBIUS)
    assert result.cost_usd == pytest.approx(0.00035)


async def test_a_cache_hit_costs_nothing():
    cache = FakeCache()
    cache.translations[("Roboty", "pl", "mt", "nebius")] = "Xogħol"
    nebius = FakeNebius()
    svc = _mk_nebius(nebius, cache=cache)
    result = await svc.translate("Roboty", "pl", ["mt"], TranslationBackend.NEBIUS)
    assert result.cost_usd == 0.0 and nebius.call_count == 0


async def test_the_local_backend_is_free():
    svc = TranslationService(
        cache=FakeCache(), mistral=None, nllb=FakeMistral(),
        mistral_breaker=CircuitBreaker(), mistral_spend_cap=SpendCap(daily_cap_usd=1.0),
    )
    result = await svc.translate("Roboty", "pl", ["mt"], TranslationBackend.NLLB_LOCAL)
    assert result.cost_usd == 0.0


# ── undetermined source: the cache holds copies from the old prompt ──


async def test_copies_cached_by_the_old_undetermined_prompt_are_never_served():
    """The first identify-the-language prompt cached the source text as
    every target's translation, keyed 'und'. The revised prompt's results
    live under their own revision, so a retry reaches the model."""
    cache = FakeCache()
    title = "Interinstitutional fwc for agency staff"
    await cache.put_translations(title, "und", "nebius", {"de": title, "pt": title})
    nebius = FakeNebius()
    svc = _mk_nebius(nebius, cache=cache)

    result = await svc.translate(title, "und", ["de", "pt"], TranslationBackend.NEBIUS)

    assert nebius.call_count == 1
    assert result.translations == {"de": f"de:{title}", "pt": f"pt:{title}"}
    assert (title, UNDETERMINED_CACHE_KEY, "de", "nebius") in cache.translations


async def test_a_known_source_keeps_its_cache_key():
    """Only the undetermined path moved: known-source translations already
    cached stay cached, and are not paid for again."""
    cache = FakeCache()
    await cache.put_translations("Roboty", "pl", "nebius", {"de": "Bauarbeiten"})
    nebius = FakeNebius()
    svc = _mk_nebius(nebius, cache=cache)

    result = await svc.translate("Roboty", "pl", ["de"], TranslationBackend.NEBIUS)

    assert nebius.call_count == 0 and result.translations == {"de": "Bauarbeiten"}
    assert cache_source("pl") == "pl"
