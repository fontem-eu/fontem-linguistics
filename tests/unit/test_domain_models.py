"""Smoke tests for domain enums/dataclasses — ensures they stay importable and stable."""
from __future__ import annotations

from src.domain.models import (
    CircuitState,
    EmbeddingBackend,
    TranslationBackend,
    TranslationResult,
)


def test_enum_values_are_stable():
    # External consumers persist these values — don't rename silently.
    assert TranslationBackend.MISTRAL.value == "mistral"
    assert TranslationBackend.NLLB_LOCAL.value == "nllb-local"
    assert EmbeddingBackend.MISTRAL_EMBED.value == "mistral-embed"
    assert EmbeddingBackend.LABSE_LOCAL.value == "labse-local"
    assert CircuitState.OPEN.value == "open"
    assert CircuitState.CLOSED.value == "closed"
    assert CircuitState.HALF_OPEN.value == "half-open"


def test_translation_result_fully_cached_flag():
    r1 = TranslationResult(
        translations={"en": "hi", "fr": "salut"},
        backend=TranslationBackend.MISTRAL,
        cached_targets=frozenset({"en", "fr"}),
    )
    assert r1.fully_cached is True

    r2 = TranslationResult(
        translations={"en": "hi", "fr": "salut"},
        backend=TranslationBackend.MISTRAL,
        cached_targets=frozenset({"en"}),
    )
    assert r2.fully_cached is False
