"""Domain types — plain values, no IO, no framework dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TranslationBackend(str, Enum):
    MISTRAL = "mistral"
    NEBIUS = "nebius"
    NLLB_LOCAL = "nllb-local"


class EmbeddingBackend(str, Enum):
    MISTRAL_EMBED = "mistral-embed"
    LABSE_LOCAL = "labse-local"
    MINILM_LOCAL = "minilm-local"


@dataclass(frozen=True)
class TranslationRequest:
    text: str
    source_lang: str
    targets: tuple[str, ...]
    backend: TranslationBackend


@dataclass(frozen=True)
class TranslationResult:
    translations: dict[str, str]   # target_lang -> translation
    backend: TranslationBackend
    cached_targets: frozenset[str]

    @property
    def fully_cached(self) -> bool:
        return self.cached_targets == frozenset(self.translations.keys())


@dataclass(frozen=True)
class EmbeddingRequest:
    text: str
    backend: EmbeddingBackend


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    dim: int
    backend: EmbeddingBackend
    cached: bool
    # Stamps the signed-mirror identity of the encoder that produced this
    # vector, so downstream stores can reject cross-version comparisons.
    # Format: "{backend}@{mirror_version}-{hf_sha7}", e.g.
    # "labse@1.0.0-836121a". Mistral-embed (a hosted API we don't mirror)
    # reports a best-effort "mistral-embed@api".
    encoder_id: str


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class SpendCapExceeded(Exception):
    """Raised when the configured daily spend cap for a paid backend is hit."""


class CircuitOpen(Exception):
    """Raised when the backend circuit breaker is open."""


class BackendUnavailable(Exception):
    """Raised when a backend's dependencies or runtime state preclude use."""
