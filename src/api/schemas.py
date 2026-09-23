"""Request/response models for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import EmbeddingBackend, TranslationBackend


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8192)
    source_lang: str = Field(min_length=2, max_length=8)
    targets: list[str] = Field(min_length=1, max_length=24)
    backend: TranslationBackend


class TranslateResponse(BaseModel):
    cached: bool
    backend: TranslationBackend
    translations: dict[str, str]
    partial_cached_targets: list[str] = []
    #: What this call cost, as the provider reported it. Zero on a cache hit
    #: and on the local backends. A caller running to a budget accumulates
    #: this rather than estimating from its own token arithmetic.
    cost_usd: float = 0.0
    #: Why this item came back without translations, when it did. A batch
    #: completes even if some items fail, so the caller needs to tell a spent
    #: budget from a provider hiccup from an open breaker — they call for
    #: different responses.
    error: str | None = None


class BatchTranslateItem(BaseModel):
    text: str = Field(min_length=1, max_length=8192)
    source_lang: str = Field(min_length=2, max_length=8)


class BatchTranslateRequest(BaseModel):
    items: list[BatchTranslateItem] = Field(min_length=1, max_length=256)
    targets: list[str] = Field(min_length=1, max_length=24)
    backend: TranslationBackend


class BatchTranslateResponse(BaseModel):
    results: list[TranslateResponse]


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8192)
    backend: EmbeddingBackend


class EmbedResponse(BaseModel):
    cached: bool
    backend: EmbeddingBackend
    dim: int
    vector: list[float]
    # Signed-mirror identity of the encoder. See EmbeddingResult.encoder_id.
    encoder_id: str




class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None


class ModelInfoResponse(BaseModel):
    backend: str
    kind: str
    quality_score: float
    dim: int | None
    languages_supported: int
    cost_tier: str
    description: str
    # encoder_id is only meaningful for embedders; None for translators.
    encoder_id: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfoResponse]


class LanguageInfo(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]

class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    backend: EmbeddingBackend


class EmbedBatchResponse(BaseModel):
    backend: EmbeddingBackend
    dim: int
    encoder_id: str
    results: list[EmbedResponse]
