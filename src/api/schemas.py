"""Request/response models for the HTTP API."""
from __future__ import annotations

import datetime as dt
from typing import Literal

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


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8192)
    source_lang: str = Field(min_length=2, max_length=8)
    #: Languages to translate the summary into, besides its own.
    targets: list[str] = Field(default_factory=list, max_length=24)
    #: A tweet is 280.
    max_chars: int = Field(default=280, ge=80, le=1000)
    #: What the summary should say, e.g. "what the organisation lobbies for".
    about: str | None = Field(default=None, max_length=120)
    backend: TranslationBackend = TranslationBackend.NEBIUS


class SummarizeResponse(BaseModel):
    #: The language the summary was written in (the source's, or English).
    lang: str
    #: language -> summary, its own language included.
    summaries: dict[str, str]
    #: The summary itself came from the cache.
    cached: bool
    backend: TranslationBackend
    cost_usd: float = 0.0


class DetectRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    backend: TranslationBackend = TranslationBackend.NEBIUS


class DetectedLanguage(BaseModel):
    #: ISO 639-1 code; "und" where the text has no words to judge by; None
    #: where no usable answer came back (then `error` says why).
    lang: str | None
    error: str | None = None


class DetectResponse(BaseModel):
    backend: TranslationBackend
    #: "<backend>:<model>" that answered, for callers that record provenance.
    model: str | None
    results: list[DetectedLanguage]
    cost_usd: float = 0.0


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


class JobItemModel(BaseModel):
    #: The caller's own reference; results come back under it.
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=8192)
    source_lang: str = Field(min_length=2, max_length=8)
    targets: list[str] = Field(min_length=1, max_length=24)
    #: "summarize": the item's result is a summary per language (its own
    #: language included) instead of translations. A job with summaries
    #: runs here, never as a provider batch.
    task: Literal["translate", "summarize"] = "translate"
    max_chars: int | None = Field(default=None, ge=80, le=1000)
    about: str | None = Field(default=None, max_length=120)


class JobSubmitRequest(BaseModel):
    items: list[JobItemModel] = Field(min_length=1, max_length=5000)
    backend: TranslationBackend = TranslationBackend.NEBIUS
    #: None takes the service's configured mode (JOB_MODE).
    mode: Literal["auto", "provider", "realtime"] | None = None


class JobItemResultModel(BaseModel):
    id: str
    translations: dict[str, str] = {}
    #: Why this item has no translations, when it has none.
    error: str | None = None
    #: True when sending the same item again later can succeed (budget,
    #: breaker, provider hiccup); False when this text itself failed.
    retryable: bool = False
    cost_usd: float = 0.0


class JobResponse(BaseModel):
    job_id: str
    #: "provider" (a Nebius batch) or "realtime" (translated by this service).
    mode: str
    #: queued, running, completed, or failed (the whole job: provider lost it).
    status: str
    n_items: int
    cost_usd: float = 0.0
    error: str | None = None
    created_at: dt.datetime | None = None
    completed_at: dt.datetime | None = None
    #: Every item's result, in submission order, once the job is completed.
    results: list[JobItemResultModel] | None = None
