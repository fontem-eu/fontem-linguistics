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


class ModelsResponse(BaseModel):
    models: list[ModelInfoResponse]


class LanguageInfo(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]
