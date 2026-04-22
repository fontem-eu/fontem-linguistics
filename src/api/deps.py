"""Dependency holders — populated by the app lifespan, read by route handlers."""
from __future__ import annotations

from dataclasses import dataclass

from src.services.embedding import EmbeddingService
from src.services.translation import TranslationService


@dataclass
class Services:
    translation: TranslationService
    embedding: EmbeddingService


# Module-level holder: set once in `lifespan`, read by handlers via request.app.state.
