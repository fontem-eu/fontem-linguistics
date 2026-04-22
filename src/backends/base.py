"""Protocols for translation + embedding backends. Narrow, explicit."""
from __future__ import annotations

from typing import Protocol


class TranslationBackendProtocol(Protocol):
    async def translate(
        self, text: str, source_lang: str, targets: list[str]
    ) -> dict[str, str]:
        """Return {target_lang: translation}. Raises on hard failure."""
        ...


class EmbeddingBackendProtocol(Protocol):
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector. Raises on hard failure."""
        ...
