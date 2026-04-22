"""LaBSE (Language-agnostic BERT Sentence Embeddings) — 768-dim, CPU-OK."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.domain.models import BackendUnavailable


@dataclass
class LabseLocalBackend:
    model_name: str
    local_path: str
    _model: object | None = None
    _loaded: bool = False
    _load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise BackendUnavailable(
                    f"labse-local: sentence-transformers not installed ({exc})"
                ) from exc
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(
                    self.model_name, cache_folder=self.local_path
                ),
            )
            self._model = model
            self._loaded = True

    async def embed(self, text: str) -> list[float]:
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(
            None, lambda: self._model.encode(text, normalize_embeddings=True).tolist()
        )
        return list(vec)
