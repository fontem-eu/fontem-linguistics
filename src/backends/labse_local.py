"""LaBSE (Language-agnostic BERT Sentence Embeddings) — 768-dim, CPU-OK."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.domain.models import BackendUnavailable


@dataclass
class LabseLocalBackend:
    model_name: str
    local_path: str
    # Dynamic int8 quantisation on the transformer's nn.Linear layers.
    # LaBSE is ~471M params — fp32 ≈ 1.9 GB resident, int8 ≈ 0.5 GB on
    # the weights, minimal accuracy drop for sentence-similarity use.
    quantize: bool = True
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
                import torch  # pylint: disable=import-outside-toplevel
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise BackendUnavailable(
                    f"labse-local: sentence-transformers/torch not installed ({exc})"
                ) from exc
            loop = asyncio.get_running_loop()

            def _load_and_quantise():
                model = SentenceTransformer(
                    self.model_name, cache_folder=self.local_path,
                )
                model.eval()
                if self.quantize:
                    # SentenceTransformer wraps a plain HF transformer —
                    # dynamic int8 on nn.Linear is safe; the pooling +
                    # normalise ops stay fp32.
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                return model

            model = await loop.run_in_executor(None, _load_and_quantise)
            self._model = model
            self._loaded = True

    async def embed(self, text: str) -> list[float]:
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(
            None, lambda: self._model.encode(text, normalize_embeddings=True).tolist()
        )
        return list(vec)
