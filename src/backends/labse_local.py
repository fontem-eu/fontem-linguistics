"""LaBSE (Language-agnostic BERT Sentence Embeddings) — 768-dim, CPU-OK.

Loads weights from a local path populated by the `mirror-pull`
InitContainer, not from HuggingFace Hub. The path is version-keyed
(e.g. `/models/labse-1.0.0`) so rolling the mirror is a path change,
not an in-place overwrite.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from src.domain.models import BackendUnavailable


@dataclass
class LabseLocalBackend:
    # Directory containing the HuggingFace snapshot bytes for the mirrored
    # LaBSE revision. Must pre-exist on disk (placed there by the pod's
    # InitContainer); we never reach out to HF Hub from this backend.
    model_path: str
    # Signed-mirror identity, e.g. "labse@1.0.0-836121a". Stamped on every
    # /embed response so downstream stores can reject cross-version
    # comparisons and spot drift.
    encoder_id: str
    # Dynamic int8 quantisation on the transformer's nn.Linear layers —
    # kept as a toggle but OFF by default pending re-evaluation with
    # torchao's non-deprecated API (same reasoning as NllbLocalBackend).
    quantize: bool = False
    _model: object | None = None
    _loaded: bool = False
    _load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            if not os.path.isdir(self.model_path):
                raise BackendUnavailable(
                    f"labse-local: model_path {self.model_path!r} does not "
                    "exist. The mirror-pull InitContainer should populate "
                    "it before the api container starts; if this fires the "
                    "pod's InitContainer failed or is disabled.",
                )
            try:
                # pylint: disable-next=import-outside-toplevel
                import torch
                # pylint: disable-next=import-outside-toplevel
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise BackendUnavailable(
                    f"labse-local: sentence-transformers/torch not installed ({exc})"
                ) from exc
            loop = asyncio.get_running_loop()

            def _load_and_quantise():
                import gc  # pylint: disable=import-outside-toplevel
                # Local path, not HF repo id — sentence-transformers
                # auto-detects this from the presence of config files.
                model = SentenceTransformer(self.model_path)
                model.eval()
                if self.quantize:
                    quantised = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                    del model
                    gc.collect()
                    return quantised
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
