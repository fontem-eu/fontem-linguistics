"""Shared implementation for the local SentenceTransformers backends.

``labse_local`` and ``minilm_local`` were byte-for-byte the same logic
apart from their docstrings and the model name in error messages — the
minilm module said so out loud: *"Same async-load + optional dynamic-int8
quantise pattern as LabseLocalBackend; if you tune one, tune both."* That
instruction is the problem: SonarQube measured them at 45% and 39%
duplication, and "tune both" is the kind of thing that gets half-done.

The two concrete backends now differ only in what they document and the
label they report in errors, which is all that was ever really different.

Weights load from a local path populated by the ``mirror-pull``
InitContainer, never from HuggingFace Hub. The path is version-keyed
(e.g. ``/models/labse-1.0.0``) so rolling the mirror is a path change
rather than an in-place overwrite.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import ClassVar

from src.domain.models import BackendUnavailable


@dataclass
class LocalSentenceTransformerBackend:
    """Async-loading SentenceTransformers backend over a mirrored snapshot.

    Subclasses set ``backend_label`` — it prefixes every error so an
    operator can tell which of the local backends failed without a
    traceback.
    """

    # Directory holding the HuggingFace snapshot bytes for the mirrored
    # revision. Must pre-exist on disk (placed by the pod's InitContainer);
    # we never reach out to HF Hub from here.
    model_path: str
    # Signed-mirror identity, e.g. "labse@1.0.0-836121a". Stamped on every
    # /embed response so downstream stores can reject cross-version
    # comparisons and spot drift.
    encoder_id: str
    # Dynamic int8 quantisation on the transformer's nn.Linear layers —
    # kept as a toggle but OFF by default pending re-evaluation with
    # torchao's non-deprecated API (torch.quantization eager mode grows
    # RSS on some models). Same reasoning as NllbLocalBackend.
    quantize: bool = False
    _model: object | None = None
    _loaded: bool = False
    _load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    backend_label: ClassVar[str] = "local-sentence-transformer"

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            if not os.path.isdir(self.model_path):
                raise BackendUnavailable(
                    f"{self.backend_label}: model_path {self.model_path!r} "
                    "does not exist. The mirror-pull InitContainer should "
                    "populate it before the api container starts; if this "
                    "fires the pod's InitContainer failed or is disabled.",
                )
            try:
                # pylint: disable-next=import-outside-toplevel
                import torch
                # pylint: disable-next=import-outside-toplevel
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise BackendUnavailable(
                    f"{self.backend_label}: sentence-transformers/torch "
                    f"not installed ({exc})"
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
            None,
            lambda: self._model.encode(text, normalize_embeddings=True).tolist(),
        )
        return list(vec)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batched inference — one model.encode() over the whole list.

        For CPU inference with SentenceTransformers, a single batched
        encode() runs BLAS-parallel and clears the Python overhead of
        per-text HTTP + coroutine hops. Empirically ~10x throughput at
        batch=32 for MiniLM-L12 on a 6-core node.
        """
        await self._ensure_loaded()
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        arr = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True),
        )
        return [list(v) for v in arr.tolist()]
