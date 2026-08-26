"""LaBSE (Language-agnostic BERT Sentence Embeddings) — 768-dim, CPU-OK.

Covers 109 languages, which is why it stays the default for cross-lingual
comparison even though minilm-local is cheaper per call.

The loading, quantisation and inference behaviour lives in
LocalSentenceTransformerBackend; this module carries only what is actually
specific to LaBSE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.backends.local_sentence_transformer import LocalSentenceTransformerBackend


@dataclass
class LabseLocalBackend(LocalSentenceTransformerBackend):
    """768-dim LaBSE embeddings from a locally mirrored snapshot."""

    backend_label: ClassVar[str] = "labse-local"
