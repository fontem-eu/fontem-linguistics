"""paraphrase-multilingual-MiniLM-L12-v2 — 384-dim, CPU-fast, multilingual.

Why this backend, alongside labse-local:
  * 384-dim vs LaBSE's 768 — half the memory + storage for downstream
    pgvector / Neo4j indexes.
  * ~2-3x faster on the same CPU (fewer transformer layers, smaller
    attention matrices).
  * 50+ languages covered vs LaBSE's 109 — narrower but still spans the
    24 EU locales the platform surfaces.
  * Better fit for high-cardinality write paths (search-index sinks)
    where the trade-off leans toward throughput over cross-lingual
    coverage of long-tail languages.

The loading, quantisation and inference behaviour lives in
LocalSentenceTransformerBackend — shared with labse-local, so tuning one
now genuinely tunes both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.backends.local_sentence_transformer import LocalSentenceTransformerBackend


@dataclass
class MinilmLocalBackend(LocalSentenceTransformerBackend):
    """384-dim MiniLM-L12 embeddings from a locally mirrored snapshot."""

    backend_label: ClassVar[str] = "minilm-local"
