"""Hand-curated catalog of available backends with quality scores.

Quality score rationale (0.0–1.0, calibrated against MT benchmarks on formal
institutional text, spot-checked on EU-body names):

  mistral          1.00   Frontier-tier chat LLM, JSON-mode disciplined.
  nllb-local       0.78   NLLB-200-distilled-600M — solid on formal text,
                          noticeably weaker on idiomatic / marketing prose.
  mistral-embed    1.00   1024-dim Mistral embeddings, multilingual.
  labse-local      0.82   LaBSE — 109-language sentence embeddings; strong
                          on cross-lingual retrieval of institutional names,
                          behind Mistral on nuance.
  minilm-local     0.72   Paraphrase-multilingual-MiniLM-L12-v2 — 384-dim,
                          50+ languages, ~2-3x faster than LaBSE on CPU.
                          Retrieval-shaped, sits behind LaBSE on nuance.

These are guidance numbers for callers picking a tier, not scientific claims.
Adjust upward/downward as real usage data accrues.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Kind = Literal["translation", "embedding"]


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    kind: Kind
    quality_score: float
    dim: int | None        # embedding dim, or None for translation
    languages_supported: int
    cost_tier: Literal["paid", "free"]
    description: str


CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo(
        backend="mistral",
        kind="translation",
        quality_score=1.00,
        dim=None,
        languages_supported=24,
        cost_tier="paid",
        description="Mistral medium — frontier quality for visible strings.",
    ),
    ModelInfo(
        backend="nllb-local",
        kind="translation",
        quality_score=0.78,
        dim=None,
        languages_supported=200,
        cost_tier="free",
        description="NLLB-200-distilled-600M — bulk / low-visibility text.",
    ),
    ModelInfo(
        backend="mistral-embed",
        kind="embedding",
        quality_score=1.00,
        dim=1024,
        languages_supported=24,
        cost_tier="paid",
        description="Mistral 1024-dim multilingual sentence embeddings.",
    ),
    ModelInfo(
        backend="labse-local",
        kind="embedding",
        quality_score=0.82,
        dim=768,
        languages_supported=109,
        cost_tier="free",
        description="LaBSE — 768-dim cross-lingual sentence embeddings.",
    ),
    ModelInfo(
        backend="minilm-local",
        kind="embedding",
        quality_score=0.72,
        dim=384,
        languages_supported=50,
        cost_tier="free",
        description=(
            "paraphrase-multilingual-MiniLM-L12-v2 — 384-dim, "
            "fast CPU sentence embeddings for high-throughput index sinks."
        ),
    ),
)
