"""Prometheus metrics — a single module so cardinality is reviewable in one place."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

TRANSLATIONS_TOTAL = Counter(
    "translations_total",
    "Total translation requests.",
    ["backend", "cached", "source_lang"],
)

EMBEDDINGS_TOTAL = Counter(
    "embeddings_total",
    "Total embedding requests.",
    ["backend", "cached"],
)

TRANSLATION_LATENCY = Histogram(
    "translation_latency_seconds",
    "Translation latency.",
    ["backend"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

EMBEDDING_LATENCY = Histogram(
    "embedding_latency_seconds",
    "Embedding latency.",
    ["backend"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MISTRAL_SPEND_USD = Counter(
    "mistral_spend_usd_total",
    "Cumulative Mistral spend (USD). Reset per pod lifetime; use SpendCap for daily budget.",
    ["endpoint"],
)

# Provider-labelled successor to MISTRAL_SPEND_USD. A second paid provider
# (Nebius) made a metric named for one of them wrong, and renaming the old
# one would break the dashboards that already query it — so Mistral keeps
# reporting to both and new providers report only here.
LLM_SPEND_USD = Counter(
    "llm_spend_usd_total",
    "Cumulative hosted-LLM spend (USD) by provider. Per pod lifetime; "
    "SpendCap holds the daily budget.",
    ["provider", "endpoint"],
)

BREAKER_STATE = Gauge(
    "mistral_circuit_breaker_state",
    "Circuit breaker state for the Mistral backend (0=closed, 1=half-open, 2=open).",
)

CACHE_HITS = Counter("cache_hits_total", "Cache hits.", ["resource"])
CACHE_MISSES = Counter("cache_misses_total", "Cache misses.", ["resource"])
