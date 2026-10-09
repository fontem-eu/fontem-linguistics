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

# Translation jobs. "event" is submitted / completed / failed / refused (the
# provider would not take a batch); "outcome" per item is ok / retryable /
# failed. Items per minute here is the pipeline's throughput.
TRANSLATION_JOBS = Counter(
    "translation_jobs_total", "Translation jobs by mode and event.", ["mode", "event"])
TRANSLATION_JOB_ITEMS = Counter(
    "translation_job_items_total", "Translation job items by mode and outcome.",
    ["mode", "outcome"])

# Translations asked again with one tagged line per target because their JSON
# answer would not parse (quotes inside a name, as a rule). "path" is realtime
# or batch. Against translation_job_items_total this is the share of texts JSON
# cannot carry; expect about 1 in 2,000.
TRANSLATION_TAGGED_RETRIES = Counter(
    "translation_tagged_retries_total",
    "Translations re-asked as tagged lines after an unreadable JSON answer.", ["path"])

# Texts with a translation that came back under half its length in a readable
# JSON answer, asked again as tagged lines for those languages (see
# openai_chat.cut_short). "path" is realtime or batch. 46 of 518 lobbying goals
# with line breaks did on 2026-10-09; titles, which are one line, rarely should.
TRANSLATION_SHORT_RETRIES = Counter(
    "translation_short_retries_total",
    "Texts whose JSON answer cut a translation short, re-asked as tagged lines.", ["path"])
