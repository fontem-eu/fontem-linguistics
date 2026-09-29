> ### 🪞 This GitHub repository is a mirror
>
> Development happens on Fontem's own infrastructure; this mirror is
> updated automatically. **Issues and pull requests opened here are not
> monitored.**
>
> If you would like to contribute — code, data sources, review, or
> anything else — please get in touch at **team@fontem.eu** and we will
> set you up.

# fontem-linguistics

Translation + embedding service. Nebius-hosted Gemma (bulk translation and language detection), NLLB-200 distilled (local translator) + LaBSE (sentence embeddings) running on cluster. Used by the consolidator for cross-language entity matching and by the API for translated authority names.

## Translation jobs

For callers that need many translations but not this minute (the translator
service). `POST /translate/jobs` takes up to 5,000 items
(`{id, text, source_lang, targets}`) and answers 202 with a `job_id`;
`GET /translate/jobs/{job_id}` reports `queued` / `running` / `completed` /
`failed`, with every item's result once completed. An item without
translations carries `error` and `retryable` (send it again later) or not
(this text itself failed).

A job runs one of two ways (`JOB_MODE`, or `mode` per request):

- `provider` — a Nebius batch: half the real-time price, outside the
  real-time rate limits, results within hours. Texts the translation cache
  already holds are answered from it and never sent.
- `realtime` — here, through the ordinary translate path, every job in the
  pod sharing `JOB_MAX_CONCURRENCY` (32) provider calls.
- `auto` (default) — provider, and realtime while the provider refuses
  batches (`JOB_PROVIDER_RETRY_S`, 900 s, before offering it another).

Jobs live in the `translation_jobs` table, so a restart loses none: a
realtime job's claim lapses after `JOB_LEASE_S` and is taken over, and what
was already translated comes back from the cache. Batch spend is reserved
against `NEBIUS_SPEND_CAP_USD_DAILY` when the batch is submitted.

## Deploy

One instance (`linguistics-service` namespace) serves every environment,
production included. A merge to main bumps its pin in
`gitops/linguistics-service/fontem-linguistics.yaml`: **merging is a
production deploy**, with no testing stage in front of it.

## Convention

See [/config/repos/CLAUDE.md](https://contribute.void42.internal/fontem/gitops) for workspace-wide rules (feature branches + CI gate, no direct push to main, full gate before declaring done, conventional commits).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
