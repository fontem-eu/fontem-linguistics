# gmr-linguistics

Durable-cached translation + embedding service for GMR. Thin dispatcher in
front of Mistral (paid, high quality) and locally-hosted NLLB-200 / LaBSE
(free, CPU-only, bulk).

Caller picks the backend per request. Exact-match cache (PostgreSQL) in front
of every backend call — re-ingests never pay twice. Circuit breaker +
daily spend cap protect the Mistral wallet.

## Endpoints

- `POST /translate` — text → {lang: translation}
- `POST /translate/batch` — bulk translations, `Idempotency-Key` header dedupes
- `POST /embed` — text → 1024-dim (Mistral) or 768-dim (LaBSE) vector
- `GET  /healthz` — liveness
- `GET  /readyz` — readiness (verifies DB)
- `GET  /metrics` — Prometheus

## Backends

| Tier | Backend | Use for |
|------|---------|---------|
| High quality ($) | `mistral` / `mistral-embed` | Authority names, visible UI strings |
| Bulk (free, CPU) | `nllb-local` / `labse-local` | Contract titles, descriptions |

## Local dev

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt    # for NLLB / LaBSE backends
export DATABASE_URL=postgresql+asyncpg://linguistics:pw@localhost:5432/linguistics
export MISTRAL_API_KEY=...             # optional; leave unset to disable mistral tier
make test
```

## Tests

- `make test` — unit + component with ≥ 90 % coverage gate.
- `make test-integration` — Postgres via testcontainers (needs Docker).
- `make test-all` — both.

**Mistral API key is never used in tests.** A FastAPI stub at
`tests/fixtures/mistral_stub.py` is wired into the httpx client in component
tests; conftest.py fails fast if a production key is set.
