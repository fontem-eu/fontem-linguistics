"""Translation jobs, kept in the linguistics Postgres beside the cache.

A job is the manifest a caller submitted (its items), how it runs (on the
provider's batch API, or here in real time), and in the end its results.
Keeping it in Postgres rather than in the pod means a restart loses no job:
a realtime job's claim lapses and another pod (or the same one, restarted)
takes it over, and the items translated before the restart come back from
the translation cache for free.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

import asyncpg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS translation_jobs (
    job_id            TEXT        PRIMARY KEY,
    mode              TEXT        NOT NULL CHECK (mode IN ('provider', 'realtime')),
    backend           TEXT        NOT NULL,
    status            TEXT        NOT NULL
                      CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    items             JSONB       NOT NULL,
    n_items           INT         NOT NULL,
    provider_batch_id TEXT,
    provider_file_id  TEXT,
    reserved_usd      DOUBLE PRECISION NOT NULL DEFAULT 0,
    reserved_on       DATE,
    prefilled         JSONB,
    results           JSONB,
    cost_usd          DOUBLE PRECISION NOT NULL DEFAULT 0,
    error             TEXT,
    claimed_by        TEXT,
    heartbeat_at      TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS translation_jobs_open_idx
    ON translation_jobs (mode, status) WHERE status IN ('queued', 'running');
"""

#: Job states. "failed" is the whole job: the provider lost or refused it.
QUEUED, RUNNING, COMPLETED, FAILED = "queued", "running", "completed", "failed"
FINAL = frozenset({COMPLETED, FAILED})


#: What a job item asks for: its translations, or a short summary per language.
TRANSLATE, SUMMARIZE = "translate", "summarize"


@dataclass
class JobItem:
    """One text to translate or summarise, as the caller identified it.
    Items stored before summaries existed load as translations."""

    id: str
    text: str
    source_lang: str
    targets: list[str]
    task: str = TRANSLATE
    max_chars: int | None = None
    about: str | None = None


@dataclass
class ItemResult:
    """One item's outcome. ``retryable`` tells the caller whether the same
    request can succeed later (budget, breaker, provider hiccup) or whether
    this text itself came back unusable."""

    id: str
    translations: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    cost_usd: float = 0.0


# A row of translation_jobs, one field per column the service reads.
@dataclass
class JobRecord:  # pylint: disable=too-many-instance-attributes
    job_id: str
    mode: str
    backend: str
    status: str
    items: list[JobItem]
    provider_batch_id: str | None = None
    provider_file_id: str | None = None
    reserved_usd: float = 0.0
    reserved_on: dt.date | None = None
    prefilled: list[ItemResult] = field(default_factory=list)
    results: list[ItemResult] | None = None
    cost_usd: float = 0.0
    error: str | None = None
    created_at: dt.datetime | None = None
    completed_at: dt.datetime | None = None

    @property
    def n_items(self) -> int:
        return len(self.items)


def _results_json(results: list[ItemResult] | None) -> str | None:
    if results is None:
        return None
    return json.dumps([r.__dict__ for r in results], ensure_ascii=False)


def _results_of(raw) -> list[ItemResult] | None:
    if raw is None:
        return None
    return [ItemResult(**r) for r in (json.loads(raw) if isinstance(raw, str) else raw)]


def _record(row) -> JobRecord:
    items = row["items"]
    items = json.loads(items) if isinstance(items, str) else items
    return JobRecord(
        job_id=row["job_id"], mode=row["mode"], backend=row["backend"], status=row["status"],
        items=[JobItem(**i) for i in items],
        provider_batch_id=row["provider_batch_id"], provider_file_id=row["provider_file_id"],
        reserved_usd=row["reserved_usd"], reserved_on=row["reserved_on"],
        prefilled=_results_of(row["prefilled"]) or [], results=_results_of(row["results"]),
        cost_usd=row["cost_usd"], error=row["error"],
        created_at=row["created_at"], completed_at=row["completed_at"],
    )


@dataclass
class JobStore:
    pool: asyncpg.Pool

    async def insert(self, job: JobRecord) -> JobRecord:
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                INSERT INTO translation_jobs
                    (job_id, mode, backend, status, items, n_items, provider_batch_id,
                     provider_file_id, reserved_usd, reserved_on, prefilled)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11::jsonb)
                RETURNING *
                """,
                job.job_id, job.mode, job.backend, job.status,
                json.dumps([i.__dict__ for i in job.items], ensure_ascii=False), job.n_items,
                job.provider_batch_id, job.provider_file_id, job.reserved_usd, job.reserved_on,
                _results_json(job.prefilled),
            )
        return _record(row)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self.pool.acquire() as con:
            row = await con.fetchrow("SELECT * FROM translation_jobs WHERE job_id = $1", job_id)
        return _record(row) if row else None

    async def claim_realtime(self, worker: str, lease_s: float, limit: int) -> list[JobRecord]:
        """Realtime jobs nobody is running: never claimed, or claimed by a
        worker that stopped renewing its claim. Oldest first."""
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                """
                UPDATE translation_jobs SET status = 'running', claimed_by = $1,
                       heartbeat_at = NOW(), updated_at = NOW()
                 WHERE job_id IN (
                       SELECT job_id FROM translation_jobs
                        WHERE mode = 'realtime' AND status IN ('queued', 'running')
                          AND (claimed_by IS NULL
                               OR heartbeat_at < NOW() - make_interval(secs => $2))
                        ORDER BY created_at
                        LIMIT $3
                        FOR UPDATE SKIP LOCKED)
                RETURNING *
                """,
                worker, float(lease_s), limit,
            )
        return sorted((_record(r) for r in rows), key=lambda j: j.created_at or dt.datetime.min)

    async def heartbeat(self, job_id: str, worker: str) -> bool:
        """Renew a claim. False when another worker has taken the job over."""
        async with self.pool.acquire() as con:
            done = await con.execute(
                "UPDATE translation_jobs SET heartbeat_at = NOW() "
                "WHERE job_id = $1 AND claimed_by = $2 AND status = 'running'",
                job_id, worker,
            )
        return done.endswith(" 1")

    async def release(self, job_id: str, worker: str) -> None:
        """Give a claim back, for another pass to take it up."""
        async with self.pool.acquire() as con:
            await con.execute(
                "UPDATE translation_jobs SET claimed_by = NULL, heartbeat_at = NULL, "
                "updated_at = NOW() WHERE job_id = $1 AND claimed_by = $2 AND status = 'running'",
                job_id, worker,
            )

    async def set_running(self, job_id: str) -> None:
        async with self.pool.acquire() as con:
            await con.execute(
                "UPDATE translation_jobs SET status = 'running', updated_at = NOW() "
                "WHERE job_id = $1 AND status = 'queued'", job_id)

    async def finish(self, job_id: str, *, status: str, results: list[ItemResult] | None,
                     cost_usd: float, error: str | None = None) -> JobRecord | None:
        """Record the outcome once: a job already final keeps its first one."""
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                UPDATE translation_jobs
                   SET status = $2, results = $3::jsonb, cost_usd = $4, error = $5,
                       completed_at = NOW(), updated_at = NOW()
                 WHERE job_id = $1 AND status NOT IN ('completed', 'failed')
                RETURNING *
                """,
                job_id, status, _results_json(results), cost_usd, error,
            )
        return _record(row) if row else None

    async def purge(self, older_than_days: int) -> int:
        """Forget finished jobs past their retention. Open ones are kept."""
        async with self.pool.acquire() as con:
            done = await con.execute(
                "DELETE FROM translation_jobs WHERE status IN ('completed', 'failed') "
                "AND created_at < NOW() - make_interval(days => $1)", older_than_days)
        return int(done.split()[-1])
