"""JobStore against real Postgres: the DDL startup runs, claims and their
lapse, the once-only finish, retention. Skips without Docker."""
from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from src.api.app import _ensure_schema
from src.cache.jobs import ItemResult, JobItem, JobRecord, JobStore
from src.cache.postgres import PostgresCache

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def pg_dsn():
    try:
        from testcontainers.postgres import PostgresContainer  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.skip("testcontainers not installed")
    try:
        ctr = PostgresContainer("postgres:16-alpine")
        ctr.start()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        pytest.skip(f"cannot start postgres container: {exc}")
    yield ctr.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    ctr.stop()


@pytest.fixture(name="store")
async def _store(pg_dsn):
    cache = await PostgresCache.connect(dsn=pg_dsn)
    await _ensure_schema(cache)
    await _ensure_schema(cache)          # twice: every pod start runs it
    async with cache.pool.acquire() as con:
        await con.execute("TRUNCATE translation_jobs")
    yield JobStore(cache.pool)
    await cache.close()


def _job(job_id: str, mode: str = "realtime") -> JobRecord:
    return JobRecord(job_id=job_id, mode=mode, backend="nebius", status="queued",
                     items=[JobItem("1", "Łódź — roboty", "pl", ["en", "de"])])


async def test_a_job_round_trips_with_its_items_and_prefilled_results(store):
    job = _job("p1", mode="provider")
    job.provider_batch_id, job.reserved_usd, job.reserved_on = "batch-1", 0.5, dt.date(2026, 9, 29)
    job.prefilled = [ItemResult("1", {"en": "Łódź works"})]
    await store.insert(job)
    got = await store.get("p1")
    assert got.items[0].text == "Łódź — roboty" and got.items[0].targets == ["en", "de"]
    assert got.prefilled[0].translations == {"en": "Łódź works"}
    assert (got.provider_batch_id, got.reserved_usd, got.reserved_on) == (
        "batch-1", 0.5, dt.date(2026, 9, 29))
    assert got.results is None and got.created_at is not None
    assert await store.get("nope") is None


async def test_a_realtime_job_is_claimed_once_until_its_claim_lapses(store):
    await store.insert(_job("r1"))
    await store.insert(_job("p1", mode="provider"))           # never claimed: not realtime
    first = await store.claim_realtime("w1", lease_s=60, limit=8)
    assert [j.job_id for j in first] == ["r1"] and first[0].status == "running"
    assert await store.claim_realtime("w2", lease_s=60, limit=8) == []
    assert await store.heartbeat("r1", "w1") and not await store.heartbeat("r1", "w2")
    await asyncio.sleep(1.1)
    taken = await store.claim_realtime("w2", lease_s=1, limit=8)
    assert [j.job_id for j in taken] == ["r1"]
    assert not await store.heartbeat("r1", "w1")              # w1 learns it lost the job


async def test_a_released_job_is_claimable_at_once(store):
    await store.insert(_job("r1"))
    await store.claim_realtime("w1", lease_s=60, limit=8)
    await store.release("r1", "w2")                           # not w2's to release
    assert await store.claim_realtime("w3", lease_s=60, limit=8) == []
    await store.release("r1", "w1")
    assert [j.job_id for j in await store.claim_realtime("w3", 60, 8)] == ["r1"]


async def test_a_job_finishes_once(store):
    await store.insert(_job("r1"))
    await store.set_running("r1")
    done = await store.finish("r1", status="completed", cost_usd=0.002,
                              results=[ItemResult("1", {"en": "works"}, cost_usd=0.002)])
    assert done.status == "completed" and done.completed_at is not None
    assert done.results[0].translations == {"en": "works"}
    assert await store.finish("r1", status="failed", results=None, cost_usd=0) is None
    assert (await store.get("r1")).status == "completed"
    assert await store.claim_realtime("w1", 60, 8) == []      # finished: never claimed


async def test_retention_forgets_only_old_finished_jobs(store):
    for job_id in ("old-done", "old-open", "new-done"):
        await store.insert(_job(job_id))
    for job_id in ("old-done", "new-done"):
        await store.finish(job_id, status="completed", results=[], cost_usd=0)
    async with store.pool.acquire() as con:
        await con.execute("UPDATE translation_jobs SET created_at = NOW() - interval '30 days' "
                          "WHERE job_id LIKE 'old-%'")
    assert await store.purge(14) == 1
    assert await store.get("old-done") is None
    assert await store.get("old-open") is not None and await store.get("new-done") is not None
