"""PostgresCache integration — real Postgres via testcontainers.

Marked integration because it starts Docker. Skips cleanly if Docker is unavailable.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def pg_dsn():
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not installed")
    try:
        ctr = PostgresContainer("postgres:16-alpine")
        ctr.start()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # testcontainers raises a grab-bag of docker.errors/urllib/socket
        # subclasses depending on the failure mode (no docker socket,
        # image pull failure, port collision); we want to skip on all.
        pytest.skip(f"cannot start postgres container: {exc}")
    # testcontainers returns a psycopg URL; asyncpg wants postgresql://
    dsn = ctr.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    yield dsn
    ctr.stop()


async def _setup_schema(pool):
    async with pool.acquire() as con:
        await con.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                source_hash   BYTEA       NOT NULL,
                source_text   TEXT        NOT NULL,
                source_lang   TEXT        NOT NULL,
                target_lang   TEXT        NOT NULL,
                translation   TEXT        NOT NULL,
                backend       TEXT        NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (source_hash, target_lang, backend)
            )
        """)
        await con.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                source_hash   BYTEA       NOT NULL,
                source_text   TEXT        NOT NULL,
                dim           INT         NOT NULL,
                vector        FLOAT4[]    NOT NULL,
                backend       TEXT        NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (source_hash, backend)
            )
        """)


async def test_translation_roundtrip(pg_dsn):
    from src.cache.postgres import PostgresCache
    cache = await PostgresCache.connect(dsn=pg_dsn)
    try:
        await _setup_schema(cache.pool)
        # empty cache
        assert await cache.get_translations("hi", "en", ["fr", "de"], "mistral") == {}

        await cache.put_translations(
            "hi", "en", "mistral", {"fr": "salut", "de": "hallo"}
        )

        # full hit via LRU
        got = await cache.get_translations("hi", "en", ["fr", "de"], "mistral")
        assert got == {"fr": "salut", "de": "hallo"}

        # partial hit (third lang missing)
        got = await cache.get_translations("hi", "en", ["fr", "es"], "mistral")
        assert got == {"fr": "salut"}

        # backend isolation: same text, different backend, separate storage
        got = await cache.get_translations("hi", "en", ["fr"], "nllb-local")
        assert got == {}
    finally:
        await cache.close()


async def test_translation_upsert_overwrites(pg_dsn):
    from src.cache.postgres import PostgresCache
    cache = await PostgresCache.connect(dsn=pg_dsn)
    try:
        await _setup_schema(cache.pool)
        await cache.put_translations("hi2", "en", "mistral", {"fr": "v1"})
        await cache.put_translations("hi2", "en", "mistral", {"fr": "v2"})
        got = await cache.get_translations("hi2", "en", ["fr"], "mistral")
        assert got == {"fr": "v2"}
    finally:
        await cache.close()


async def test_embedding_roundtrip(pg_dsn):
    from src.cache.postgres import PostgresCache
    cache = await PostgresCache.connect(dsn=pg_dsn)
    try:
        await _setup_schema(cache.pool)
        assert await cache.get_embedding("hello", "mistral-embed") is None

        await cache.put_embedding("hello", "mistral-embed", [0.1, 0.2, 0.3])
        vec = await cache.get_embedding("hello", "mistral-embed")
        assert vec == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)

        # backend isolation
        assert await cache.get_embedding("hello", "labse-local") is None
    finally:
        await cache.close()


async def test_source_hash_deterministic():
    from src.cache.postgres import source_hash
    assert source_hash("hi", "en") == source_hash("hi", "en")
    assert source_hash("hi", "en") != source_hash("hi", "fr")
    assert source_hash("hi") != source_hash("hi", "en")
