"""Postgres cache for translations + embeddings. asyncpg for simplicity."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import asyncpg
from cachetools import LRUCache

from src.infra.metrics import CACHE_HITS, CACHE_MISSES


def source_hash(text: str, source_lang: str | None = None) -> bytes:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    if source_lang is not None:
        h.update(b"\x00")
        h.update(source_lang.encode("utf-8"))
    return h.digest()


@dataclass
class PostgresCache:
    pool: asyncpg.Pool
    translation_lru: LRUCache
    embedding_lru: LRUCache

    @classmethod
    async def connect(cls, dsn: str, lru_size: int = 1024, min_size: int = 2, max_size: int = 10) -> "PostgresCache":
        pool = await asyncpg.create_pool(dsn=dsn, min_size=min_size, max_size=max_size)
        return cls(
            pool=pool,
            translation_lru=LRUCache(maxsize=lru_size),
            embedding_lru=LRUCache(maxsize=lru_size),
        )

    async def close(self) -> None:
        await self.pool.close()

    # ── Translations ──────────────────────────────────────────────

    async def get_translations(
        self, text: str, source_lang: str, targets: list[str], backend: str
    ) -> dict[str, str]:
        """Return {target_lang: translation} for all hits; missing targets are absent."""
        key_src = source_hash(text, source_lang)
        out: dict[str, str] = {}
        missing: list[str] = []
        for t in targets:
            lru_key = (key_src, t, backend)
            if lru_key in self.translation_lru:
                out[t] = self.translation_lru[lru_key]
                CACHE_HITS.labels(resource="translation").inc()
            else:
                missing.append(t)

        if missing:
            async with self.pool.acquire() as con:
                rows = await con.fetch(
                    """
                    SELECT target_lang, translation
                    FROM translations
                    WHERE source_hash = $1 AND backend = $2 AND target_lang = ANY($3::text[])
                    """,
                    key_src, backend, missing,
                )
            for row in rows:
                out[row["target_lang"]] = row["translation"]
                self.translation_lru[(key_src, row["target_lang"], backend)] = row["translation"]
                CACHE_HITS.labels(resource="translation").inc()
            for t in missing:
                if t not in out:
                    CACHE_MISSES.labels(resource="translation").inc()
        return out

    async def put_translations(
        self, text: str, source_lang: str, backend: str, translations: dict[str, str]
    ) -> None:
        if not translations:
            return
        key_src = source_hash(text, source_lang)
        rows = [(key_src, text, source_lang, t, v, backend) for t, v in translations.items()]
        async with self.pool.acquire() as con:
            await con.executemany(
                """
                INSERT INTO translations
                    (source_hash, source_text, source_lang, target_lang, translation, backend)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_hash, target_lang, backend) DO UPDATE
                SET translation = EXCLUDED.translation, updated_at = NOW()
                """,
                rows,
            )
        for t, v in translations.items():
            self.translation_lru[(key_src, t, backend)] = v

    # ── Embeddings ────────────────────────────────────────────────

    async def get_embedding(self, text: str, backend: str) -> list[float] | None:
        key_src = source_hash(text)
        lru_key = (key_src, backend)
        if lru_key in self.embedding_lru:
            CACHE_HITS.labels(resource="embedding").inc()
            return list(self.embedding_lru[lru_key])

        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                "SELECT vector FROM embeddings WHERE source_hash = $1 AND backend = $2",
                key_src, backend,
            )
        if row is None:
            CACHE_MISSES.labels(resource="embedding").inc()
            return None
        vec = list(row["vector"])
        self.embedding_lru[lru_key] = tuple(vec)
        CACHE_HITS.labels(resource="embedding").inc()
        return vec

    async def put_embedding(self, text: str, backend: str, vector: list[float]) -> None:
        key_src = source_hash(text)
        async with self.pool.acquire() as con:
            await con.execute(
                """
                INSERT INTO embeddings (source_hash, source_text, dim, vector, backend)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (source_hash, backend) DO UPDATE
                SET vector = EXCLUDED.vector, dim = EXCLUDED.dim
                """,
                key_src, text, len(vector), vector, backend,
            )
        self.embedding_lru[(key_src, backend)] = tuple(vector)
