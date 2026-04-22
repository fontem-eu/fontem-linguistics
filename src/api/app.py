"""FastAPI app factory + lifespan."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from src.api.deps import Services
from src.api.routes import router
from src.backends.labse_local import LabseLocalBackend
from src.backends.mistral import MistralBackend
from src.backends.nllb_local import NllbLocalBackend
from src.cache.postgres import PostgresCache
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.config import Settings, get_settings
from src.infra.spend_cap import SpendCap
from src.services.embedding import EmbeddingService
from src.services.translation import TranslationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = app.state.settings

    cache = await PostgresCache.connect(
        dsn=_asyncpg_dsn(settings.database_url),
        lru_size=settings.inprocess_lru_size,
    )
    await _ensure_schema(cache)
    app.state.cache = cache

    mistral: MistralBackend | None = None
    if settings.mistral_api_key:
        mistral = MistralBackend.build(
            api_url=settings.mistral_api_url,
            api_key=settings.mistral_api_key,
            chat_model=settings.mistral_chat_model,
            embed_model=settings.mistral_embed_model,
            timeout_s=settings.mistral_timeout_s,
            max_retries=settings.mistral_max_retries,
            price_input_per_mtok=settings.mistral_price_input_per_mtok,
            price_output_per_mtok=settings.mistral_price_output_per_mtok,
            price_embed_per_mtok=settings.mistral_price_embed_per_mtok,
        )

    nllb = NllbLocalBackend(
        model_name=settings.nllb_model, local_path=settings.local_models_path,
    )
    labse = LabseLocalBackend(
        model_name=settings.labse_model, local_path=settings.local_models_path,
    )

    breaker = CircuitBreaker(
        failure_threshold=settings.breaker_failure_threshold,
        window_s=settings.breaker_window_s,
        cooldown_s=settings.breaker_cooldown_s,
        min_requests=settings.breaker_min_requests,
    )
    spend_cap = SpendCap(daily_cap_usd=settings.spend_cap_usd_daily)

    translation = TranslationService(
        cache=cache, mistral=mistral, nllb=nllb,
        mistral_breaker=breaker, mistral_spend_cap=spend_cap,
    )
    embedding = EmbeddingService(
        cache=cache, mistral=mistral, labse=labse,
        mistral_breaker=breaker, mistral_spend_cap=spend_cap,
    )
    app.state.services = Services(translation=translation, embedding=embedding)
    logger.info("gmr-linguistics ready (mistral={}, nllb={}, labse={})",
                mistral is not None, True, True)

    try:
        yield
    finally:
        if mistral is not None:
            await mistral.aclose()
        await cache.close()


def _asyncpg_dsn(url: str) -> str:
    """asyncpg uses postgres:// rather than postgresql+asyncpg://."""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://") :]
    return url


async def _ensure_schema(cache: PostgresCache) -> None:
    """Create tables on first startup. Tablespace assumed pre-provisioned at DB level."""
    async with cache.pool.acquire() as con:
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
        await con.execute(
            "CREATE INDEX IF NOT EXISTS translations_backend_idx ON translations (backend)"
        )
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
        await con.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_backend_idx ON embeddings (backend)"
        )


def build_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    application = FastAPI(title="gmr-linguistics", version="0.1.0", lifespan=lifespan)
    application.state.settings = s
    application.include_router(router)
    return application


app = build_app()
