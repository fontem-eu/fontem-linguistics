"""FastAPI app factory + lifespan."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from src.api.deps import Services
from src.api.routes import router
from src.backends.labse_local import LabseLocalBackend
from src.backends.minilm_local import MinilmLocalBackend
from src.backends.mistral import MistralBackend
from src.backends.nebius import NebiusBackend
from src.backends.nllb_local import NllbLocalBackend
from src.cache.postgres import PostgresCache
from src.infra.circuit_breaker import CircuitBreaker
from src.infra.config import Settings, get_settings
from src.infra.spend_cap import SpendCap
from src.services.embedding import EmbeddingService
from src.services.translation import TranslationService


def _build_hosted_backends(
    settings: Settings,
) -> tuple[MistralBackend | None, NebiusBackend | None]:
    """The two paid providers, each built only if it has a key.

    Lifted out of `lifespan` because it is configuration assembly, not
    startup sequencing, and because a provider without a key must come back
    as None rather than as a backend that fails on first use.
    """
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

    nebius: NebiusBackend | None = None
    if settings.nebius_api_key:
        nebius = NebiusBackend.build(
            api_url=settings.nebius_api_url,
            api_key=settings.nebius_api_key,
            chat_model=settings.nebius_chat_model,
            timeout_s=settings.nebius_timeout_s,
            max_retries=settings.nebius_max_retries,
            price_input_per_mtok=settings.nebius_price_input_per_mtok,
            price_output_per_mtok=settings.nebius_price_output_per_mtok,
        )
    return mistral, nebius


# Startup sequence: six backends, two breakers, two budgets and two
# services, in dependency order. The count is the point — splitting it
# further only moves the locals into a helper with seven parameters.
@asynccontextmanager
async def lifespan(  # pylint: disable=too-many-locals
    application: FastAPI,
) -> AsyncGenerator[None, None]:
    settings: Settings = application.state.settings

    cache = await PostgresCache.connect(
        dsn=_asyncpg_dsn(settings.database_url),
        lru_size=settings.inprocess_lru_size,
    )
    await _ensure_schema(cache)
    application.state.cache = cache

    mistral, nebius = _build_hosted_backends(settings)

    nllb = NllbLocalBackend(
        model_name=settings.nllb_model,
        local_path=settings.local_models_path,
        quantize=settings.local_quantize_int8,
    )
    labse = LabseLocalBackend(
        model_path=settings.labse_model_path,
        encoder_id=settings.labse_encoder_id,
        quantize=settings.local_quantize_int8,
    )
    minilm = MinilmLocalBackend(
        model_path=settings.minilm_model_path,
        encoder_id=settings.minilm_encoder_id,
        quantize=settings.local_quantize_int8,
    )

    def _breaker() -> CircuitBreaker:
        return CircuitBreaker(
            failure_threshold=settings.breaker_failure_threshold,
            window_s=settings.breaker_window_s,
            cooldown_s=settings.breaker_cooldown_s,
            min_requests=settings.breaker_min_requests,
        )

    breaker = _breaker()
    spend_cap = SpendCap(daily_cap_usd=settings.spend_cap_usd_daily)
    # Its own breaker and its own budget: Mistral degrading must not close
    # the door on Nebius, and a bulk translation run must not spend the
    # budget the assistant's traffic depends on.
    nebius_breaker = _breaker()
    nebius_spend_cap = SpendCap(daily_cap_usd=settings.nebius_spend_cap_usd_daily)

    translation = TranslationService(
        cache=cache, mistral=mistral, nllb=nllb,
        mistral_breaker=breaker, mistral_spend_cap=spend_cap,
        nebius=nebius, nebius_breaker=nebius_breaker,
        nebius_spend_cap=nebius_spend_cap,
    )
    embedding = EmbeddingService(
        cache=cache, mistral=mistral, labse=labse, minilm=minilm,
        mistral_breaker=breaker, mistral_spend_cap=spend_cap,
    )
    application.state.services = Services(translation=translation, embedding=embedding)

    # Warm the local encoder-only backends BEFORE opening the port.
    # First-request cold-start on SentenceTransformer.encode is ~3s for
    # MiniLM and ~2s for LaBSE; without this preload the first /embed
    # after every pod restart hangs downstream callers (fontem-api
    # search falls back to lexical_only after its 3s timeout). NLLB
    # is translation-only and heavier; leave it lazy.
    for name, backend in (("labse", labse), ("minilm", minilm)):
        try:
            await backend.embed("warmup")
            logger.info("{} preload OK", name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("{} preload skipped: {}", name, exc)

    logger.info(
        "fontem-linguistics ready (mistral={}, nllb={}, labse={}, minilm={})",
        mistral is not None, True, True, True,
    )

    try:
        yield
    finally:
        for provider in (mistral, nebius):
            if provider is not None:
                await provider.aclose()
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
    application = FastAPI(title="fontem-linguistics", version="0.1.0", lifespan=lifespan)
    application.state.settings = s
    application.include_router(router)
    return application


app = build_app()
