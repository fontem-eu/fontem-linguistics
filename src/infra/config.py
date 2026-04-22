"""Runtime configuration — loaded from env, frozen once."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database — no default: DATABASE_URL must be provided explicitly in all
    # environments so no credential can leak via a dev fallback.
    database_url: str = Field(
        default="",
        description="asyncpg DSN for the linguistics cache DB (required).",
    )
    tablespace: str = Field(default="linguistics_ts")

    # Mistral backend
    mistral_api_url: str = Field(default="https://api.mistral.ai/v1")
    mistral_api_key: str | None = Field(default=None)
    mistral_chat_model: str = Field(default="mistral-medium-latest")
    mistral_embed_model: str = Field(default="mistral-embed")
    mistral_timeout_s: float = Field(default=30.0)
    mistral_max_retries: int = Field(default=3)

    # Stability
    breaker_failure_threshold: float = Field(default=0.05, description="Open above this ratio in window.")
    breaker_window_s: int = Field(default=60)
    breaker_cooldown_s: int = Field(default=30)
    breaker_min_requests: int = Field(default=20)
    spend_cap_usd_daily: float = Field(default=50.0)

    # Token cost for spend tracking (USD per 1M tokens; approximate mistral-small pricing).
    # mistral-medium-latest pricing (Apr 2026): $0.40 / $2.00 per MTok.
    mistral_price_input_per_mtok: float = Field(default=0.40)
    mistral_price_output_per_mtok: float = Field(default=2.00)
    mistral_price_embed_per_mtok: float = Field(default=0.10)

    # Local models
    nllb_model: str = Field(default="facebook/nllb-200-distilled-600M")
    labse_model: str = Field(default="sentence-transformers/LaBSE")
    local_models_path: str = Field(default="/models")

    # LRU
    inprocess_lru_size: int = Field(default=1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
