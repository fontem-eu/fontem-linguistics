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
    # LaBSE is loaded from a local path populated by the pod's InitContainer
    # (oras-pulled, cosign-verified OCI artifact). `labse_model` stays as a
    # legacy config knob but is no longer read by the backend — every
    # labse-local deploy must set labse_model_path + labse_encoder_id.
    labse_model: str = Field(default="sentence-transformers/LaBSE")
    labse_model_path: str = Field(
        default="/models/labse-1.0.0",
        description="Filesystem path to the mirrored LaBSE snapshot.",
    )
    labse_encoder_id: str = Field(
        default="labse@1.0.0-836121a",
        description=(
            "Signed-mirror identity of the loaded LaBSE, surfaced on every "
            "/embed response. Must match the tag under which the artifact "
            "was pushed by the mirror-labse workflow."
        ),
    )
    local_models_path: str = Field(default="/models")
    # Dynamic int8 quantisation on nn.Linear layers. Intuition says this
    # should shrink the resident footprint; in practice, measured on
    # torch 2.11 + transformers 5.5, eager-mode `quantize_dynamic` on
    # NLLB-200 allocates per-layer scale + zero-point tensors AND keeps
    # the fp32 state alive long enough that total RSS goes UP (0.84 GB
    # fp32 → ~3.4 GB quantised, observed in prod pod). Default off.
    # Leave the toggle in place so we can re-enable once we migrate to
    # torchao's int8_weight_only (the non-deprecated API) and confirm
    # the savings there.
    local_quantize_int8: bool = Field(default=False)

    # LRU
    inprocess_lru_size: int = Field(default=1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
