"""Settings smoke test — ensures env parsing and defaults stay consistent."""
from __future__ import annotations


def test_settings_loads_defaults(monkeypatch):
    from src.infra.config import Settings
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings(_env_file=None)  # ignore any local .env
    assert s.tablespace == "linguistics_ts"
    assert s.breaker_failure_threshold == 0.05
    assert s.spend_cap_usd_daily == 50.0
    assert s.mistral_chat_model.startswith("mistral-")


def test_settings_reads_env(monkeypatch):
    from src.infra.config import Settings
    monkeypatch.setenv("SPEND_CAP_USD_DAILY", "12.5")
    monkeypatch.setenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
    s = Settings(_env_file=None)
    assert s.spend_cap_usd_daily == 12.5
    assert s.mistral_chat_model == "mistral-large-latest"
