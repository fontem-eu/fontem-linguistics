"""Settings smoke test — ensures env parsing and defaults stay consistent."""
from __future__ import annotations


def test_settings_loads_defaults(monkeypatch):
    from src.infra.config import Settings
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings(_env_file=None)  # ignore any local .env
    assert s.tablespace == "linguistics_ts"
    assert s.breaker_failure_threshold == 0.05
    assert s.spend_cap_usd_daily == 50.0
    # pylint sees the pydantic Field default as a FieldInfo object rather
    # than the resolved string at runtime; the assertion runs against the
    # real value.
    assert s.mistral_chat_model.startswith("mistral-")  # pylint: disable=no-member


def test_settings_reads_env(monkeypatch):
    from src.infra.config import Settings
    monkeypatch.setenv("SPEND_CAP_USD_DAILY", "12.5")
    monkeypatch.setenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
    s = Settings(_env_file=None)
    assert s.spend_cap_usd_daily == 12.5
    assert s.mistral_chat_model == "mistral-large-latest"


def test_local_quantize_int8_default_off():
    # Counter-intuitive but measured: eager-mode `quantize_dynamic` on
    # NLLB-200 on our torch 2.11 / transformers 5.5 stack grows RSS
    # (fp32 0.84 GB → ~3.4 GB quantised). Default off; fp32 is smaller.
    from src.infra.config import Settings
    s = Settings(_env_file=None)
    assert s.local_quantize_int8 is False


def test_local_quantize_int8_overridable(monkeypatch):
    from src.infra.config import Settings
    monkeypatch.setenv("LOCAL_QUANTIZE_INT8", "true")
    s = Settings(_env_file=None)
    assert s.local_quantize_int8 is True
