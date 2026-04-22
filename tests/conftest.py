"""Test-wide fixtures + safety checks.

Critical: the real MISTRAL_API_KEY must never be read during tests. The fixture
below enforces it by failing fast if the env var is set to anything other than
a stub value (prefixed with `test-` by our stub).
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _forbid_real_mistral_key(monkeypatch):
    """Fail any test that would pick up a production Mistral key."""
    val = os.environ.get("MISTRAL_API_KEY", "")
    if val and not val.startswith("test-"):
        pytest.fail(
            "MISTRAL_API_KEY is set to a non-test value during tests. "
            "Unset it or export MISTRAL_API_KEY=test-xxx before running pytest."
        )
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-never-used")
    yield
