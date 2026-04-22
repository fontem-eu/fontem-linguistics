"""SpendCap: reservation/finalize/release lifecycle + rollover."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.domain.models import SpendCapExceeded
from src.infra.spend_cap import SpendCap


pytestmark = pytest.mark.asyncio


async def test_reserve_under_cap_succeeds():
    cap = SpendCap(daily_cap_usd=1.0)
    await cap.reserve(0.3)
    await cap.reserve(0.2)
    assert await cap.spent() == pytest.approx(0.5)


async def test_reserve_over_cap_raises():
    cap = SpendCap(daily_cap_usd=1.0)
    await cap.reserve(0.7)
    with pytest.raises(SpendCapExceeded):
        await cap.reserve(0.5)
    # Failed reservation must not be added.
    assert await cap.spent() == pytest.approx(0.7)


async def test_finalize_adjusts_to_actual():
    cap = SpendCap(daily_cap_usd=10.0)
    await cap.reserve(1.0)           # estimated
    await cap.finalize(1.0, 0.4)     # actual less than estimate
    assert await cap.spent() == pytest.approx(0.4)

    await cap.reserve(0.2)
    await cap.finalize(0.2, 1.1)     # actual more than estimate
    assert await cap.spent() == pytest.approx(0.4 + 1.1)


async def test_release_rolls_back_reservation():
    cap = SpendCap(daily_cap_usd=10.0)
    await cap.reserve(1.5)
    await cap.release(1.5)
    assert await cap.spent() == pytest.approx(0.0)


async def test_release_never_goes_negative():
    cap = SpendCap(daily_cap_usd=10.0)
    await cap.release(5.0)
    assert await cap.spent() == pytest.approx(0.0)


async def test_rollover_on_new_day(monkeypatch):
    cap = SpendCap(daily_cap_usd=1.0)
    await cap.reserve(0.8)
    assert await cap.spent() == pytest.approx(0.8)

    # Simulate a day passing by rewinding _today directly.
    cap._today = date.today() - timedelta(days=1)
    await cap.reserve(0.5)
    assert await cap.spent() == pytest.approx(0.5)
