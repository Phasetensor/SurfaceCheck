"""Optional published-package integrations, imported on demand."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from math import exp, isfinite


def historical_snapshot(releases: list[dict], decision: datetime) -> list[dict]:
    from vintagecheck import VintageIndex

    return VintageIndex(releases, policy="operator").snapshot(decision)


def forward_inputs(
    curve: Any,
    valuation: date,
    expiry: date,
    currency: str,
    spot: float,
    continuous_dividend: float,
) -> tuple[float, float, float]:
    from curvemetric import year_fraction

    c = curve.conventions
    if (
        c.valuation_date != valuation.isoformat()
        or c.currency != currency
        or c.day_count not in ("ACT/365F", "ACT/360")
    ):
        raise ValueError("Curve snapshot or conventions disagree")
    years = year_fraction(valuation, expiry, c.day_count)
    if (
        years <= 0
        or not isfinite(spot)
        or spot <= 0
        or not isfinite(continuous_dividend)
    ):
        raise ValueError("Invalid expiry, spot or carry")
    discount = float(curve.discount([years])[0])
    return years, discount, spot * exp(-continuous_dividend * years) / discount
