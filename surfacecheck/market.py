"""Snapshot conventions, temporal eligibility and interval parity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from .pricing import _inputs


@dataclass(frozen=True)
class Quote:
    underlying: str
    observed: datetime
    available: datetime
    expiry: datetime
    kind: str
    strike: float
    bid: float
    ask: float
    forward: float
    discount: float
    years: float
    currency: str
    provenance: str
    units: str = "currency_per_unit"
    basis: str = "ACT/365F"

    def __post_init__(self) -> None:
        _inputs(self.forward, self.strike, self.years, self.discount, self.kind)
        if any(
            x.utcoffset() is None for x in (self.observed, self.available, self.expiry)
        ):
            raise ValueError("Timezone required")
        for name in ("observed", "available", "expiry"):
            object.__setattr__(self, name, getattr(self, name).astimezone(timezone.utc))
        if self.available < self.observed or self.expiry <= self.observed:
            raise ValueError("Inconsistent timestamps")
        if self.units != "currency_per_unit" or not all(
            x.strip() for x in (self.currency, self.provenance, self.underlying)
        ):
            raise ValueError("Missing metadata or unsupported units")
        if self.basis not in ("ACT/365F", "ACT/360"):
            raise ValueError("Unsupported day count")
        days = (self.expiry - self.observed).total_seconds() / 86400
        if abs(self.years - days / (365 if self.basis == "ACT/365F" else 360)) > 1e-10:
            raise ValueError("Expiration and year fraction disagree")
        if (
            not all(isfinite(x) for x in (self.bid, self.ask))
            or not 0 <= self.bid <= self.ask
        ):
            raise ValueError("Invalid bid/ask")


def snapshot(quotes: list[Quote], at: datetime, max_age: timedelta) -> list[Quote]:
    if at.utcoffset() is None or max_age.total_seconds() < 0:
        raise ValueError("Invalid eligibility policy")
    selected = [
        q for q in quotes if q.available <= at < q.expiry and at - q.observed <= max_age
    ]
    keys = [(q.underlying, q.currency, q.expiry, q.strike, q.kind) for q in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Resolve duplicate or competing quotes before validation")
    return selected


def parity(call: Quote, put: Quote | None, tol: float = 1e-10) -> str:
    if not isfinite(tol) or tol < 0:
        raise ValueError("Invalid parity tolerance")
    if put is None:
        return "insufficient_evidence"
    fields = (
        "underlying",
        "observed",
        "expiry",
        "strike",
        "currency",
        "basis",
        "forward",
        "discount",
    )
    if (
        call.kind != "call"
        or put.kind != "put"
        or any(getattr(call, n) != getattr(put, n) for n in fields)
    ):
        raise ValueError("Different contracts or conventions")
    target = call.discount * (call.forward - call.strike)
    low, high = call.bid - put.ask, call.ask - put.bid
    if target < low - tol or target > high + tol:
        return "interval_violation"
    return (
        "mid_consistent"
        if abs((low + high) / 2 - target) <= tol
        else "spread_compatible"
    )
