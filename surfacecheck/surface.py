"""Normalized-grid repair and conditional scenarios on a common forward basis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from .grid import axis, vector, calendar, repair
from .pricing import black
from .models import SVI


def from_variance(
    log_moneyness: ArrayLike, w: ArrayLike, forward: float, years: float
) -> tuple[NDArray, NDArray, NDArray]:
    x, v = vector(log_moneyness), vector(w)
    if (
        x.shape != v.shape
        or np.any(v < 0)
        or not np.isfinite([forward, years]).all()
        or min(forward, years) <= 0
    ):
        raise ValueError("Invalid normalized coordinates")
    m = np.exp(x)
    return forward * m, m, np.sqrt(v / years)


def svi_density(model: SVI, k: ArrayLike, forward: float) -> NDArray:
    x = vector(k)
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("Positive forward required")
    w = model.derivatives(x)[0]
    z = -x / np.sqrt(w) - np.sqrt(w) / 2
    return (
        np.exp(-0.5 * z * z)
        * model.g(x)
        / (forward * np.exp(x) * np.sqrt(2 * np.pi * w))
    )


def extrapolated_wings(
    k: ArrayLike,
    w: ArrayLike,
    query: ArrayLike,
    forward: float,
    discount: float,
    years: float,
    rule: str,
) -> dict:
    x, y, z = axis(k, 3), vector(w), vector(query)
    if (
        y.shape != x.shape
        or min(y) <= 0
        or rule not in ("flat_variance", "linear_variance")
        or min(years, forward, discount) <= 0
    ):
        raise ValueError("Invalid wings or conventions")
    result = np.interp(z, x, y)
    if rule == "linear_variance":
        for mask, i, j in ((z < x[0], 0, 1), (z > x[-1], -1, -2)):
            result[mask] = y[i] + (z[mask] - x[i]) * (y[i] - y[j]) / (x[i] - x[j])
    prices = np.array(
        [
            black(forward, forward * np.exp(a), years, discount, np.sqrt(b / years))
            if b >= 0
            else np.nan
            for a, b in zip(z, result)
        ]
    )
    return dict(
        variance=result, calls=prices, extrapolated=(z < x[0]) | (z > x[-1]), rule=rule
    )


def stress(k: ArrayLike, times: ArrayLike, vols: ArrayLike, shock: ArrayLike) -> dict:
    x, t = axis(k), axis(times)
    v = np.asarray(vols, float) + np.asarray(shock, float)
    if v.shape != (len(t), len(x)) or np.any(v < 0) or not np.isfinite(v).all():
        raise ValueError("Invalid aligned volatility shock")
    w = v * v * t[:, None]
    calls = np.array(
        [[black(1, np.exp(a), u, 1, s) for a, s in zip(x, row)] for u, row in zip(t, v)]
    )
    return dict(
        vols=v,
        smiles=[repair(np.exp(x), p, p, 1, 1)["status"] for p in calls],
        calendars=[
            calendar(x, a, a, x, b, b, u, z, same_normalized_martingale=True)["status"]
            for a, b, u, z in zip(w, w[1:], t, t[1:])
        ],
    )


def repair_panel(
    moneyness: ArrayLike,
    times: ArrayLike,
    bids: ArrayLike,
    asks: ArrayLike,
    *,
    same_normalized_martingale: bool,
    weights: ArrayLike | None = None,
    tol: float = 1e-8,
) -> dict:
    """Minimize weighted squared displacement in normalized call units C/(DF)."""
    x, t = axis(moneyness), axis(times)
    b, a = np.asarray(bids, float), np.asarray(asks, float)
    w = np.ones_like(b) if weights is None else np.asarray(weights, float)
    if not same_normalized_martingale:
        return dict(status="insufficient_evidence")
    if (
        min(x) <= 0
        or min(t) <= 0
        or b.shape != (len(t), len(x))
        or a.shape != b.shape
        or w.shape != b.shape
        or not np.isfinite([b, a, w]).all()
        or np.any(a < b)
        or np.any(b < 0)
        or np.any(w <= 0)
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("Invalid normalized panel")
    eye = np.eye(len(x))
    slopes = np.diff(eye, axis=0) / np.diff(x)[:, None]
    rows = np.vstack((eye, -eye, slopes, -slopes, -np.diff(slopes, axis=0)))
    h = np.r_[
        np.ones_like(x),
        -np.maximum(1 - x, 0),
        np.zeros(len(slopes)),
        np.ones(len(slopes)),
        np.zeros(len(slopes) - 1),
    ]
    A = np.vstack(
        (np.kron(np.eye(len(t)), rows), -np.kron(np.diff(np.eye(len(t)), axis=0), eye))
    )
    limits = np.r_[np.tile(h, len(t)), np.zeros((len(t) - 1) * len(x))]
    mid = (b / 2 + a / 2).ravel()
    base = dict(
        original=mid.reshape(b.shape),
        bids=b,
        asks=a,
        violated_rows=np.flatnonzero(A @ mid > limits + tol),
    )
    if not len(base["violated_rows"]):
        return dict(
            base,
            status="mid_feasible",
            prices=mid.reshape(b.shape),
            adjustments=np.zeros_like(b),
            max_residual=max(0.0, float(max(A @ mid - limits))),
        )
    bounds = list(zip(b.ravel(), a.ravel()))
    fit = linprog(np.zeros_like(mid), A_ub=A, b_ub=limits, bounds=bounds)
    if not fit.success:
        return dict(
            base,
            status="infeasible" if fit.status == 2 else "numerical_failure",
            solver=fit.message,
        )
    result = minimize(
        lambda p: 0.5 * np.dot(w.ravel(), (p - mid) ** 2),
        fit.x,
        jac=lambda p: w.ravel() * (p - mid),
        constraints=[LinearConstraint(A, -np.inf, limits)],
        bounds=Bounds(b.ravel(), a.ravel()),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    p = result.x.reshape(b.shape)
    error = max(
        0.0,
        float(np.max(b - p)),
        float(np.max(p - a)),
        float(max(A @ result.x - limits)),
    )
    if not result.success or error > tol or not np.isfinite(p).all():
        return dict(base, status="numerical_failure", solver=result.message)
    return dict(
        base,
        status="spread_feasible",
        prices=p,
        adjustments=p - mid.reshape(b.shape),
        solver=result.message,
        max_residual=error,
    )
