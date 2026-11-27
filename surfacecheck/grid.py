"""Discrete grid diagnostics. All adjacent strike slopes use actual spacing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def vector(x: ArrayLike, minimum: int = 1) -> NDArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError("Expected finite vector of sufficient length")
    return a


def axis(x: ArrayLike, minimum: int = 2) -> NDArray:
    a = vector(x, minimum)
    if np.any(np.diff(a) <= 0):
        raise ValueError("Coordinates must increase strictly")
    return a


def coordinates(
    strikes: ArrayLike, forward: float, years: float, vol: ArrayLike
) -> tuple[NDArray, NDArray]:
    k, v = vector(strikes), vector(vol)
    if (
        k.shape != v.shape
        or np.any(k <= 0)
        or np.any(v < 0)
        or not np.isfinite([forward, years]).all()
        or min(forward, years) <= 0
    ):
        raise ValueError("Invalid smile or conventions")
    return np.log(k) - np.log(forward), v * v * years


def repair(
    strikes: ArrayLike,
    bids: ArrayLike,
    asks: ArrayLike,
    forward: float,
    discount: float,
    weights: ArrayLike | None = None,
    tol: float = 1e-8,
) -> dict:
    k, b, a = axis(strikes), vector(bids), vector(asks)
    w = np.ones_like(k) if weights is None else vector(weights)
    if (
        any(x.shape != k.shape for x in (b, a, w))
        or np.any(b < 0)
        or np.any(a < b)
        or np.any(w <= 0)
        or not np.isfinite([forward, discount, tol]).all()
        or min(k[0], forward, discount, tol) <= 0
    ):
        raise ValueError("Invalid grid or bounds")
    s = np.diff(np.eye(len(k)), axis=0) / np.diff(k)[:, None]
    # A p <= h, including price bounds, monotonicity, verticals and butterflies.
    eye = np.eye(len(k))
    A = np.vstack((-eye, eye, s, -s, -np.diff(s, axis=0)))
    h = np.r_[
        -discount * np.maximum(forward - k, 0),
        np.full(len(k), discount * forward),
        np.zeros(len(s)),
        np.full(len(s), discount),
        np.zeros(len(s) - 1),
    ]
    mid = b / 2 + a / 2
    residual = A @ mid - h
    base = dict(
        original=mid,
        bids=b,
        asks=a,
        violated_strikes=[k[row != 0].tolist() for row in A[residual > tol]],
        prices=None,
        adjustments=None,
        max_residual=None,
    )
    if max(residual) <= tol:
        return dict(
            base,
            status="mid_feasible",
            prices=mid,
            adjustments=np.zeros_like(k),
            max_residual=float(max(0, max(residual))),
        )
    fit = linprog(
        np.zeros_like(k), A_ub=A, b_ub=h, bounds=list(zip(b, a)), method="highs"
    )
    if not fit.success:
        return dict(
            base,
            status="infeasible" if fit.status == 2 else "numerical_failure",
            solver=fit.message,
        )
    result = minimize(
        lambda p: 0.5 * np.dot(w, (p - mid) ** 2),
        fit.x,
        jac=lambda p: w * (p - mid),
        bounds=Bounds(b, a),
        constraints=[LinearConstraint(A, -np.inf, h)],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    p = result.x
    err = float(max(0, max(A @ p - h), max(b - p), max(p - a)))
    if not result.success or not np.isfinite(p).all() or err > tol:
        return dict(
            base, status="numerical_failure", solver=result.message, max_residual=err
        )
    return dict(
        base,
        status="spread_feasible",
        prices=p,
        adjustments=p - mid,
        max_residual=err,
        solver=result.message,
    )


def interpolate(
    x: ArrayLike, y: ArrayLike, query: ArrayLike, method: str = "linear"
) -> NDArray:
    knots, values, q = axis(x), vector(y), vector(query)
    if values.shape != knots.shape or min(q) < knots[0] or max(q) > knots[-1]:
        raise ValueError("Mismatched data or extrapolation")
    if method == "linear":
        return np.interp(q, knots, values)
    if method not in ("pchip", "cubic"):
        raise ValueError("Unknown interpolator")
    return (PchipInterpolator if method == "pchip" else CubicSpline)(knots, values)(q)


def calendar(
    k1: ArrayLike,
    early_low: ArrayLike,
    early_high: ArrayLike,
    k2: ArrayLike,
    late_low: ArrayLike,
    late_high: ArrayLike,
    t1: float,
    t2: float,
    *,
    same_normalized_martingale: bool,
    tol: float = 1e-10,
) -> dict:
    """Linear total-variance intervals on common support; conditional on interpolation."""
    x, y = axis(k1), axis(k2)
    if not 0 < t1 < t2 or not np.isfinite([t1, t2, tol]).all() or tol < 0:
        raise ValueError("Invalid maturities or tolerance")
    for z, lo, hi in ((x, early_low, early_high), (y, late_low, late_high)):
        l, u = vector(lo), vector(hi)
        if l.shape != z.shape or u.shape != z.shape or np.any(l < 0) or np.any(u < l):
            raise ValueError("Invalid variance intervals")
    nodes = np.union1d(x, y)
    nodes = nodes[(nodes >= max(x[0], y[0])) & (nodes <= min(x[-1], y[-1]))]
    if not same_normalized_martingale or not nodes.size:
        return dict(status="insufficient_evidence")
    gap = np.interp(nodes, y, vector(late_high)) - np.interp(nodes, x, vector(early_low))
    direct = np.isin(nodes, x) & np.isin(nodes, y)
    return dict(
        status="interval_violation"
        if np.any((gap < -tol) & direct)
        else "interpolated_violation"
        if np.any(gap < -tol)
        else "no_violation_detected",
        nodes=nodes,
        gap=gap,
        shared_observations=direct,
    )
