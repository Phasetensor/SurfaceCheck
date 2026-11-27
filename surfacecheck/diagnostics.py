"""Finite-support derivative, extrapolation and aligned-history diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from .grid import axis, vector


def curvature(k: ArrayLike, c: ArrayLike) -> NDArray:
    x, y = axis(k, 3), vector(c)
    if x.shape != y.shape or min(x) <= 0:
        raise ValueError("Invalid strike grid")
    return 2 * np.diff(np.diff(y) / np.diff(x)) / (x[2:] - x[:-2])


def density(k: ArrayLike, c: ArrayLike, discount: float) -> dict:
    x = axis(k, 4)
    if not np.isfinite(discount) or discount <= 0:
        raise ValueError("Invalid discount factor")
    f = curvature(x, c) / discount
    interior = x[1:-1]
    return dict(
        strikes=interior,
        density=f,
        negative=interior[f < 0],
        mass=float(trapezoid(f, interior)),
        first_moment=float(trapezoid(f * interior, interior)),
        support=(interior[0], interior[-1]),
    )


def local_variance(
    k: ArrayLike,
    t: ArrayLike,
    calls: ArrayLike,
    rates: ArrayLike,
    dividends: ArrayLike,
    denom_tol: float = 1e-12,
) -> dict:
    x, y = axis(k, 3), axis(t, 3)
    c = np.asarray(calls, float)
    r, q = vector(rates), vector(dividends)
    if (
        min(y) <= 0
        or c.shape != (len(y), len(x))
        or not np.isfinite(c).all()
        or r.shape != y.shape
        or q.shape != y.shape
        or not np.isfinite(denom_tol)
        or denom_tol <= 0
    ):
        raise ValueError("Invalid common-strike surface or rate curves")
    ct = np.gradient(c, y, axis=0, edge_order=2)
    ck = np.gradient(c, x, axis=1, edge_order=2)
    gamma = np.stack([curvature(x, row) for row in c])
    den = x[None, 1:-1] ** 2 * gamma
    num = 2 * (ct + q[:, None] * c + (r - q)[:, None] * x * ck)[:, 1:-1]
    valid = den > denom_tol
    variance = np.divide(num, den, out=np.full_like(den, np.nan), where=valid)
    return dict(
        strikes=x[1:-1],
        times=y[1:-1],
        variance=variance[1:-1],
        denominator=den[1:-1],
        invalid=(~valid)[1:-1],
    )


def wing_slopes(k: ArrayLike, w: ArrayLike) -> dict:
    x, y = axis(k, 3), vector(w)
    if x.shape != y.shape or min(y) <= 0 or not x[0] < 0 < x[-1]:
        raise ValueError("Need both observed wings")
    s = np.diff(y) / np.diff(x)
    return dict(
        left_secant=-s[0],
        right_secant=s[-1],
        support=(x[0], x[-1]),
        evidence="finite_secants_only",
    )


def historical_pca(
    times: ArrayLike,
    vols: ArrayLike,
    k: ArrayLike,
    maturities: ArrayLike,
    mode: str = "absolute",
) -> dict:
    """Complete aligned panels only. Caller supplies increasing as-of times and common axes."""
    clock = axis(times, 3)
    x, y = axis(k), axis(maturities)
    v = np.asarray(vols, float)
    if (
        v.shape != (len(clock), len(y), len(x))
        or not np.isfinite(v).all()
        or np.any(v <= 0)
        or min(y) <= 0
        or mode not in ("absolute", "relative", "variance")
    ):
        raise ValueError("Invalid aligned panel, missing observations or mode")
    level = v * v * y[None, :, None] if mode == "variance" else v
    change = np.diff(level, axis=0)
    if mode == "relative":
        change /= level[:-1]
    flat = change.reshape(len(change), -1)
    _, s, vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    return dict(
        changes=change,
        factors=vt.reshape((-1, len(y), len(x))),
        variances=s * s / (len(flat) - 1),
        mean_change=flat.mean(axis=0),
        training_end=clock[-1],
    )


def historical_summary(times: ArrayLike,vols: ArrayLike,k: ArrayLike,maturities: ArrayLike,parameters: ArrayLike | None=None) -> dict:
    """Daily histories at fixed log-forward moneyness. Irregular dates keep period changes."""
    clock=axis(times,3)
    x,terms=axis(k),axis(maturities)
    panel=np.asarray(vols,float)
    if not x[0]<=0<=x[-1]:
        raise ValueError("ATM must lie inside common support")
    pca=historical_pca(clock,panel,x,terms)
    atm=np.stack([[np.interp(0.,x,row) for row in day] for day in panel])
    left=max(0,int(np.searchsorted(x,0.,side="left"))-1)
    right=min(len(x)-1,int(np.searchsorted(x,0.,side="right")))
    skew=(panel[:,:,right]-panel[:,:,left])/(x[right]-x[left])
    history=None if parameters is None else np.asarray(parameters,float)
    if history is not None and (history.ndim!=2 or len(history)!=len(clock) or not np.isfinite(history).all()):
        raise ValueError("Parameter history must align with snapshots")
    return dict(times=clock,atm=atm,skew=skew,atm_changes=np.diff(atm,axis=0),skew_changes=np.diff(skew,axis=0),term_spreads=np.diff(atm,axis=1),parameters=history,pca=pca)


def align_smiles(grids: list[ArrayLike],vols: list[ArrayLike],target: ArrayLike) -> NDArray:
    """Interpolate each smile inside observed support; emit NaN for unsupported cells."""
    if not grids or len(grids)!=len(vols):
        raise ValueError("One volatility vector per smile required")
    q=axis(target)
    rows=[]
    for grid,values in zip(grids,vols):
        x,v=axis(grid),vector(values)
        if x.shape!=v.shape or np.any(v<0):
            raise ValueError("Invalid smile")
        rows.append(np.interp(q,x,v,left=np.nan,right=np.nan))
    return np.stack(rows)
