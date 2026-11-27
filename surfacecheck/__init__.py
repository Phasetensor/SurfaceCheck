"""Strike-shape feasibility for a single European call expiry."""

import numpy as np
from scipy.optimize import linprog


def check_slice(strikes, bids, asks, discount=1.0):
    """Test strike shape, returning midpoint violations and feasible prices."""
    k, bid, ask = (np.asarray(x, dtype=float) for x in (strikes, bids, asks))
    if k.ndim != 1 or len(k) < 2 or bid.shape != k.shape or ask.shape != k.shape:
        raise ValueError("Supply equal-length vectors with at least two strikes")
    if not all(np.isfinite(x).all() for x in (k, bid, ask, discount)):
        raise ValueError("Inputs must be finite")
    if np.any(k < 0) or np.any(np.diff(k) <= 0) or np.any(bid < 0) or np.any(ask < bid) or discount <= 0:
        raise ValueError("Invalid strikes, spreads or discount factor")
    slopes = np.diff(np.eye(len(k)), axis=0) / np.diff(k)[:, None]
    rows = np.vstack((slopes, -slopes, -np.diff(slopes, axis=0)))
    limits = np.r_[np.zeros(len(slopes)), np.full(len(slopes), discount), np.zeros(len(slopes)-1)]
    mid = bid / 2 + ask / 2
    violations = [k[row != 0].tolist() for row in rows[rows @ mid > limits + 1e-9]]
    if not violations:
        return dict(status="mid_feasible", violations=[], prices=mid.tolist())
    fit = linprog(np.zeros(len(k)), A_ub=rows, b_ub=limits, bounds=list(zip(bid, ask)), method="highs")
    if fit.status == 2:
        return dict(status="spreads_infeasible", violations=violations, prices=None)
    if not fit.success:
        raise RuntimeError(fit.message)
    return dict(status="spreads_feasible", violations=violations, prices=fit.x.tolist())
