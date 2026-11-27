import itertools
import numpy as np
from surfacecheck.grid import repair
from surfacecheck.pricing import black
from surfacecheck.models import SVI, SSVI
from surfacecheck.diagnostics import density, local_variance


def vertices_feasible(bid, ask):
    # Enumerate all three-active-plane intersections on a fixed tiny polytope.
    # Ordered strikes 90,100,120; F=100,D=1. No production constraint builder.
    rows = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [-1, 1, 0],
            [0, -1, 1],
            [1, -1, 0],
            [0, 1, -1],
            [-2, 3, -1],
        ]
    )
    bounds = np.r_[np.minimum(100, ask), -np.maximum([10, 0, 0], bid), 0, 0, 10, 20, 0]
    for idx in itertools.combinations(range(len(rows)), 3):
        ids = list(idx)
        if abs(np.linalg.det(rows[ids])) < 1e-10:
            continue
        candidate = np.linalg.solve(rows[ids], bounds[ids])
        if np.all(rows @ candidate <= bounds + 1e-9):
            return True
    return False


def test_exhaustive_feasibility():
    rng = np.random.default_rng(805)
    for _ in range(24):
        bid = np.maximum(0, [15, 9, 3] + rng.normal(0, 4, 3))
        ask = bid + rng.uniform(0, 3, 3)
        result = repair([90, 100, 120], bid, ask, 100, 1)
        assert result["status"] in ("infeasible", "spread_feasible", "mid_feasible")
        assert (result["status"] != "infeasible") == vertices_feasible(bid, ask)


def test_convergence_and_signed_variance():
    errors = []
    for n in (31, 61, 121):
        k = np.linspace(70, 130, n)
        c = [black(100, s, 1, 1, 0.2) for s in k]
        obs = density(k, c, 1)
        at = np.argmin(abs(obs["strikes"] - 100))
        exact = np.exp(-(0.1**2) / 2) / (20 * np.sqrt(2 * np.pi))
        errors.append(abs(obs["density"][at] - exact))
    assert errors[0] > 3.5 * errors[1] and errors[1] > 3.5 * errors[2]
    k = [90, 100, 110]
    prices = np.array([[black(100, s, 1, 1, v) for s in k] for v in [0.3, 0.25, 0.2]])
    assert local_variance(k, [1, 2, 3], prices, [0] * 3, [0] * 3)["variance"][0, 0] < 0


def test_positive_svi_can_violate_butterfly():
    m = SVI(0.001, 2.0, -0.8, 0.0, 0.05)
    assert min(m.derivatives([-0.5, 0, 0.5])[0]) > 0
    assert min(m.g(np.linspace(-3, 3, 301))) < 0
    assert not SSVI(0.8, 3.0).sufficient()
