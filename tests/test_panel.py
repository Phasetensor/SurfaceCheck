import numpy as np
import pytest
from scipy.stats import lognorm
from surfacecheck.models import SVI
from surfacecheck.surface import (
    repair_panel,
    svi_density,
    from_variance,
    extrapolated_wings,
    stress,
)


def test_panel():
    x = [0.8, 1, 1.3]
    t = [0.5, 1]
    ref = np.array([[0.22, 0.08, 0.03], [0.24, 0.10, 0.05]])

    def run(a, b, flag=True):
        return repair_panel(x, t, a, b, same_normalized_martingale=flag)

    assert run(ref, ref)["status"] == "mid_feasible"
    assert run(ref, ref, False)["status"] == "insufficient_evidence"
    bad = ref[::-1]
    assert run(bad, bad)["status"] == "infeasible"
    repaired = run(bad - 0.025, bad + 0.025)
    assert repaired["status"] == "spread_feasible"
    np.testing.assert_allclose(
        repaired["prices"], np.tile(ref.mean(axis=0), (2, 1)), atol=2e-7
    )
    with pytest.raises(ValueError):
        run(ref, ref - 1)


def test_exact_density():
    m = SVI(0.04, 0, 0, 0, 0.2)
    x = np.linspace(-0.4, 0.4, 21)
    k, ratio, vol = from_variance(x, np.full_like(x, 0.04), 100, 1)
    np.testing.assert_allclose(k, ratio * 100)
    np.testing.assert_allclose(vol, 0.2)
    np.testing.assert_allclose(
        svi_density(m, x, 100), lognorm.pdf(k, s=0.2, scale=100 * np.exp(-0.02))
    )
    with pytest.raises(ValueError):
        svi_density(m, x, 0)
    with pytest.raises(ValueError):
        from_variance(x, [-1], 100, 1)


def test_wings_stresses():
    for rule in ("linear_variance", "flat_variance"):
        r = extrapolated_wings(
            [-1, 0, 1], [0.06, 0.04, 0.05], [-2, 0, 2], 100, 0.98, 1, rule
        )
        assert r["extrapolated"].tolist() == [True, False, True]
        assert np.all(r["calls"] >= 0)
    with pytest.raises(ValueError):
        extrapolated_wings([-1, 0, 1], [1] * 3, [0], 100, 1, 1, "unknown")
    v = np.full((2, 3), 0.2)
    assert stress([-0.2, 0, 0.2], [0.5, 1], v, 0)["calendars"] == [
        "no_violation_detected"
    ]
    assert stress([-0.2, 0, 0.2], [0.5, 1], v, [[0.2], [0]])["calendars"] == [
        "interval_violation"
    ]
    with pytest.raises(ValueError):
        stress([-0.2, 0, 0.2], [0.5, 1], v, -1)


def test_history_alignment_and_certificate():
    from surfacecheck.models import ssvi_surface, SSVI
    from surfacecheck.diagnostics import historical_summary, align_smiles

    common = [-0.2, 0, 0.2]
    a = align_smiles([common, [-0.1, 0.1]], [[0.2] * 3, [0.21] * 2], common)
    assert np.isnan(a[1, 0]) and a[1, 1] == 0.21
    for args in [([], [], common), ([common], [[0.2]], common)]:
        with pytest.raises(ValueError):
            align_smiles(*args)
    panel = np.ones((4, 2, 3)) * 0.2
    result = historical_summary([1, 2, 3, 4], panel, common, [0.5, 1], np.ones((4, 5)))
    np.testing.assert_allclose(result["atm"], 0.2)
    np.testing.assert_allclose(result["skew"], 0)
    with pytest.raises(ValueError):
        historical_summary([1, 2, 3, 4], panel, [1, 2, 3], [0.5, 1])
    with pytest.raises(ValueError):
        historical_summary([1, 2, 3, 4], panel, common, [0.5, 1], [2])
    model = SSVI(-0.3, 0.6)
    assert ssvi_surface(model, [0.5, 1], [0.02, 0.04], common, [0.25, 0.75])[
        "sufficient_conditions_met"
    ]
    assert not ssvi_surface(model, [0.5, 1], [0.04, 0.02], common, [0.25, 0.75])[
        "sufficient_conditions_met"
    ]
    with pytest.raises(ValueError):
        ssvi_surface(model, [0.5, 1], [0.04, 0.02], common, [2])
