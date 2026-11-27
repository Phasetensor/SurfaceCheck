import pytest

from surfacecheck.pricing import black, implied_vol


h = pytest.importorskip("hypothesis")
given, settings, st = h.given, h.settings, h.strategies

@given(st.floats(75, 130), st.floats(0.15, 0.8), st.floats(0.2, 2))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_black_properties(k, sigma, t):
    c = black(100, k, t, 0.98, sigma)
    p = black(100, k, t, 0.98, sigma, "put")
    assert 0.98 * max(100 - k, 0) <= c <= 98
    assert c - p == pytest.approx(0.98 * (100 - k), abs=1e-11)
    result = implied_vol(c, 100, k, t, 0.98)
    if result.volatility is not None:
        assert black(100, k, t, 0.98, result.volatility) == pytest.approx(c, abs=1e-10)
