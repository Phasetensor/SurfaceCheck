import pytest
h=pytest.importorskip("hypothesis")
from hypothesis import given,settings,strategies as st
from surfacecheck.pricing import black,implied_vol

@given(st.floats(75,130),st.floats(.15,.8),st.floats(.2,2))
@settings(max_examples=100,deadline=None,derandomize=True)
def test_black_properties(k,sigma,t):
    c=black(100,k,t,.98,sigma)
    p=black(100,k,t,.98,sigma,"put")
    assert .98*max(100-k,0)<=c<=98
    assert c-p==pytest.approx(.98*(100-k),abs=1e-11)
    result=implied_vol(c,100,k,t,.98)
    if result.volatility is not None:
        assert black(100,k,t,.98,result.volatility)==pytest.approx(c,abs=1e-10)
