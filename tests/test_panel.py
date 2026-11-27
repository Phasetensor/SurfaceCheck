import numpy as np
import pytest
from scipy.stats import lognorm
from surfacecheck.models import SVI
from surfacecheck.surface import repair_panel,svi_density,from_variance,extrapolated_wings,stress

def test_panel():
    x=[.8,1,1.3]
    t=[.5,1]
    ref=np.array([[.22,.08,.03],[.24,.10,.05]])
    def run(a,b,flag=True):
        return repair_panel(x,t,a,b,same_normalized_martingale=flag)
    assert run(ref,ref)["status"]=="mid_feasible"
    assert run(ref,ref,False)["status"]=="insufficient_evidence"
    bad=ref[::-1]
    assert run(bad,bad)["status"]=="infeasible"
    repaired=run(bad-.025,bad+.025)
    assert repaired["status"]=="spread_feasible"
    np.testing.assert_allclose(repaired["prices"],np.tile(ref.mean(axis=0),(2,1)),atol=2e-7)
    with pytest.raises(ValueError):
        run(ref,ref-1)

def test_exact_density():
    m=SVI(.04,0,0,0,.2)
    x=np.linspace(-.4,.4,21)
    k,ratio,vol=from_variance(x,np.full_like(x,.04),100,1)
    np.testing.assert_allclose(k,ratio*100)
    np.testing.assert_allclose(vol,.2)
    np.testing.assert_allclose(svi_density(m,x,100),lognorm.pdf(k,s=.2,scale=100*np.exp(-.02)))
    with pytest.raises(ValueError):
        svi_density(m,x,0)
    with pytest.raises(ValueError):
        from_variance(x,[-1],100,1)

def test_wings_stresses():
    for rule in ("linear_variance","flat_variance"):
        r=extrapolated_wings([-1,0,1],[.06,.04,.05],[-2,0,2],100,.98,1,rule)
        assert r["extrapolated"].tolist()==[True,False,True]
        assert np.all(r["calls"]>=0)
    with pytest.raises(ValueError):
        extrapolated_wings([-1,0,1],[1]*3,[0],100,1,1,"unknown")
    v=np.full((2,3),.2)
    assert stress([-.2,0,.2],[.5,1],v,0)["calendars"]==["no_violation_detected"]
    assert stress([-.2,0,.2],[.5,1],v,[[.2],[0]])["calendars"]==["interval_violation"]
    with pytest.raises(ValueError):
        stress([-.2,0,.2],[.5,1],v,-1)
