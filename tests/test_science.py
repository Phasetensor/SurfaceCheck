from __future__ import annotations

import json
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
from scipy.stats import norm
from surfacecheck.market import Quote, snapshot, parity
from surfacecheck.pricing import black, implied_vol
from surfacecheck.grid import coordinates, repair, calendar, interpolate
from surfacecheck.models import SVI,SSVI,fit_svi,fit_ssvi,stability,propagation
from surfacecheck.diagnostics import curvature,density,local_variance,historical_pca,wing_slopes
from surfacecheck.reporting import Report


@pytest.mark.parametrize("method",["brent","bisect","newton"])
@pytest.mark.parametrize("kind",["call","put"])
@pytest.mark.parametrize("f,k,t,v",[(100,100,.01,.3),(100,150,.5,.5),(120,90,1,.25),(100,100,1,4)])
def test_inversion(method,kind,f,k,t,v):
    d=.97
    s=v*np.sqrt(t)
    z=np.log(f/k)/s+s/2
    sign=1 if kind=="call" else -1
    price=sign*d*(f*norm.cdf(sign*z)-k*norm.cdf(sign*(z-s)))
    assert black(f,k,t,d,v,kind)==pytest.approx(price,abs=2e-12)
    result=implied_vol(price,f,k,t,d,kind,method)
    assert result.status=="valid"
    assert result.volatility==pytest.approx(v,abs=1e-10)


def test_edges():
    for p,status in [(-1,"infeasible"),(101,"infeasible"),(100,"boundary"),(0,"boundary"),(1e-12,"poor_conditioning")]:
        assert implied_vol(p,100,100,1,1).status==status
    for method in ("brent","bisect","newton"):
        assert implied_vol(10,100,100,1,1,method=method,maxiter=1).status=="nonconvergence"
    for args in [(0,100,1,1,.2),(100,100,1,1,-.2)]:
        with pytest.raises(ValueError):
            black(*args)
    with pytest.raises(ValueError):
        implied_vol(float("nan"),100,100,1,1)


def test_market():
    now=datetime(2025,1,1,tzinfo=timezone.utc)
    fields=dict(underlying="X",observed=now,available=now,expiry=now+timedelta(days=365),strike=100,bid=5,ask=7,forward=100,discount=.99,years=1,currency="USD",provenance="synthetic")
    c=Quote(kind="call",**fields)
    p=Quote(kind="put",**fields)
    assert parity(c,p)=="mid_consistent"
    assert parity(c,None)=="insufficient_evidence"
    assert parity(c,Quote(kind="put",**dict(fields,bid=6)))=="spread_compatible"
    assert parity(c,Quote(kind="put",**dict(fields,bid=8,ask=9)))=="interval_violation"
    with pytest.raises(ValueError):
        parity(c,c)
    assert snapshot([c],now,timedelta(0))==[c]
    assert snapshot([c],now+timedelta(hours=1),timedelta(0))==[]
    with pytest.raises(ValueError):
        snapshot([c,c],now,timedelta(0))
    for bad in (dict(bid=8),dict(years=2),dict(basis="x"),dict(currency=""),dict(available=now.replace(tzinfo=None)),dict(expiry=now)):
        with pytest.raises(ValueError):
            Quote(kind="call",**dict(fields,**bad))


def test_grid():
    k=[80,95,100,120]
    mid=np.array([23.,14.,13.,2.])
    a=repair(k,mid-2,mid+2,100,1)
    assert a["status"]=="spread_feasible" and a["max_residual"]<1e-8
    assert min(np.diff(np.diff(a["prices"])/np.diff(k)))>=-1e-9
    assert repair(k,mid,mid,100,1)["status"]=="infeasible"
    assert repair(k,a["prices"],a["prices"],100,1)["status"]=="mid_feasible"
    x,w=coordinates(k,100,.25,[.2]*4)
    np.testing.assert_allclose(np.exp(x)*100,k)
    np.testing.assert_allclose(np.sqrt(w/.25),.2)
    for method in ("linear","cubic","pchip"):
        np.testing.assert_allclose(interpolate(k,mid,k,method),mid)
    for target,kw in [([1],{}),(k,{"method":"x"})]:
        with pytest.raises(ValueError):
            interpolate(k,mid,target,**kw)


def test_calendar():
    def check(x,y,flag=True):
        return calendar(x,[.1]*2,[.11]*2,y,[.08]*2,[.09]*2,1,2,same_normalized_martingale=flag)["status"]
    assert check([0,1],[0,.5])=="interval_violation"
    assert check([0,1],[.1,.9])=="interpolated_violation"
    assert check([0,1],[2,3])=="insufficient_evidence"
    assert check([0,1],[0,1],False)=="insufficient_evidence"


@pytest.mark.parametrize("objective",["variance","volatility","price"])
def test_svi(objective):
    x=np.linspace(-.45,.4,15)
    model=SVI(.02,.12,-.4,.03,.18)
    w,d,dd=model.derivatives(x)
    h=1e-4
    up=model.derivatives(x+h)[0]
    down=model.derivatives(x-h)[0]
    np.testing.assert_allclose((up-down)/(2*h),d,atol=2e-8)
    np.testing.assert_allclose((up+down-2*w)/h**2,dd,atol=8e-8)
    assert model.wings()==pytest.approx((.168,.072))
    assert np.all(model.g(x)>0)
    fit=fit_svi(x,w,objective=objective)
    assert fit["converged"] and fit["rmse"]<1e-8


def test_ssvi_and_svd():
    x=np.tile(np.linspace(-.4,.4,9),3)
    t=np.repeat([.01,.04,.09],9)
    model=SSVI(-.4,.65)
    w=model.variance(x,t)
    f=fit_ssvi(x,t,w)
    assert f["converged"] and f["model"].sufficient()
    assert f["model"].scale==pytest.approx(.65,abs=1e-7)
    np.testing.assert_allclose(model.variance(0,t),t)
    j=np.array([[1.,2.],[2.,1.],[1.,1.]])
    sd=np.array([1,2,3])
    np.testing.assert_allclose(stability(j,sd)["covariance"],np.linalg.inv((j/sd[:,None]).T@(j/sd[:,None])))
    assert stability(j[:1])["covariance"] is None
    assert propagation(lambda y:y,[1,2],[.1,.1],4).shape==(4,2)


def test_density_dupire():
    k=np.linspace(35,210,301)
    t=np.linspace(.75,1.25,21)
    r,q,v=.035,.014,.22
    calls=np.array([[black(100*np.exp((r-q)*u),s,u,np.exp(-r*u),v) for s in k] for u in t])
    obs=density(k,calls[10],np.exp(-r))
    theory=norm.pdf((np.log(k[1:-1]/100)-(r-q-v*v/2))/v)/(v*k[1:-1])
    assert max(abs(obs["density"]-theory))<2e-5
    assert .99<obs["mass"]<1.001
    values=local_variance(k,t,calls,[r]*len(t),[q]*len(t))["variance"]
    assert np.max(abs(values[:,(k[1:-1]>80)&(k[1:-1]<120)]-v*v))<5e-5
    assert np.all(local_variance([1,2,4],[1,2,3],np.ones((3,3)),[0]*3,[0]*3)["invalid"])


def test_history_report():
    v=np.arange(24).reshape(4,2,3)/300+.15
    for mode in ("absolute","relative","variance"):
        out=historical_pca([0,1,2,3],v,[-1,0,1],[.5,1],mode)
        assert out["changes"].shape==(3,2,3)
    assert wing_slopes([-1,0,1],[2,1,3])["right_secant"]==2
    report=Report(datetime.now(timezone.utc),"<synthetic>",{},{"wing":np.float64(2),"raw":[np.nan]},v)
    assert json.loads(report.render())["diagnostics"]["raw"][0]=={"undefined":"nan"}
    assert "&lt;synthetic&gt;" in report.render("html")
    assert "input_sha256" in report.render("csv")
    with pytest.raises(ValueError):
        report.render("x")
