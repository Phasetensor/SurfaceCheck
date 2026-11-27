import numpy as np
import pytest
from surfacecheck.grid import axis,vector,calendar,repair,coordinates
from surfacecheck.models import SVI,SSVI,fit_svi,fit_ssvi,stability,propagation
from surfacecheck.diagnostics import curvature,density,local_variance,wing_slopes,historical_pca
from surfacecheck.pricing import black

@pytest.mark.parametrize("function,args",[
    (axis,([1,1],)),(vector,([np.nan],)),(coordinates,([1],0,1,[1])),
    (repair,([1,2],[1,2],[0,2],1,1)),(curvature,([1,2,3],[1],)),
    (density,([1,2,3,4],[1]*4,0)),(local_variance,([1,2,3],[1,2,3],[[1]],[0]*3,[0]*3)),
    (wing_slopes,([1,2,3],[1]*3)),(historical_pca,([1,2,3],[1],[1,2],[1,2])),
    (SVI,(-1,1,0,0,.1)),(SSVI,(1,.1)),(SSVI(0,.1).variance,([0],[0])),
    (fit_svi,([1,2,3,4,5],[1]*4)),(fit_ssvi,([1,2],[1],[1,1])),
    (stability,([1],)),(stability,([[1,2]],[-1])),(propagation,(lambda y:y,[1],[-1])),
])
def test_invalid(function,args):
    with pytest.raises(ValueError):
        function(*args)

def test_calendar_controls():
    args=([0,1],[.1,.1],[.1,.1],[0,1],[.2,.2],[.2,.2])
    flag={"same_normalized_martingale":True}
    assert calendar(*args,1,2,**flag)["status"]=="no_violation_detected"
    with pytest.raises(ValueError):
        calendar(*args,2,1,**flag)
    with pytest.raises(ValueError):
        calendar([0,1],[-.1,.1],*args[2:],1,2,**flag)

def test_vanishing_time_value():
    assert black(200,100,1,1,0)==100
    assert black(1,100,1,1,1e-200)==0
