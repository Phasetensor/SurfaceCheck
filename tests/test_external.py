from datetime import date,datetime,timezone
import numpy as np
import pytest
from surfacecheck.pricing import black,implied_vol
from surfacecheck.adapters import forward_inputs,historical_snapshot


def test_quantlib():
    ql=pytest.importorskip("QuantLib")
    for k in [65,95,100,110,155]:
        for kind,qtype in [("call",ql.Option.Call),("put",ql.Option.Put)]:
            expected=ql.blackFormula(qtype,k,103,.32*np.sqrt(.7),.965)
            assert black(103,k,.7,.965,.32,kind)==pytest.approx(expected,abs=5e-12)
            assert implied_vol(expected,103,k,.7,.965,kind).volatility==pytest.approx(.32,abs=1e-10)


def test_integrated_history():
    cm=pytest.importorskip("curvemetric")
    pytest.importorskip("vintagecheck")
    now=datetime(2025,1,1,tzinfo=timezone.utc)
    base=dict(period="2026",available_at="2024-12-31T23:00:00Z")
    rows=[dict(base,series=s,value=v) for s,v in [("call",9),("spot",100),("rate",.03),("yield",.01)]]
    rows.append(dict(base,series="call",value=19,available_at="2025-01-01T00:01:00Z"))
    selected={row["series"]:row for row in historical_snapshot(rows,now)}
    assert float(selected["call"]["value"])==9
    conventions=dict(valuation_date="2025-01-01",currency="USD",day_count="ACT/365F")
    def infer(rate):
        curve=cm.Curve([1],[np.exp(-rate)],conventions=conventions)
        t,d,f=forward_inputs(curve,now.date(),date(2026,1,1),"USD",float(selected["spot"]["value"]),float(selected["yield"]["value"]))
        assert f*d==pytest.approx(100*np.exp(-.01))
        return implied_vol(9,f,100,t,d).volatility
    assert infer(.03)>infer(.031)
    curve=cm.Curve([1],[.97],conventions=conventions)
    with pytest.raises(ValueError):
        forward_inputs(curve,now.date(),date(2026,1,1),"GBP",100,.01)
    with pytest.raises(ValueError):
        forward_inputs(curve,now.date(),now.date(),"USD",100,.01)
