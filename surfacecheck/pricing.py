"""Discounted forward Black pricing in underlying-currency units per option."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, log, pi, sqrt
from scipy.optimize import bisect, brentq
from scipy.special import log_ndtr


def _inputs(f: float, k: float, t: float, d: float, kind: str) -> None:
    if not all(isfinite(x) and x > 0 for x in (f,k,t,d)) or kind not in ("call","put"):
        raise ValueError("Require positive finite F, K, T, D and call or put")


def black(f: float, k: float, t: float, d: float, vol: float, kind: str = "call") -> float:
    _inputs(f,k,t,d,kind)
    if not isfinite(vol) or vol < 0:
        raise ValueError("Expected finite nonnegative volatility")
    intrinsic = d * max((f-k) if kind=="call" else (k-f), 0.)
    s = vol*sqrt(t)
    if s == 0:
        return intrinsic
    if f == k:
        from scipy.special import erf
        return float(d*f*erf(s/sqrt(8)))
    x = -abs(log(f)-log(k))/s
    a = float(log_ndtr(x+s/2))
    b = abs(log(f)-log(k)) + float(log_ndtr(x-s/2))
    if a == float("-inf"):
        return intrinsic
    return intrinsic + d*min(f,k)*exp(a)*(-expm1(min(0.,b-a)))


@dataclass(frozen=True)
class Inversion:
    status: str
    volatility: float | None
    residual: float | None
    vega: float | None


def implied_vol(price: float,f: float,k: float,t: float,d: float,kind: str="call",method: str="brent", price_tol: float=1e-10, maxiter: int=150) -> Inversion:
    _inputs(f,k,t,d,kind)
    if not isfinite(price) or not isfinite(price_tol) or price_tol<=0 or method not in ("brent","bisect","newton") or maxiter<1:
        raise ValueError("Invalid inversion controls or price")
    lo = black(f,k,t,d,0,kind)
    upper = d*(f if kind=="call" else k)
    if price < lo or price > upper:
        return Inversion("infeasible",None,None,None)
    if price == lo or price == upper:
        return Inversion("boundary",0. if price==lo else None,0.,None)
    def error(s: float) -> float:
        return black(f,k,t,d,s/sqrt(t),kind)-price
    def sensitivity(s: float) -> float:
        z=(log(f)-log(k))/s+s/2
        return d*f*exp(-0.5*z*z)/sqrt(2*pi)
    right=1.
    while error(right)<0 and right<128:
        right*=2
    if error(right)<0:
        return Inversion("nonconvergence",None,None,None)
    try:
        if method=="newton":
            left=0.
            root=right/2
            for _ in range(maxiter):
                e=error(root)
                if abs(e)<=min(price_tol,1e-13*min(price-lo,upper-price)):
                    break
                if e>0:
                    right=root
                else:
                    left=root
                slope=sensitivity(root)
                new=root-e/slope if slope else left
                root=new if left<new<right else (left+right)/2
                if right-left<=1e-14:
                    break
            else:
                return Inversion("nonconvergence",None,None,None)
        else:
            solver=brentq if method=="brent" else bisect
            root=solver(error,0.,right,xtol=1e-14,maxiter=maxiter)
    except RuntimeError:
        return Inversion("nonconvergence",None,None,None)
    resid=error(root)
    vega=sensitivity(root)*sqrt(t) if root else 0.
    if abs(resid)>price_tol:
        return Inversion("nonconvergence",None,resid,vega)
    if min(price-lo,upper-price)<=price_tol or vega<=price_tol:
        return Inversion("poor_conditioning",None,resid,vega)
    return Inversion("valid",root/sqrt(t),resid,vega)
