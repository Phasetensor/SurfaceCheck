"""Raw SVI and a restricted globally certifiable SSVI family."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from .grid import axis, vector
from .pricing import black


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    eta: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.a, self.b, self.rho, self.m, self.eta]).all()
            or self.b < 0
            or abs(self.rho) >= 1
            or self.eta <= 0
            or self.a + self.b * self.eta * sqrt(1 - self.rho**2) <= 0
        ):
            raise ValueError("Invalid raw SVI parameters")

    def derivatives(self, k: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        x = vector(k) - self.m
        r = np.hypot(x, self.eta)
        return (
            self.a + self.b * (self.rho * x + r),
            self.b * (self.rho + x / r),
            self.b * self.eta**2 / r**3,
        )

    def g(self, k: ArrayLike) -> NDArray:
        x = vector(k)
        w, d, dd = self.derivatives(x)
        return (1 - x * d / (2 * w)) ** 2 - d * d / 4 * (1 / w + 0.25) + dd / 2

    def wings(self) -> tuple[float, float]:
        return self.b * (1 - self.rho), self.b * (1 + self.rho)


def fit_svi(
    k: ArrayLike,
    w: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    objective: str = "variance",
    years: float = 1.0,
    forward: float = 1.0,
    discount: float = 1.0,
) -> dict:
    x, y = axis(k, 5), vector(w)
    wt = np.ones_like(x) if weights is None else vector(weights)
    if (
        y.shape != x.shape
        or wt.shape != x.shape
        or np.any(y <= 0)
        or np.any(wt <= 0)
        or objective not in ("variance", "volatility", "price")
        or not np.isfinite([years, forward, discount]).all()
        or min(years, forward, discount) <= 0
    ):
        raise ValueError("Invalid calibration inputs")

    def raw(p: NDArray) -> SVI:
        floor, b, rho, m, eta = p
        return SVI(floor - b * eta * sqrt(1 - rho * rho), b, rho, m, eta)

    def transform(v: NDArray) -> NDArray:
        if objective == "variance":
            return v
        sigma = np.sqrt(v / years)
        if objective == "volatility":
            return sigma
        return np.array(
            [
                black(forward, forward * np.exp(z), years, discount, s)
                for z, s in zip(x, sigma)
            ]
        )

    target = transform(y)

    def residual(p: NDArray) -> NDArray:
        return np.sqrt(wt) * (transform(raw(p).derivatives(x)[0]) - target)

    fits = [
        least_squares(
            residual,
            [min(y), 0.1, r, x[np.argmin(y)], 0.15],
            bounds=(
                [1e-14, 0, -0.999999, -np.inf, 1e-7],
                [np.inf, np.inf, 0.999999, np.inf, np.inf],
            ),
            max_nfev=2000,
            gtol=1e-13,
            ftol=1e-13,
            xtol=1e-13,
        )
        for r in (-0.6, 0, 0.6)
    ]
    fit = min(fits, key=lambda f: (not f.success, f.cost))
    return dict(
        model=raw(fit.x),
        converged=bool(fit.success),
        solver=fit.message,
        residuals=fit.fun,
        rmse=float(np.sqrt(np.mean(fit.fun**2))),
        jacobian=fit.jac,
        parameter_coordinates=("floor", "b", "rho", "m", "eta"),
        objective=objective,
    )


@dataclass(frozen=True)
class SSVI:
    """Constant rho; phi(theta)=scale/sqrt(theta*(1+theta))."""

    rho: float
    scale: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.rho, self.scale]).all()
            or abs(self.rho) >= 1
            or self.scale <= 0
        ):
            raise ValueError("Invalid SSVI parameters")

    def variance(self, k: ArrayLike, theta: ArrayLike) -> NDArray:
        x, t = np.broadcast_arrays(np.asarray(k, float), np.asarray(theta, float))
        if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(t <= 0):
            raise ValueError("Expected finite k and positive theta")
        z = self.scale * x / np.sqrt(t * (1 + t))
        return (
            t / 2 * (1 + self.rho * z + np.hypot(z + self.rho, sqrt(1 - self.rho**2)))
        )

    def sufficient(self) -> bool:
        return (
            self.scale * (1 + abs(self.rho)) < 4
            and self.scale**2 * (1 + abs(self.rho)) <= 4
        )


def fit_ssvi(k: ArrayLike, theta: ArrayLike, w: ArrayLike) -> dict:
    x, t, y = vector(k, 2), vector(theta), vector(w)
    if x.shape != t.shape or x.shape != y.shape or min(t) <= 0 or min(y) <= 0:
        raise ValueError("Invalid observations")

    def model(p: NDArray) -> SSVI:
        r, fraction = p
        return SSVI(r, fraction * 2 / sqrt(1 + abs(r)))

    fit = least_squares(
        lambda p: model(p).variance(x, t) - y,
        [0.0, 0.5],
        bounds=([-0.999999, 1e-8], [0.999999, 1.0]),
        gtol=1e-13,
    )
    return dict(
        model=model(fit.x),
        residuals=fit.fun,
        converged=bool(fit.success),
        jacobian=fit.jac,
        solver=fit.message,
    )


def stability(
    jacobian: ArrayLike, errors: ArrayLike | None = None, rcond: float = 1e-10
) -> dict:
    """Known independent error SDs; covariance is withheld under deficient rank."""
    j = np.asarray(jacobian, float)
    if j.ndim != 2 or min(j.shape) < 1 or not np.isfinite(j).all() or not 0 < rcond < 1:
        raise ValueError("Invalid Jacobian or cutoff")
    sd = np.ones(j.shape[0]) if errors is None else vector(errors)
    if len(sd) != j.shape[0] or min(sd) <= 0:
        raise ValueError("Invalid standard deviations")
    _, s, vt = np.linalg.svd(j / sd[:, None], full_matrices=True)
    rank = int(np.sum(s > s[0] * rcond))
    full = rank == j.shape[1]
    cov = (vt.T / s**2) @ vt if full else None
    return dict(
        singular_values=s,
        rank=rank,
        condition=float(s[0] / s[-1]) if full else None,
        weak_directions=vt[rank:],
        covariance=cov,
    )


def propagation(
    refit: Callable[[NDArray], NDArray],
    mid: ArrayLike,
    errors: ArrayLike,
    draws: int = 100,
    seed: int = 0,
) -> NDArray:
    y, sd = vector(mid), vector(errors)
    if y.shape != sd.shape or min(sd) < 0 or draws < 2:
        raise ValueError("Invalid perturbation model")
    sample = np.random.default_rng(seed).normal(y, sd, (draws, len(y)))
    return np.stack([refit(row) for row in sample])
