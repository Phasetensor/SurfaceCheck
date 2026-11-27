"""Reproduce four synthetic studies. Run after installing .[validation]."""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from surfacecheck.adapters import historical_snapshot, forward_inputs
from surfacecheck.grid import repair, interpolate
from surfacecheck.pricing import black, implied_vol
from surfacecheck.models import SVI, SSVI, fit_svi, fit_ssvi, stability, propagation
from surfacecheck.surface import stress, extrapolated_wings
from surfacecheck.diagnostics import density, local_variance, historical_pca
from surfacecheck.reporting import native, Report


def run(output: Path, integrated: bool = True) -> dict:
    output.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    rng = np.random.default_rng(47)
    k = np.linspace(-0.4, 0.4, 21)
    terms = np.array([0.25, 0.5, 1.0])
    theta = 0.04 * terms
    family = SSVI(-0.4, 0.65)
    variance = family.variance(k[None, :], theta[:, None])
    vols = np.sqrt(variance / terms[:, None])
    x, y = np.meshgrid(k, terms)

    def plot(name, draw, xlabel, ylabel, projection=None):
        fig = plt.figure(figsize=(6, 4), layout="constrained")
        ax = fig.add_subplot(projection=projection)
        draw(ax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").capitalize())
        fig.savefig(output / (name + ".png"), dpi=130)
        plt.close(fig)

    for name, z in (("implied_volatility", vols), ("total_variance", variance)):

        def draw(ax):
            ax.plot_surface(x, y, z, cmap="viridis")
            ax.set_zlabel(name.replace("_", " "))

        plot(name, draw, "log(K/F)", "years", "3d")

    def smiles(ax):
        for t, row in zip(terms, vols):
            ax.plot(k, row, label=f"{t:g} yr")
        ax.legend()

    plot("smiles", smiles, "log(K/F)", "annual volatility")
    dirty = vols.copy()
    dirty[0, 10] += 0.25
    dirty[-1, 4:9] *= 0.3
    baseline = stress(k, terms, vols, 0)
    shock = stress(k, terms, vols, dirty - vols)
    contaminated = np.array(
        [
            [black(1, np.exp(a), t, 1, s) for a, s in zip(k, row)]
            for t, row in zip(terms, dirty)
        ]
    )
    butterfly = np.diff(np.diff(contaminated, axis=1) / np.diff(np.exp(k)), axis=1)
    gaps = np.diff(dirty**2 * terms[:, None], axis=0)
    for name, z in (
        ("butterfly_slope_differences", butterfly),
        ("calendar_variance_changes", gaps),
    ):

        def heat(ax):
            ax.figure.colorbar(ax.imshow(z, aspect="auto", cmap="coolwarm"), ax=ax)

        plot(name, heat, "strike node index", "maturity interval or slice index")
    plot(
        "stress_scenario",
        lambda ax: ax.plot(k, (dirty - vols).T),
        "log(K/F)",
        "absolute volatility shock",
    )
    correct = fp = fn = unknown = 0
    for i in range(80):
        strikes = np.sort(rng.uniform(60, 145, 8))
        calls = np.array([black(100, s, 0.8, 0.98, 0.24) for s in strikes])
        label = i % 2 == 1
        if label:
            calls[3] = calls[2] + 1
        status = repair(strikes, calls, calls, 100, 0.98)["status"]
        positive = status == "infeasible"
        correct += int(positive and label)
        fp += int(positive and not label)
        fn += int(not positive and label)
        unknown += int(status == "numerical_failure")
    model = SVI(0.02, 0.12, -0.4, 0.03, 0.18)
    clean = model.derivatives(k)[0]
    sd = 0.00025
    noisy = clean + rng.normal(0, sd, len(k))
    fit = fit_svi(k, noisy)
    fit_surface = fit_ssvi(
        x.ravel(), np.broadcast_to(theta[:, None], x.shape).ravel(), variance.ravel()
    )
    risk = stability(fit["jacobian"], np.full(len(k), sd))

    def refit(row):
        f = fit_svi(k, row)
        if not f["converged"]:
            raise ArithmeticError("Monte Carlo calibration failed")
        m = f["model"]
        return np.array([m.a, m.b, m.rho, m.m, m.eta])

    draws = propagation(refit, clean, np.full(len(k), sd), 20, 47)
    plot(
        "svi_residuals",
        lambda ax: ax.plot(k, fit["residuals"], "o"),
        "log(K/F)",
        "total variance residual",
    )
    plot(
        "singular_values",
        lambda ax: ax.semilogy(risk["singular_values"], "o"),
        "singular value index",
        "weighted singular value",
    )
    plot(
        "parameter_dispersion",
        lambda ax: ax.bar(["a", "b", "rho", "m", "eta"], draws.std(axis=0, ddof=1)),
        "raw SVI parameter",
        "Monte Carlo SD (parameter units)",
    )
    strikes = np.linspace(40, 200, 161)
    times = np.linspace(0.8, 1.2, 21)
    prices = np.array([[black(100, s, t, 1, 0.2) for s in strikes] for t in times])
    true = prices[10]
    bad = true.copy()
    bad[60] += 0.4
    fixed = repair(strikes, np.maximum(0, bad - 0.5), bad + 0.5, 100, 1)
    den = density(strikes, true, 1)
    restored = density(strikes, fixed["prices"], 1)
    neg = density(strikes, bad, 1)
    theory = norm.pdf((np.log(den["strikes"] / 100) + 0.02) / 0.2) / (
        0.2 * den["strikes"]
    )
    plot(
        "density",
        lambda ax: ax.plot(
            den["strikes"], np.c_[neg["density"], restored["density"], theory]
        ),
        "strike",
        "density (per currency unit)",
    )
    plot(
        "repair_displacement",
        lambda ax: ax.plot(strikes, fixed["adjustments"]),
        "strike",
        "repaired minus original",
    )
    lv = local_variance(
        strikes, times, prices, np.zeros_like(times), np.zeros_like(times)
    )
    safe = (lv["strikes"] >= 80) & (lv["strikes"] <= 120)
    plot(
        "local_volatility",
        lambda ax: ax.plot(lv["strikes"][safe], np.sqrt(lv["variance"][:, safe]).T),
        "strike",
        "local volatility",
    )
    history = np.array(
        [
            vols + z * np.cos(k)[None, :] + s * k[None, :]
            for z, s in rng.normal(0, 0.002, (30, 2))
        ]
    )
    pca = historical_pca(np.arange(30), history, k, terms)
    plot(
        "historical_pca",
        lambda ax: ax.plot(k, pca["factors"][:2, 1].T),
        "log(K/F)",
        "loading, 0.5 year",
    )
    sparse = fit_svi(k[8:13], clean[8:13] + rng.normal(0, sd, 5))
    tail = {
        rule: extrapolated_wings(k, clean, [-2, 2], 100, 1, 1, rule)["variance"]
        for rule in ("flat_variance", "linear_variance")
    }
    # Convex monotone knots with decreasing slope increments of unequal size.
    knots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    knot_prices = np.array([22.0, 14.0, 7.0, 6.0, 5.5])
    dense = np.linspace(80, 120, 101)
    inter = {
        method: float(
            np.min(
                np.diff(
                    np.diff(interpolate(knots, knot_prices, dense, method))
                    / np.diff(dense)
                )
            )
        )
        for method in ("linear", "pchip", "cubic")
    }
    results = dict(
        synthetic=True,
        seed=47,
        baseline=baseline["smiles"],
        stressed=shock["smiles"],
        calendar_stressed=shock["calendars"],
        detection_cases=80,
        precision=correct / (correct + fp),
        recall=correct / (correct + fn),
        unknown_rate=unknown / 80,
        svi_rmse=fit["rmse"],
        ssvi_rmse=np.sqrt(np.mean(fit_surface["residuals"] ** 2)),
        svi_converged=fit["converged"],
        ssvi_converged=fit_surface["converged"],
        sparse_condition=stability(sparse["jacobian"])["condition"],
        parameter_sd=draws.std(axis=0, ddof=1),
        tail_variances=tail,
        minimum_interpolated_slope_increment=inter,
        density_mass=den["mass"],
        density_mass_error=den["mass"] - 1,
        first_moment_error=den["first_moment"] - 100,
        density_max_point_error=max(abs(den["density"] - theory)),
        negative_nodes_before=len(neg["negative"]),
        minimum_density_after=min(restored["density"]),
        repair_max_residual=fixed["max_residual"],
        repair_max_adjustment=max(abs(fixed["adjustments"])),
        local_vol_max_error=np.max(abs(np.sqrt(lv["variance"][:, safe]) - 0.2)),
    )
    noisy_calls = prices + rng.normal(0, 0.008, prices.shape)
    noisy_lv = local_variance(
        strikes, times, noisy_calls, np.zeros_like(times), np.zeros_like(times)
    )
    results["noise_stress"] = dict(
        invalid_gamma=int(np.sum(noisy_lv["invalid"][:, safe])),
        negative_variance=int(np.sum(noisy_lv["variance"][:, safe] < 0)),
        negative_density_nodes=len(density(strikes, noisy_calls[10], 1)["negative"]),
    )
    independent = 100 * norm.cdf(
        (np.log(100 / strikes) + 0.02) / 0.2
    ) - strikes * norm.cdf((np.log(100 / strikes) - 0.02) / 0.2)
    ivs = [implied_vol(p, 100, s, 1, 1) for p, s in zip(independent, strikes)]
    results["price_max_error"] = max(abs(independent - true))
    results["identified_iv_max_error"] = max(
        abs(o.volatility - 0.2) for o in ivs if o.volatility is not None
    )
    if integrated:
        import curvemetric as cm

        stamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        base = dict(period="2026", available_at=stamp.isoformat())
        rows = [
            dict(base, series=n, value=v)
            for n, v in (("rate", 0.03), ("spot", 100), ("dividend", 0.01), ("call", 9))
        ]
        rows.append(
            dict(base, series="call", value=19, available_at="2025-01-01T00:01:00Z")
        )
        listed = np.arange(80.0, 125.0, 5.0)
        for strike in listed:
            price = black(100 * np.exp(0.02), strike, 1, np.exp(-0.03), 0.22)
            rows.append(dict(base, series=f"chain-{strike:g}", value=price))
        atm = next(r for r in rows if r["series"] == "chain-100")
        rows.append(
            dict(atm, value=atm["value"] + 2, available_at="2025-01-01T00:01:00Z")
        )
        selection = {r["series"]: r for r in historical_snapshot(rows, stamp)}
        future = {r["series"]: r for r in rows}

        def invert(shift, selected):
            curve = cm.Curve(
                [1],
                [np.exp(-float(selection["rate"]["value"]) - shift)],
                conventions=dict(
                    valuation_date=stamp.date().isoformat(),
                    currency="USD",
                    day_count="ACT/365F",
                ),
            )
            t, d, f = forward_inputs(
                curve,
                stamp.date(),
                date(2026, 1, 1),
                "USD",
                float(selection["spot"]["value"]),
                float(selection["dividend"]["value"]),
            )
            mids = np.array(
                [float(selected[f"chain-{strike:g}"]["value"]) for strike in listed]
            )
            iv = np.array(
                [
                    implied_vol(p, f, strike, t, d).volatility
                    for p, strike in zip(mids, listed)
                ]
            )
            fit = fit_svi(np.log(listed / f), t * iv**2)
            return dict(
                single_iv=implied_vol(
                    float(selected["call"]["value"]), f, 100, t, d
                ).volatility,
                atm_iv=iv[4],
                rmse=fit["rmse"],
                slice_status=repair(listed, mids, mids, f, d)["status"],
            )

        results["historical_case"] = dict(
            selected_call=selection["call"],
            excluded_future_price=19,
            eligible=invert(0, selection),
            rate_plus_10bp=invert(0.001, selection),
            contaminated=invert(0, future),
        )
    results.update(
        runtime_seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
    )
    (output / "metrics.json").write_text(
        json.dumps(native(results), indent=2, allow_nan=False)
    )
    report = Report(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        "Synthetic generator, seed 47",
        dict(
            units="decimal volatility",
            coordinates="log(K/F)",
            discounting="deterministic",
        ),
        dict(
            metrics=results,
            model=fit["model"],
            objective=fit["objective"],
            calibration_residuals=fit["residuals"],
            solver=fit["solver"],
            stability=risk,
            repair=fixed,
            density=den,
            local_variance=lv,
            stress=shock,
            limitations=[
                "Synthetic inputs",
                "Finite strike support",
                "Sampled raw SVI diagnostics",
            ],
            evidence={
                "stressed_calendars": "confirmed_violation",
                "baseline_slices": "no_violation_detected_within_checks",
                "global_raw_svi": "insufficient_evidence",
            },
        ),
        dict(
            k=k,
            terms=terms,
            variance=variance,
            noisy_smile=noisy,
            repair_bids=fixed["bids"],
            repair_asks=fixed["asks"],
            historical_inputs=rows if integrated else [],
            seed=47,
        ),
    )
    for fmt in ("json", "csv", "html"):
        (output / f"validation.{fmt}").write_text(report.render(fmt))
    print(json.dumps(native(results)))
    return results


if __name__ == "__main__":
    run(Path("results"))
