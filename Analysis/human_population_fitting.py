"""
Fit growth models to world population N(t) over several time ranges.

Within each range, models are fitted to ln X with X = N / N(t0), where t0
is the first year of the range. The figure shows N(t) in millions, one
column per range.

Data: Sojecka and Drozd-Rzoska, Sci. Rep. 14, 9853 (2024), appendix table.

Usage: python3 human_population_fitting.py
"""

import numpy as np
import pandas as pd

from common import (
    P,
    apply_style,
    fit_model,
    make_figure,
    plot_fits,
    plot_residuals,
    print_comparison,
    print_fit,
    save_figure,
    trajectory_spec,
)
from models import (
    exponential_trajectory,
    generalised_TZ_trajectory_ode,
    power_law_trajectory,
)


# ============================================================
# INITIAL GUESSES AND BOUNDS
# ============================================================

# Starting points are built from the mean growth rate of each range
# (ln_slope = ln[(y_end - y_start) / (t_end - t_start)]). Several are
# tried and the lowest-RSS fit is kept.

# Exponential: ln a
BOUNDS_EXPONENTIAL = ([np.log(1e-6)], [np.log(0.2)])

# Generalised TZ: ln lambda0, b, c   (lambda0 = a exp(b))
TZ_STARTS_BC = [
    (-2, 0.8), (-0.5, 0.5), (0.5, 0.8), (2, 0.5),
    (10, 0.1), (50, 0.03), (150, 0.01)
]
BOUNDS_TZ = (
    [np.log(1e-6), -P.HUMAN_BMAX, 1e-4],
    [np.log(0.2), P.HUMAN_BMAX, 10]
)

# Power law: ln a, b
POWER_STARTS_B = [-1, 0, 0.5, 1, 1.5, 2, 3, 5]
BOUNDS_POWER = ([np.log(1e-6), -10.0], [np.log(0.2), 10.0])

FIT_OPTIONS = {
    "ftol": 1e-11,
    "xtol": 1e-11,
    "gtol": 1e-11,
    "max_nfev": 1200,
}


# ============================================================
# MODELS
# ============================================================

def exp_first(p):
    # Report a rather than the fitted ln a
    return [np.exp(p[0]), *p[1:]]


def tz_parameters(p):
    # Report a = lambda0 exp(-b), b, c
    return [np.exp(p[0] - p[1]), p[1], p[2]]


def model_specs(ln_slope, N0):

    # X = N / N0, so the observed X(t0) = 1. X(t0) is either fixed or
    # fitted (P.FIX_INITIAL_CONDITION); a fitted value is reported as N0
    # in millions.

    return {
        "exponential": trajectory_spec(
            lambda t, ln_a, x0: exponential_trajectory(t, np.exp(ln_a), x0),
            1.0,
            p0=[[ln_slope]],
            bounds=BOUNDS_EXPONENTIAL,
            names=["a"],
            transform=exp_first,
            options=FIT_OPTIONS,
            x0_name="N0",
            x0_scale=N0
        ),
        "generalised_TZ": trajectory_spec(
            generalised_TZ_trajectory_ode,
            1.0,
            p0=[[ln_slope, b, c] for b, c in TZ_STARTS_BC],
            bounds=BOUNDS_TZ,
            names=["a", "b", "c"],
            transform=tz_parameters,
            options=FIT_OPTIONS,
            x0_name="N0",
            x0_scale=N0
        ),
        "power_law": trajectory_spec(
            lambda t, ln_a, b, x0: power_law_trajectory(t, np.exp(ln_a), b, x0),
            1.0,
            p0=[[ln_slope, b] for b in POWER_STARTS_B],
            bounds=BOUNDS_POWER,
            names=["a", "b"],
            transform=exp_first,
            options=FIT_OPTIONS,
            x0_name="N0",
            x0_scale=N0
        ),
    }


# ============================================================
# DATA
# ============================================================

data = pd.read_csv(P.HUMAN_DATA)


# ============================================================
# FIT EACH TIME RANGE
# ============================================================

regimes = []

for title, first_year, last_year in P.HUMAN_REGIMES:

    d = data[(data.year >= first_year) & (data.year <= last_year)]

    years = d.year.to_numpy(float)
    N = d.population_millions.to_numpy(float)

    year0 = years[0]
    N0 = N[0]

    t = years - year0
    y = np.log(N / N0)

    ln_slope = np.log(max((y[-1] - y[0]) / (t[-1] - t[0]), 1e-5))

    MODELS = model_specs(ln_slope, N0)

    results = [
        fit_model(model, MODELS[model], t, y)
        for model in P.HUMAN_MODELS
    ]

    # Smooth model curves for plotting
    t_plot = np.linspace(0, t[-1], 500)

    for result in results:
        result["curve"] = N0 * np.exp(
            MODELS[result["model"]]["function"](t_plot, *result["popt"])
        )

    regimes.append({
        "title": title,
        "years": years,
        "N": N,
        "years_plot": year0 + t_plot,
        "results": results,
    })


# ============================================================
# OUTPUT
# ============================================================

for regime in regimes:

    for result in regime["results"]:
        print_fit(result, title=regime["title"])

    print_comparison(regime["results"], title=regime["title"])


# ============================================================
# PLOT
# ============================================================

apply_style()

fig, axes = make_figure(
    ncols=len(regimes),
    panel_width_mm=P.HUMAN_PANEL_WIDTH_MM
)

for column, regime in enumerate(regimes):

    ax_fit, ax_res = axes[:, column]

    years = regime["years"]
    fit_window = (years[0], years[-1])

    plot_fits(
        ax_fit,
        years,
        regime["N"],
        [
            (r["model"], regime["years_plot"], r["curve"])
            for r in regime["results"]
        ],
        ylabel="population $N(t)$ (millions)" if column == 0 else None,
        fit_window=fit_window,
        legend=(column == 0),
        legend_loc=P.HUMAN_LEGEND_LOC
    )

    ax_fit.set_title(regime["title"], fontsize=P.FONT_SIZE)

    plot_residuals(
        ax_res,
        [(r["model"], years, r["residuals"]) for r in regime["results"]],
        xlabel="year (CE)",
        ylabel=r"residuals in $\log_{10} N(t)$" if column == 0 else None,
        fit_window=fit_window
    )

save_figure(fig, P.HUMAN_FIGURE)
