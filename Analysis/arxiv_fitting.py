"""
Fit growth-rate models dx/dt = f(x) to cumulative arXiv submissions.

x is the cumulative number of submissions, in units of 10^6 for the fits,
and dx/dt is in submissions per month. Models are fitted to ln(dx/dt).
The figure shows dx/dt against time.

Usage: python3 arxiv_fitting.py
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
    print_latex_table,
    save_figure,
)
from models import (
    exponential_rate,
    generalised_TZ_rate,
    non_per_capita_generalised_TZ_rate,
    power_law_rate,
)


# ============================================================
# INITIAL GUESSES AND BOUNDS
# ============================================================

# Exponential: ln a
P0_EXPONENTIAL = [np.log(0.01)]
BOUNDS_EXPONENTIAL = ([-100], [100])

# Power law: ln a, b
P0_POWER = [np.log(3e5), 0.65]
BOUNDS_POWER = ([-np.inf, -10], [np.inf, 10])

# Generalised TZ: ln a, b, c
P0_TZ = [np.log(1e-10), 40, 0.02]
BOUNDS_TZ = ([-100, 0, -5], [100, 1000, 2])

# Non-per-capita generalised TZ: ln a, b, c
P0_NPC_TZ = [np.log(1e-10), 40, 0.02]
BOUNDS_NPC_TZ = ([-100, 0, 0], [100, 1000, 2])


# ============================================================
# DATA
# ============================================================

data = pd.read_csv(P.ARXIV_DATA)
data["month"] = pd.to_datetime(data["month"], format="%Y-%m")
data["submissions"] = pd.to_numeric(data["submissions"])
data = data.sort_values("month").reset_index(drop=True)

# No data after the cutoff year
data = data[data["month"].dt.year <= P.ARXIV_CUTOFF_YEAR]

if P.ARXIV_DATA_MODE == "monthly":

    submissions = data["submissions"].to_numpy(float)
    time = np.arange(len(submissions), dtype=float)
    calendar = (
        data["month"].dt.year
        + (data["month"].dt.month - 1) / 12
    ).to_numpy(float)
    DT = 1.0

elif P.ARXIV_DATA_MODE == "yearly":

    data["year"] = data["month"].dt.year

    # Discard the first (incomplete) year
    data = data[data["year"] > data["year"].iloc[0]]

    # Discard the final year if incomplete
    months_in_final_year = (data["year"] == data["year"].iloc[-1]).sum()

    if P.ARXIV_DISCARD_INCOMPLETE_FINAL_YEAR and months_in_final_year < 12:
        data = data[data["year"] < data["year"].iloc[-1]]

    yearly = data.groupby("year")["submissions"].sum()

    # Mean monthly submissions in each year
    submissions = yearly.to_numpy(float) / 12.0
    time = np.arange(1, len(submissions) + 1) * 12.0
    calendar = yearly.index.to_numpy(float)
    DT = 12.0

else:
    raise ValueError("ARXIV_DATA_MODE must be 'monthly' or 'yearly'.")

total_submissions = np.cumsum(submissions * DT)


# ============================================================
# GROWTH RATE
# ============================================================

# Forward difference: dx/dt at time t is the rate over the following
# interval. The final time has no forward difference.
x = total_submissions[:-1]
dxdt = np.diff(total_submissions) / np.diff(time)
t = calendar[:-1]

mask = (
    (t >= P.ARXIV_START_YEAR) &
    (x > 0) &
    (dxdt > 0)
)

x = x[mask]
dxdt = dxdt[mask]
t = t[mask]

# Work in millions for numerical conditioning
X = x / 1e6

y = np.log(dxdt)

fit_window = (t[0], t[-1])


# ============================================================
# MODELS
# ============================================================

def exp_first(p):
    # Report a rather than the fitted ln a
    return [np.exp(p[0]), *p[1:]]


MODELS = {
    "exponential": {
        "function": exponential_rate,
        "p0": [P0_EXPONENTIAL],
        "bounds": BOUNDS_EXPONENTIAL,
        "names": ["a"],
        "transform": exp_first,
        "options": {"maxfev": 100000},
    },
    "power_law": {
        "function": power_law_rate,
        "p0": [P0_POWER],
        "bounds": BOUNDS_POWER,
        "names": ["a", "b"],
        "transform": exp_first,
        "options": {"maxfev": 100000},
    },
    "generalised_TZ": {
        "function": generalised_TZ_rate,
        "p0": [P0_TZ],
        "bounds": BOUNDS_TZ,
        "names": ["a", "b", "c"],
        "transform": exp_first,
        "options": {"maxfev": 100000},
    },
    "non_per_capita_generalised_TZ": {
        "function": non_per_capita_generalised_TZ_rate,
        "p0": [P0_NPC_TZ],
        "bounds": BOUNDS_NPC_TZ,
        "names": ["a", "b", "c"],
        "transform": exp_first,
        "options": {"maxfev": 100000},
    },
}


# ============================================================
# FIT
# ============================================================

results = [
    fit_model(model, MODELS[model], X, y)
    for model in P.ARXIV_MODELS
]


# ============================================================
# OUTPUT
# ============================================================

for result in results:
    print_fit(result)

print_comparison(results)
print_latex_table(results)


# ============================================================
# PLOT
# ============================================================

apply_style()

fig, axes = make_figure()
ax_fit, ax_res = axes[:, 0]

ylabel = r"$dx/dt$ (submissions month$^{-1}$)"

if P.ARXIV_DATA_MODE != "monthly":
    ylabel = r"$dx/dt$ (submissions yr$^{-1}$)"


plot_fits(
    ax_fit,
    t,
    dxdt,
    [(r["model"], t, np.exp(r["prediction"])) for r in results],
    ylabel=ylabel,
    fit_window=fit_window,
    legend_loc=P.ARXIV_LEGEND_LOC
)

plot_residuals(
    ax_res,
    [(r["model"], t, r["residuals"]) for r in results],
    xlabel="year",
    ylabel=r"residuals in $\log_{10}(dx/dt)$",
    fit_window=fit_window
)

save_figure(fig, P.ARXIV_FIGURE)
