"""
Fit growth-rate models dx/dt = f(x) to cumulative patent applications.

x is the cumulative number of patent applications, in units of 10^6 for
the fits, and dx/dt is in patents per year. Models are fitted to
ln(dx/dt). The figure shows dx/dt against time.

Usage: python3 patent_fitting.py
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
BOUNDS_POWER = (-np.inf, np.inf)

# Generalised TZ: ln a, b, c
P0_TZ = [np.log(1e-10), 40, 0.02]
BOUNDS_TZ = ([-100, -1000, 0], [100, 1000, 2])

# Non-per-capita generalised TZ: ln a, b, c
P0_NPC_TZ = [np.log(1e-10), 40, 0.02]
BOUNDS_NPC_TZ = ([-100, -1000, 0], [100, 1000, 2])


# ============================================================
# DATA
# ============================================================

data = pd.read_csv(P.PATENT_DATA)

# No data after the cutoff year
data = data[data["year"] <= P.PATENT_CUTOFF_YEAR]

years = data["year"].to_numpy(int)
total_patents = np.cumsum(data["patent_applications"].to_numpy(float))


# ============================================================
# GROWTH RATE
# ============================================================

# Forward difference: dx/dt at year t is the number of applications in
# year t + 1. The final year has no forward difference.
x = total_patents[:-1]
dxdt = np.diff(total_patents)
t = years[:-1]

mask = (
    (t >= P.PATENT_START_YEAR) &
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
    for model in P.PATENT_MODELS
]


# ============================================================
# OUTPUT
# ============================================================

for result in results:
    print_fit(result)

print_comparison(results)


# ============================================================
# PLOT
# ============================================================

apply_style()

fig, axes = make_figure()
ax_fit, ax_res = axes[:, 0]

plot_fits(
    ax_fit,
    t,
    dxdt,
    [(r["model"], t, np.exp(r["prediction"])) for r in results],
    ylabel=r"$dx/dt$ (patents yr$^{-1}$)",
    fit_window=fit_window,
    legend_loc=P.PATENT_LEGEND_LOC
)

plot_residuals(
    ax_res,
    [(r["model"], t, r["residuals"]) for r in results],
    xlabel="year",
    ylabel=r"residuals in $\log_{10}(dx/dt)$",
    fit_window=fit_window
)

save_figure(fig, P.PATENT_FIGURE)
