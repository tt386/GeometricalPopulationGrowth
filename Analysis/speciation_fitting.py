"""
Fit growth models to the number of species X(t) through time.

Models are fitted to ln X, where X is the number of species normalised by
the number at t = 0. The figure shows the number of species itself.

Usage: python3 speciation_fitting.py
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
    trajectory_spec,
)
from models import (
    exponential_trajectory,
    generalised_TZ_trajectory,
    power_law_trajectory,
)


# ============================================================
# INITIAL GUESSES AND BOUNDS
# ============================================================

# Exponential: a
P0_EXPONENTIAL = [0.003]
BOUNDS_EXPONENTIAL = ([1e-10], [0.1])

# Generalised TZ: a, b, c
P0_TZ = [0.000345383, 1.11631, 0.147853]
BOUNDS_TZ = ([0, 0, 0], [0.01, 10, 1])

# Power law: a, b
P0_POWER = [0.0015, 0.01]
BOUNDS_POWER = ([1e-10, 1e-6], [0.1, 10])


# ============================================================
# DATA
# ============================================================

data = pd.read_csv(P.SPECIATION_DATA)

t = data["time_My"].to_numpy(float)
species = data["species"].to_numpy(float)

N_initial = species[0]
x = species / N_initial

# Sort by time
order = np.argsort(t, kind="stable")
t, x = t[order], x[order]

# Remove duplicate timesteps, retaining final occurrence
_, ind = np.unique(t[::-1], return_index=True)
ind = np.sort(len(t) - 1 - ind)

t, x = t[ind], x[ind]


# ============================================================
# SAMPLE AT FIXED TIME INTERVALS
# ============================================================

# Only use data from the fit start onwards
mask = t >= P.SPECIATION_FIT_START
t_raw = t[mask]
x_raw = x[mask]

# Ignore the final time unit
t_end = t_raw[-1] - P.SPECIATION_EXCLUDE_FINAL_TIME

# Regular time grid: FIT_START, FIT_START + INTERVAL, ...
t_sample = np.arange(
    P.SPECIATION_FIT_START,
    t_end + 1e-12,
    P.SPECIATION_INTERVAL
)

# Find most recent data point at or before each sampled time
idx = np.searchsorted(t_raw, t_sample, side="right") - 1

# Keep only valid samples
valid = idx >= 0

t = t_sample[valid]
x = x_raw[idx[valid]]

y = np.log(x)


# ============================================================
# FITTING REGION
# ============================================================

start = np.where(t >= P.SPECIATION_FIT_START)[0][0]

t0 = t[start]
x0 = x[start]

t_final = t[-1]

fit_mask = (
    (t >= P.SPECIATION_FIT_START) &
    (t_final - t >= P.SPECIATION_EXCLUDE_FINAL_TIME)
)

t_fit = t[fit_mask] - t0
y_fit = y[fit_mask]

fit_window = (
    P.SPECIATION_FIT_START,
    t_final - P.SPECIATION_EXCLUDE_FINAL_TIME
)


# ============================================================
# MODELS
# ============================================================

# X(t0) is either fixed to x0 or fitted (P.FIX_INITIAL_CONDITION).
# A fitted X(t0) is reported as a number of species.

MODELS = {
    "exponential": trajectory_spec(
        exponential_trajectory,
        x0,
        p0=[P0_EXPONENTIAL],
        bounds=BOUNDS_EXPONENTIAL,
        names=["a"],
        options={"maxfev": 1000000},
        x0_scale=N_initial
    ),
    "generalised_TZ": trajectory_spec(
        generalised_TZ_trajectory,
        x0,
        p0=[P0_TZ],
        bounds=BOUNDS_TZ,
        names=["a", "b", "c"],
        options={"maxfev": 1000000},
        x0_scale=N_initial
    ),
    "power_law": trajectory_spec(
        power_law_trajectory,
        x0,
        p0=[P0_POWER],
        bounds=BOUNDS_POWER,
        names=["a", "b"],
        options={"maxfev": 1000000},
        x0_scale=N_initial
    ),
}


# ============================================================
# FIT
# ============================================================

results = [
    fit_model(model, MODELS[model], t_fit, y_fit)
    for model in P.SPECIATION_MODELS
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

plot_fits(
    ax_fit,
    t,
    N_initial * x,
    [
        (r["model"], t[fit_mask], N_initial * np.exp(r["prediction"]))
        for r in results
    ],
    ylabel="number of species $X(t)$",
    fit_window=fit_window,
    legend_loc=P.SPECIATION_LEGEND_LOC
)

plot_residuals(
    ax_res,
    [(r["model"], t[fit_mask], r["residuals"]) for r in results],
    xlabel="time (My)",
    ylabel=r"residuals in $\log_{10} X(t)$",
    fit_window=fit_window
)

ax_res.set_xticks(P.SPECIATION_XTICKS)

save_figure(fig, P.SPECIATION_FIGURE)
