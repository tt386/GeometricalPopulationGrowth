"""
Fitting, statistics, output and plotting helpers shared by every analysis
script. Global settings are read from parameters.py in the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import parameters as P  # noqa: E402


# ============================================================
# FITTING
# ============================================================

def fit_model(name, spec, xdata, ydata):
    """
    Fit one model and return its parameters, errors and statistics.

    spec keys:
        function     model function f(xdata, *params)
        p0           list of starting points; the lowest-RSS fit is kept
        bounds       (lower, upper) bounds on the fitted parameters
        names        names of the reported parameters
        transform    optional map from fitted to reported parameters;
                     errors are propagated with the delta method
        options      optional keyword arguments passed to curve_fit
    """

    best = None

    for p0 in spec["p0"]:

        try:
            popt, pcov = curve_fit(
                spec["function"],
                xdata,
                ydata,
                p0=p0,
                bounds=spec["bounds"],
                **spec.get("options", {})
            )
        except (RuntimeError, ValueError):
            continue

        rss = np.sum((ydata - spec["function"](xdata, *popt))**2)

        if np.isfinite(rss) and (best is None or rss < best[0]):
            best = (rss, popt, pcov)

    if best is None:
        raise RuntimeError(f"Fit failed for {name} at every starting point")

    _, popt, pcov = best

    values, errors = reported_parameters(
        spec.get("transform"),
        popt,
        pcov
    )

    prediction = spec["function"](xdata, *popt)

    return {
        "model": name,
        "popt": popt,
        "names": spec["names"],
        "values": values,
        "errors": errors,
        "at_bound": parameters_at_bound(popt, spec["bounds"]),
        "prediction": prediction,
        "residuals": log10_residuals(ydata, prediction),
        "stats": fit_statistics(ydata, prediction, len(popt)),
    }


def trajectory_spec(trajectory, x0, p0, bounds, names, transform=None,
                    options=None, x0_name="X0", x0_scale=1.0):
    """
    Model spec for a trajectory model trajectory(t, *params, x0=...).

    If P.FIX_INITIAL_CONDITION is True, X(t0) = x0 is fixed. Otherwise
    ln X(t0) is appended as a fitted parameter, started at ln x0, and
    reported as x0_name = x0_scale * X(t0).
    """

    if P.FIX_INITIAL_CONDITION:
        return {
            "function": lambda t, *p: trajectory(t, *p, x0=x0),
            "p0": p0,
            "bounds": bounds,
            "names": names,
            "transform": transform,
            "options": options or {},
        }

    ln_x0 = np.log(x0)
    width = P.INITIAL_CONDITION_LN_RANGE

    def free_transform(p):
        base = p[:-1] if transform is None else transform(p[:-1])
        return [*base, x0_scale * np.exp(p[-1])]

    return {
        "function": lambda t, *p: trajectory(t, *p[:-1], x0=np.exp(p[-1])),
        "p0": [[*start, ln_x0] for start in p0],
        "bounds": (
            [*np.broadcast_to(bounds[0], len(p0[0])), ln_x0 - width],
            [*np.broadcast_to(bounds[1], len(p0[0])), ln_x0 + width],
        ),
        "names": [*names, x0_name],
        "transform": free_transform,
        "options": options or {},
    }


def reported_parameters(transform, popt, pcov):

    if transform is None:
        return popt, np.sqrt(np.diag(pcov))

    values = np.asarray(transform(popt), dtype=float)

    # Numerical Jacobian of the transform for error propagation
    J = np.empty((len(values), len(popt)))

    for j in range(len(popt)):
        h = 1e-6 * max(abs(popt[j]), 1.0)
        step = np.zeros_like(popt)
        step[j] = h
        J[:, j] = (
            np.asarray(transform(popt + step))
            - np.asarray(transform(popt - step))
        ) / (2 * h)

    cov = J @ pcov @ J.T

    return values, np.sqrt(np.abs(np.diag(cov)))


def parameters_at_bound(popt, bounds, rtol=1e-6):

    lower = np.broadcast_to(bounds[0], popt.shape)
    upper = np.broadcast_to(bounds[1], popt.shape)

    at_bound = []

    for i, (p, lo, hi) in enumerate(zip(popt, lower, upper)):
        for bound in (lo, hi):
            if np.isfinite(bound) and abs(p - bound) <= rtol * max(abs(bound), 1.0):
                at_bound.append(i)

    return at_bound


# ============================================================
# FIT QUALITY
# ============================================================

def fit_statistics(y_data, y_model, k):

    residuals = y_data - y_model

    n = len(y_data)

    RSS = np.sum(residuals**2)

    SST = np.sum(
        (y_data - np.mean(y_data))**2
    )

    return {
        "R2": 1 - RSS / SST,
        "AIC": n * np.log(RSS / n) + 2 * k,
        "BIC": n * np.log(RSS / n) + k * np.log(n),
    }


def log10_residuals(y_data, y_model):

    # y is a natural logarithm; residuals are model minus data in log10
    return (y_model - y_data) / np.log(10)


# ============================================================
# OUTPUT
# ============================================================

def print_fit(result, title=None):

    label = P.MODEL_STYLES[result["model"]]["label"]

    if title is not None:
        label = f"{title}: {label}"

    print("\n============================================================")
    print(label.upper())
    print("============================================================")

    for name, value, error in zip(
        result["names"],
        result["values"],
        result["errors"]
    ):
        print(f"{name} = {value:.6g} +/- {error:.2g}")

    if result["at_bound"]:
        print("(fit reached a parameter bound; errors unreliable)")

    print()
    print(f"R²   = {result['stats']['R2']:.6f}")
    print(f"AIC  = {result['stats']['AIC']:.6f}")
    print(f"BIC  = {result['stats']['BIC']:.6f}")


def print_comparison(results, title=None):

    heading = "MODEL COMPARISON"

    if title is not None:
        heading = f"{heading}: {title}"

    print("\n============================================================")
    print(heading)
    print("============================================================")

    print(
        f"{'Model':<32}"
        f"{'R²':>12}"
        f"{'AIC':>16}"
        f"{'BIC':>16}"
    )

    print("-" * 76)

    for result in results:

        stats = result["stats"]

        print(
            f"{P.MODEL_STYLES[result['model']]['label']:<32}"
            f"{stats['R2']:>12.6f}"
            f"{stats['AIC']:>16.6f}"
            f"{stats['BIC']:>16.6f}"
        )

    print("============================================================\n")


def latex_number(value):

    if not np.isfinite(value):
        return "-"

    if value == 0:
        return "0"

    sf = P.LATEX_SIG_FIGS
    magnitude = abs(value)

    if magnitude < P.LATEX_SCI_BELOW or magnitude >= P.LATEX_SCI_ABOVE:
        mantissa, exponent = f"{value:.{sf - 1}e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"

    decimals = sf - 1 - int(np.floor(np.log10(magnitude)))
    rounded = round(value, decimals)

    if decimals <= 0:
        return f"{int(rounded)}"

    return f"{rounded:.{decimals}f}"


def latex_columns(results):

    columns = list(P.LATEX_COLUMNS)

    # Initial-condition column when X(t0) is fitted
    for name in results[0]["names"]:
        if name in P.LATEX_INITIAL_CONDITION_COLUMNS:
            columns.append(P.LATEX_INITIAL_CONDITION_COLUMNS[name])

    return columns


def latex_cells(result, columns):

    # Parameter cells in column order, then AIC and BIC
    column_of = {
        **P.LATEX_PARAMETER_COLUMNS[result["model"]],
        **P.LATEX_INITIAL_CONDITION_COLUMNS,
    }

    cells = dict.fromkeys(columns, "-")

    for name, value in zip(result["names"], result["values"]):
        cells[column_of[name]] = latex_number(value)

    return [
        *(cells[column] for column in columns),
        latex_number(result["stats"]["AIC"]),
        latex_number(result["stats"]["BIC"]),
    ]


def print_latex_table(results, title=None):

    columns = latex_columns(results)

    rows = results

    if P.LATEX_SORT_BY_AIC:
        rows = sorted(results, key=lambda r: r["stats"]["AIC"])

    header = " & ".join(["Model", *columns, "AIC", "BIC"])
    spec = "|" + " |".join(["c"] * (len(columns) + 3)) + "|"

    if title is not None:
        print(f"% {title}")

    print(rf"\begin{{tabular}}{{{spec}}}")
    print(r"         \hline")
    print(rf"         {header} \\ [0.5ex] ")
    print(r"         \hline\hline")

    for result in rows:

        entries = [
            P.LATEX_MODEL_NAMES[result["model"]],
            *latex_cells(result, columns),
        ]

        print("         " + " & ".join(entries) + r"\\ ")
        print(r"         \hline")

    print(r"    \end{tabular}")
    print()


def print_latex_wide_table(groups):
    r"""
    One table with the groups side by side, e.g. the human-population
    time ranges. groups is a list of (title, results), each fitting the
    same models. Rows follow the model order in results.

    With P.LATEX_WIDE_FULL_WIDTH the tabular is wrapped in table* and
    scaled to \textwidth, spanning both columns of a two-column paper
    (needs \usepackage{graphicx}).
    """

    columns = latex_columns(groups[0][1])
    per_group = len(columns) + 2
    n_columns = 1 + len(groups) * per_group

    spec = "|c||" + "||".join(
        " |".join(["c"] * per_group) for _ in groups
    ) + "|"

    titles = " & ".join(
        rf"\multicolumn{{{per_group}}}{{c|}}{{{title.replace('–', '--')}}}"
        for title, _ in groups
    )

    header = " & ".join(
        ["Model", *([*columns, "AIC", "BIC"] * len(groups))]
    )

    indent = "         "

    if P.LATEX_WIDE_FULL_WIDTH:
        print(r"\begin{table*}")
        print(r"    \centering")
        print(r"    \resizebox{\textwidth}{!}{%")

    print(rf"\begin{{tabular}}{{{spec}}}")
    print(indent + r"\hline")
    print(indent + f" & {titles} \\\\")
    print(indent + rf"\cline{{2-{n_columns}}}")
    print(indent + rf"{header} \\ [0.5ex] ")
    print(indent + r"\hline\hline")

    for i, result in enumerate(groups[0][1]):

        entries = [P.LATEX_MODEL_NAMES[result["model"]]]

        for _, results in groups:
            entries += latex_cells(results[i], columns)

        print(indent + " & ".join(entries) + r"\\ ")
        print(indent + r"\hline")

    print(r"    \end{tabular}")

    if P.LATEX_WIDE_FULL_WIDTH:
        print(r"    }")
        print(r"\end{table*}")

    print()


# ============================================================
# PLOTTING
# ============================================================

def apply_style():

    plt.rcParams["font.family"] = P.FONT_FAMILY
    plt.rcParams["font.size"] = P.FONT_SIZE


def make_figure(ncols=1, panel_width_mm=P.FIG_WIDTH_MM):
    """
    Figure with fit panels on the top row and residual panels below.
    The x axis is shared within each column only.
    """

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(ncols * panel_width_mm / 25.4,
                 P.FIG_HEIGHT_MM / 25.4),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": P.HEIGHT_RATIOS}
    )

    return fig, axes


def plot_fits(ax, t_data, v_data, curves, ylabel,
              fit_window=None, legend=True, legend_loc=P.LEGEND_LOC):
    """
    Top panel: data and fitted models on a logarithmic y axis.
    curves is a list of (model, t, value) tuples.
    """

    ax.scatter(
        t_data,
        v_data,
        label=P.DATA_LABEL,
        s=P.DATA_MARKER_SIZE,
        facecolors="none",
        edgecolors="k",
        alpha=P.DATA_ALPHA
    )

    for model, t, value in curves:
        ax.plot(t, value, **P.MODEL_STYLES[model])

    ax.set_yscale("log")

    plot_fit_window(ax, fit_window)

    ax.set_ylabel(ylabel, fontsize=P.FONT_SIZE)

    ax.tick_params(
        axis="both",
        which="both",
        width=P.TICK_WIDTH,
        labelsize=P.FONT_SIZE,
        labelbottom=False
    )

    if legend:
        ax.legend(
            loc=legend_loc,
            fontsize=P.FONT_SIZE,
            prop={"family": P.FONT_FAMILY}
        )


def plot_residuals(ax, curves, xlabel, ylabel, fit_window=None):
    """
    Bottom panel: log10 residuals of each model.
    curves is a list of (model, t, residual) tuples.
    """

    for model, t, residual in curves:
        ax.plot(t, residual, **P.MODEL_STYLES[model])

    # Zero residual
    ax.axhline(
        0,
        color="black",
        linewidth=1,
        alpha=0.5
    )

    plot_fit_window(ax, fit_window)

    ax.set_xlabel(xlabel, fontsize=P.FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=P.FONT_SIZE)

    ax.tick_params(
        axis="both",
        width=P.TICK_WIDTH,
        labelsize=P.FONT_SIZE
    )


def plot_fit_window(ax, fit_window):

    if not P.SHOW_FIT_WINDOW or fit_window is None:
        return

    start, end = fit_window

    # Fit start
    ax.axvline(start, ls="--", alpha=P.FIT_WINDOW_ALPHA)

    # Fit end
    ax.axvline(end, ls=":", alpha=P.FIT_WINDOW_ALPHA)


def save_figure(fig, name):

    P.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    path = P.FIGURE_DIR / f"{name}.pdf"

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=P.DPI)
    plt.close(fig)
