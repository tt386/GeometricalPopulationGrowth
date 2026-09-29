"""
Global parameters shared by every analysis script in Analysis/.

Edit this file to change which models are fitted, how they are drawn,
figure sizes, font sizes, and the data windows used for each dataset.
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "Data"
FIGURE_DIR = ROOT / "Figures"

SPECIATION_DATA = DATA_DIR / "speciation_data.csv"
HUMAN_DATA = DATA_DIR / "human_population_appendix_data.csv"
ARXIV_DATA = DATA_DIR / "arxiv_data.csv"
PATENT_DATA = DATA_DIR / "patent_data.csv"


# ============================================================
# FIGURE SETTINGS
# ============================================================

FONT_FAMILY = "Arial"
FONT_SIZE = 10

# Size of one column of a figure (fit panel on top, residual panel below).
# The human-population figure has one column per time range, so its total
# width is HUMAN_PANEL_WIDTH_MM multiplied by the number of ranges.
FIG_WIDTH_MM = 80 * 1.5
FIG_HEIGHT_MM = 80 * 1.5
HUMAN_PANEL_WIDTH_MM = FIG_WIDTH_MM

# Height ratio of fit panel to residual panel
HEIGHT_RATIOS = [2, 1]

TICK_WIDTH = 2
DPI = 300

# Data markers
DATA_LABEL = "data"
DATA_MARKER_SIZE = 36
DATA_ALPHA = 0.5

# Legend location (matplotlib loc string); datasets can override below
LEGEND_LOC = "best"

# Vertical lines marking the start and end of the fitting window
SHOW_FIT_WINDOW = True
FIT_WINDOW_ALPHA = 0.5


# ============================================================
# MODELS
# ============================================================

# Plot style for every model. The same model is drawn identically in
# every figure. zorder sets drawing order: the thick generalised T-Z line
# is drawn underneath so that models lying close to it stay visible.
MODEL_STYLES = {
    "exponential": {
        "label": "exponential",
        "color": "red",
        "linestyle": "dashed",
        "linewidth": 1.5,
        "zorder": 3,
    },
    "power_law": {
        "label": "power law",
        "color": "orange",
        "linestyle": "dashdot",
        "linewidth": 1.5,
        "zorder": 3,
    },
    "generalised_TZ": {
        "label": "replicative geometric",
        "color": "cyan",
        "linestyle": "solid",
        "linewidth": 2,
        "zorder": 2,
    },
    "non_per_capita_generalised_TZ": {
        "label": "direct geometric",
        "color": "purple",
        "linestyle": "dotted",
        "linewidth": 2,
        "zorder": 3,
    },
}

# Initial condition of the trajectory fits (speciation, human population).
# True:  X(t0) is fixed to the observed value at the start of the fit.
# False: X(t0) is fitted as an extra parameter, searched within a factor
#        exp(INITIAL_CONDITION_LN_RANGE) of the observed value, and counted
#        in AIC and BIC.
# The arXiv and patent fits relate dx/dt to the observed x directly and do
# not integrate a trajectory, so they have no initial condition.
FIX_INITIAL_CONDITION = True
INITIAL_CONDITION_LN_RANGE = 1.0

# Models fitted to each dataset (fitted, printed and plotted in this order)
SPECIATION_MODELS = ["exponential", "generalised_TZ", "power_law"]
HUMAN_MODELS = ["exponential", "generalised_TZ", "power_law"]
ARXIV_MODELS = ["power_law", "generalised_TZ", "non_per_capita_generalised_TZ"]
PATENT_MODELS = ["power_law", "generalised_TZ", "non_per_capita_generalised_TZ"]


# ============================================================
# LATEX TABLE OUTPUT
# ============================================================

# Each script prints a LaTeX tabular of fitted parameters, AIC and BIC.

# Model names in the table
LATEX_MODEL_NAMES = {
    "exponential": "Exponential",
    "power_law": "Power-Law",
    "generalised_TZ": "Generalised T-Z",
    "non_per_capita_generalised_TZ": "Non-per-capita generalised T-Z",
}

# Parameter columns, and the column each model parameter is placed in.
# The power-law exponent b shares the gamma column with the T-Z exponent c.
LATEX_COLUMNS = [r"$\alpha$", r"$\beta$", r"$\gamma$"]
LATEX_PARAMETER_COLUMNS = {
    "exponential": {"a": r"$\alpha$"},
    "power_law": {"a": r"$\alpha$", "b": r"$\gamma$"},
    "generalised_TZ": {"a": r"$\alpha$", "b": r"$\beta$", "c": r"$\gamma$"},
    "non_per_capita_generalised_TZ": {
        "a": r"$\alpha$", "b": r"$\beta$", "c": r"$\gamma$"
    },
}

# Extra column added when FIX_INITIAL_CONDITION is False
LATEX_INITIAL_CONDITION_COLUMNS = {"X0": r"$X_0$", "N0": r"$N_0$"}

LATEX_SIG_FIGS = 2

# Numbers with magnitude below / at or above these are written as
# $m\times10^{e}$
LATEX_SCI_BELOW = 1e-2
LATEX_SCI_ABOVE = 1e4

# True: rows ordered by AIC (best first). False: order of the model list.
# The human-population table puts the time ranges side by side, so its
# rows always follow HUMAN_MODELS.
LATEX_SORT_BY_AIC = True

# Wrap the human-population table in table* scaled to \textwidth so it
# spans both columns (needs \usepackage{graphicx})
LATEX_WIDE_FULL_WIDTH = True

# ============================================================
# SPECIATION SETTINGS
# ============================================================

SPECIATION_FIT_START = 1000          # My
SPECIATION_INTERVAL = 10             # Sampling interval (My)
SPECIATION_EXCLUDE_FINAL_TIME = 1.0  # Ignore final 1 My
SPECIATION_XTICKS = [1000, 1200, 1400, 1600, 1800, 2000]
SPECIATION_LEGEND_LOC = LEGEND_LOC
SPECIATION_FIGURE = "speciation_fits_and_residuals"


# ============================================================
# HUMAN POPULATION SETTINGS
# ============================================================

# (panel title, first year, last year)
# The appendix data contain no observations exactly at 1780 or 1929, so
# the fits start at the first observed points in those windows.
HUMAN_REGIMES = [
    ("CEF 1790–1948", 1790, 1948),
    ("CEF 1930–1978", 1930, 1978),
    ("SEF 1970–2023", 1970, 2023),
]

# Bound on |b| for the generalised T-Z fit. If b reaches this bound with
# c -> 0 the model is approaching its power-law limit.
HUMAN_BMAX = 200.0

HUMAN_LEGEND_LOC = LEGEND_LOC
HUMAN_FIGURE = "human_population_fits_and_residuals"


# ============================================================
# ARXIV SETTINGS
# ============================================================

ARXIV_DATA_MODE = "yearly"    # "monthly" or "yearly"

# First year of growth-rate points in the fit, and the last year of data
# used (no data after the cutoff enter the fit, including through the
# forward difference dx/dt). 2019 excludes the COVID period.
ARXIV_START_YEAR = 1992
ARXIV_CUTOFF_YEAR = 2019

# The yearly mode discards the first (incomplete) year of data. This also
# discards the final year if it does not contain 12 months of data.
ARXIV_DISCARD_INCOMPLETE_FINAL_YEAR = True

ARXIV_LEGEND_LOC = "lower right"
ARXIV_FIGURE = "arxiv_fits_and_residuals"


# ============================================================
# PATENT SETTINGS
# ============================================================

# Data from:
# https://data.worldbank.org/indicator/IP.PAT.RESD?end=2021&start=1980&view=chart

# First year of growth-rate points in the fit, and the last year of data
# used (no data after the cutoff enter the fit, including through the
# forward difference dx/dt). 2019 excludes the COVID period.
PATENT_START_YEAR = 1984
PATENT_CUTOFF_YEAR = 2019

PATENT_LEGEND_LOC = "upper left"
PATENT_FIGURE = "patent_fits_and_residuals"
