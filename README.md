# Growth-model fittings

Fits of exponential, power-law and generalised Tamm–Zaccone (T-Z) growth models to four datasets.

## Layout

```
parameters.py   Global settings: models fitted per dataset, model colours and
                labels, font and figure sizes, and data windows
Data/           Input data files
Analysis/       Analysis scripts
  models.py     Model definitions
  common.py     Shared fitting, statistics, printing and plotting code
Figures/        Output PDFs
```

## Running

From any directory:

```
python3 Analysis/speciation_fitting.py
python3 Analysis/human_population_fitting.py
python3 Analysis/arxiv_fitting.py
python3 Analysis/patent_fitting.py
```

Each script prints the fitted parameters with standard errors, and R², AIC and BIC for each model. It writes one PDF to `Figures/`. The PDF has the fits on top and residuals in log10 below.

| Script | Fitted quantity | Models | Figure |
|---|---|---|---|
| `speciation_fitting.py` | ln X(t), number of species | exponential, generalised T-Z, power law | X vs time (My) |
| `human_population_fitting.py` | ln N(t)/N(t0), per time range | exponential, generalised T-Z, power law | N vs year, one column per range |
| `arxiv_fitting.py` | ln(dx/dt) vs x | power law, generalised T-Z, non-per-capita generalised T-Z | dx/dt vs year |
| `patent_fitting.py` | ln(dx/dt) vs x | power law, generalised T-Z, non-per-capita generalised T-Z | dx/dt vs year |

Models, with the same parameter names in every script:

| Model | Equation |
|---|---|
| exponential | dx/dt = a x |
| power law | dx/dt = a x^b |
| generalised T-Z | dx/dt = a x exp(b x^c) |
| non-per-capita generalised T-Z | dx/dt = a exp(b x^c) |

The trajectory fits (speciation and human population) fix X(t0) to the value observed at the start of the fit window by default. Set `FIX_INITIAL_CONDITION = False` in `parameters.py` to fit X(t0) as an extra parameter instead. It is then counted in AIC and BIC and reported as X0 (species) or N0 (millions). The arXiv and patent fits use the observed x directly, so they have no initial condition.

Units of the parameters:

- Speciation: x is the number of species divided by the number at t = 0, and t is in My.
- Human population: x is N/N(t0) for each range, and t is in years.
- arXiv: x is cumulative submissions in units of 10^6, and dx/dt is in submissions per month.
- Patents: x is cumulative applications in units of 10^6, and dx/dt is in applications per year.

## Data

- `speciation_data.csv`: time (My) and number of species.
- `human_population_appendix_data.csv`: world population in millions. It is transcribed from the appendix table of A. A. Sojecka and A. Drozd-Rzoska, *Sci. Rep.* **14**, 9853 (2024), doi:10.1038/s41598-024-60589-3. See `README_SOURCE.txt`.
- `arxiv_data.csv`: monthly arXiv submissions.
- `patent_data.csv`: resident patent applications per year, from the World Bank indicator IP.PAT.RESD.
