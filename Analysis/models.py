"""
Growth models used by the analysis scripts.

Trajectory models return ln X(t) for dX/dt = f(X) with X(0) = x0.
They are fitted to time series of X (speciation, human population).

    exponential                     dX/dt = a X
    power law                       dX/dt = a X^b
    generalised T-Z                 dX/dt = a X exp(b X^c)

Rate models return ln(dx/dt) as a function of X. They are fitted to
growth-rate data (arXiv, patents), with the prefactor fitted as ln a.

    exponential                     dx/dt = a X
    power law                       dx/dt = a X^b
    generalised T-Z                 dx/dt = a X exp(b X^c)
    non-per-capita generalised T-Z  dx/dt = a exp(b X^c)
"""

import numpy as np
from scipy.special import expi
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


# Returned by trajectory models when no finite solution exists
INVALID = 1e10


# ============================================================
# TRAJECTORY MODELS: ln X(t)
# ============================================================

def exponential_trajectory(t, a, x0=1.0):

    return a * np.asarray(t, dtype=float) + np.log(x0)


def power_law_trajectory(t, a, b, x0=1.0):

    # X(t) = [x0^(1-b) + (1-b) a t]^(1/(1-b))

    t = np.asarray(t, dtype=float)

    if abs(b - 1) < 1e-8:
        return exponential_trajectory(t, a, x0)

    q = x0**(1 - b) + (1 - b) * a * t

    # Invalid region / blow-up
    if np.any(q <= 0):
        return np.full_like(t, INVALID)

    return np.log(q) / (1 - b)


def generalised_TZ_trajectory(t, a, b, c, x0=1.0):

    # Closed form for b > 0:
    # Ei(-b X^c) = Ei(-b x0^c) + a c t

    t = np.asarray(t, dtype=float)

    E0 = expi(-b * x0**c)
    target = E0 + a * c * t

    # No finite solution if blow-up occurs
    if np.any(target >= 0):
        return np.full_like(t, INVALID)

    u0 = b * x0**c

    result = []

    for E in target:

        def f(u):
            return expi(-u) - E

        lo = u0
        hi = max(2 * u0, 1.0)

        while f(hi) < 0:
            hi *= 2

        u = brentq(f, lo, hi)

        result.append(
            np.log((u / b)**(1 / c))
        )

    return np.asarray(result)


def generalised_TZ_trajectory_ode(t, log_lambda0, b, c, x0=1.0):

    # Numerical integration of y = ln X with X(0) = x0:
    #
    # dy/dt = lambda0 exp[b (X^c - 1)],  lambda0 = a exp(b)
    #
    # Valid for either sign of b, and better conditioned than fitting a
    # directly when b is large.

    t = np.asarray(t, dtype=float)
    lambda0 = np.exp(log_lambda0)

    def rhs(_, yy):
        xc = np.exp(np.clip(c * yy[0], -700.0, 700.0))
        exponent = np.clip(b * (xc - 1.0), -700.0, 700.0)
        return [lambda0 * np.exp(exponent)]

    solution = solve_ivp(
        rhs,
        (0.0, float(t[-1])),
        [np.log(x0)],
        t_eval=t,
        rtol=2e-8,
        atol=1e-10,
        method="RK45"
    )

    if (not solution.success) or solution.y.shape[1] != len(t):
        return np.full_like(t, 1e8)

    return solution.y[0]


# ============================================================
# RATE MODELS: ln(dx/dt) as a function of X
# ============================================================

def exponential_rate(X, log_a):

    return log_a + np.log(X)


def power_law_rate(X, log_a, b):

    return log_a + b * np.log(X)


def generalised_TZ_rate(X, log_a, b, c):

    return log_a + np.log(X) + b * X**c


def non_per_capita_generalised_TZ_rate(X, log_a, b, c):

    return log_a + b * X**c
