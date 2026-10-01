"""The stage cost of the paper: the square of the derivative of the GARL.

    N. N. Deniz, G. M. Sanchez, F. A. Auat Cheein and L. L. Giovanini, "Robust Moving Horizon
    Estimation for Autonomous Agricultural Vehicles With GNSS Outliers Using a Robust Loss
    Function", IEEE Robotics and Automation Letters, 10(10):10815-10820, 2025.
    doi:10.1109/LRA.2025.3606377

Eq. (2), the General Adaptive Robust Loss, with shape parameter ``alpha`` and scale ``c``:

    f(x, alpha, c) = |alpha - 2| / alpha * ( ( (x/c)^2 / |alpha - 2| + 1 )^(alpha/2) - 1 )

Eq. (3), its squared derivative, which is what this estimator minimises:

    psi(x, alpha, c) = (x^2 / c^4) * ( x^2 / (c^2 |alpha - 2|) + 1 )^(alpha - 2)

Both ``alpha`` and ``c`` are arguments, never baked in: the whole point of Eqs. (3) and (7) is
that the choice of alpha is a design decision, and the paper's own field run uses alpha = 1.

Why alpha is confined to [1, 2). The GARL is singular at alpha = 0 and alpha = 2; taking the
derivative removes those, and what is left is a family whose tail is

    psi ~ |x|^(2 alpha - 2)     for large |x|,

so alpha = 1 gives a BOUNDED cost -- it saturates at 1/c^2, an outlier a hundred metres out
costs no more than one thirty metres out -- and alpha -> 2 returns the quadratic. ``c`` sets
where the changeover happens. ``tail_exponent`` and ``saturation`` below make that testable.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

ALPHA_MIN, ALPHA_MAX = 1.0, 2.0
EPS_ALPHA = 1e-6     # keeps |alpha - 2| away from zero when alpha is numerically close to 2


def _backend(x):
    return ca if isinstance(x, (ca.SX, ca.MX, ca.DM)) else np


def check_alpha(alpha: float) -> float:
    """The paper's design space: alpha in [1, 2). Outside it the guarantees do not apply."""
    a = float(alpha)
    if not ALPHA_MIN <= a < ALPHA_MAX:
        raise ValueError(f"alpha must lie in [{ALPHA_MIN}, {ALPHA_MAX}), got {a}")
    return a


def garl(x, alpha, c=1.0):
    """Eq. (2). alpha = 1 is Charbonnier, alpha -> 2 is the quadratic."""
    m = _backend(x)
    a = abs(float(alpha) - 2.0) + EPS_ALPHA
    return (a / float(alpha)) * (((x / c) ** 2 / a + 1.0) ** (float(alpha) / 2.0) - 1.0)


def psi(x, alpha, c=1.0):
    """Eq. (3): the squared derivative of the GARL -- the estimator's stage cost."""
    _backend(x)
    a = abs(float(alpha) - 2.0) + EPS_ALPHA
    return (x ** 2 / c ** 4) * (x ** 2 / (c ** 2 * a) + 1.0) ** (float(alpha) - 2.0)


def psi_symbolic(alpha_sym, c_sym):
    """``psi`` as a CasADi Function of (x, alpha, c), for use inside an NLP.

    alpha and c stay symbolic parameters so a sweep does not rebuild the problem.
    """
    x = ca.MX.sym("x")
    a = ca.fabs(alpha_sym - 2.0) + EPS_ALPHA
    expr = (x ** 2 / c_sym ** 4) * (x ** 2 / (c_sym ** 2 * a) + 1.0) ** (alpha_sym - 2.0)
    return ca.Function("psi", [x, alpha_sym, c_sym], [expr], ["x", "alpha", "c"], ["psi"])


def tail_exponent(alpha: float) -> float:
    """Growth exponent of psi for large residuals: psi ~ |x|^(2 alpha - 2)."""
    return 2.0 * float(alpha) - 2.0


def saturation(alpha: float, c: float = 1.0) -> float:
    """What psi tends to as |x| -> infinity. Finite only at alpha = 1, where it is 1/c^2."""
    return 1.0 / c ** 2 if abs(float(alpha) - 1.0) < 1e-12 else float("inf")


def l2(x, c=1.0):
    """The stage cost being replaced, on the same scale as psi (the alpha -> 2 limit)."""
    return (x / c) ** 2 / c ** 2
