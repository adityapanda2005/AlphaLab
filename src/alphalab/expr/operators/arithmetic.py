"""Element-wise arithmetic and mathematical operators.

These operate point by point on the (date x instrument) panel and never look
across dates or across instruments, so they can be applied to a DataFrame or a
scalar interchangeably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register

Panel = "pd.DataFrame | float"


@register("add", category="arithmetic", n_args=2)
def add(x, y):
    """Element-wise sum."""
    return x + y


@register("subtract", category="arithmetic", n_args=2)
def subtract(x, y):
    """Element-wise difference."""
    return x - y


@register("multiply", category="arithmetic", n_args=2)
def multiply(x, y):
    """Element-wise product."""
    return x * y


@register("divide", category="arithmetic", n_args=2)
def divide(x, y):
    """Element-wise division, returning NaN rather than inf on divide-by-zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x / y
    return _clean_inf(out)


@register("modulo", category="arithmetic", n_args=2)
def modulo(x, y):
    """Element-wise remainder."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x % y
    return _clean_inf(out)


@register("power", category="arithmetic", n_args=2)
def power(x, y):
    """Raise ``x`` to the power ``y``, guarding against complex results."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.power(x, y)
    return _clean_inf(out)


@register("negate", category="arithmetic", n_args=1)
def negate(x):
    """Flip the sign.  Used for the unary minus in formulas."""
    return -x


@register("signed_power", category="arithmetic", n_args=2)
def signed_power(x, y):
    """``sign(x) * |x| ** y`` -- a power that preserves the original sign.

    This is the standard way to dampen or amplify a signal without turning
    negative values into NaN: ``signed_power(x, 0.5)`` square-roots the
    magnitude while leaving shorts short.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sign(x) * np.power(np.abs(x), y)
    return _clean_inf(out)


@register("log", category="arithmetic", n_args=1)
def log(x):
    """Natural logarithm; non-positive inputs become NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
    return _clean_inf(out)


@register("log1p", category="arithmetic", n_args=1)
def log1p(x):
    """``log(1 + x)``, numerically stable for small ``x``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log1p(x)
    return _clean_inf(out)


@register("exp", category="arithmetic", n_args=1)
def exp(x):
    """Exponential."""
    with np.errstate(over="ignore"):
        out = np.exp(x)
    return _clean_inf(out)


@register("sqrt", category="arithmetic", n_args=1)
def sqrt(x):
    """Square root; negative inputs become NaN."""
    with np.errstate(invalid="ignore"):
        out = np.sqrt(x)
    return _clean_inf(out)


@register("abs", category="arithmetic", n_args=1)
def abs_(x):
    """Absolute value."""
    return np.abs(x)


@register("sign", category="arithmetic", n_args=1)
def sign(x):
    """-1, 0 or +1 depending on the sign."""
    return np.sign(x)


@register("inverse", category="arithmetic", n_args=1)
def inverse(x):
    """Reciprocal ``1 / x``, NaN where ``x`` is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / x
    return _clean_inf(out)


@register("min", category="arithmetic", n_args=2)
def min_(x, y):
    """Element-wise minimum of two panels."""
    return _pairwise(x, y, np.minimum)


@register("max", category="arithmetic", n_args=2)
def max_(x, y):
    """Element-wise maximum of two panels."""
    return _pairwise(x, y, np.maximum)


@register("clip", category="arithmetic", n_args=3)
def clip(x, lower, upper):
    """Clamp values into ``[lower, upper]``."""
    return np.clip(x, lower, upper)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _clean_inf(out):
    """Replace +/-inf with NaN.

    Infinities poison every downstream statistic (a single inf makes a whole
    cross-sectional mean inf), so the engine's convention is that an undefined
    value is NaN and NaN alone.
    """
    if isinstance(out, pd.DataFrame):
        return out.replace([np.inf, -np.inf], np.nan)
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    if isinstance(out, np.ndarray):
        return np.where(np.isfinite(out), out, np.nan)
    if isinstance(out, float) and not np.isfinite(out):
        return float("nan")
    return out


def _pairwise(x, y, ufunc):
    """Apply a two-argument numpy ufunc, keeping DataFrame labels intact."""
    if isinstance(x, pd.DataFrame):
        other = y.values if isinstance(y, pd.DataFrame) else y
        return pd.DataFrame(
            ufunc(x.values, other), index=x.index, columns=x.columns
        )
    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(
            ufunc(x, y.values), index=y.index, columns=y.columns
        )
    return ufunc(x, y)
