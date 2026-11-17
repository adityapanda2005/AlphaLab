"""Comparison, boolean and conditional operators.

Booleans travel through the engine as floats (1.0 true, 0.0 false, NaN
unknown) so that every value in the pipeline has the same type and the same
NaN semantics.  ``if_else`` is what the ``cond ? a : b`` ternary desugars into.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register


@register("greater", category="logic", n_args=2)
def greater(x, y):
    """``x > y`` as 1.0 / 0.0, NaN where either side is NaN."""
    return _compare(x, y, np.greater)


@register("greater_equal", category="logic", n_args=2)
def greater_equal(x, y):
    """``x >= y`` as 1.0 / 0.0."""
    return _compare(x, y, np.greater_equal)


@register("less", category="logic", n_args=2)
def less(x, y):
    """``x < y`` as 1.0 / 0.0."""
    return _compare(x, y, np.less)


@register("less_equal", category="logic", n_args=2)
def less_equal(x, y):
    """``x <= y`` as 1.0 / 0.0."""
    return _compare(x, y, np.less_equal)


@register("equal", category="logic", n_args=2)
def equal(x, y):
    """``x == y`` as 1.0 / 0.0."""
    return _compare(x, y, np.equal)


@register("not_equal", category="logic", n_args=2)
def not_equal(x, y):
    """``x != y`` as 1.0 / 0.0."""
    return _compare(x, y, np.not_equal)


@register("logical_and", category="logic", n_args=2)
def logical_and(x, y):
    """Both sides non-zero."""
    return _compare(x, y, lambda a, b: np.logical_and(a != 0, b != 0))


@register("logical_or", category="logic", n_args=2)
def logical_or(x, y):
    """Either side non-zero."""
    return _compare(x, y, lambda a, b: np.logical_or(a != 0, b != 0))


@register("logical_not", category="logic", n_args=1)
def logical_not(x):
    """Zero becomes 1.0, non-zero becomes 0.0, NaN stays NaN."""
    values, index, columns = _unwrap(x)
    out = np.where(np.isnan(values), np.nan, (values == 0).astype(float))
    return _wrap(out, index, columns)


@register("if_else", category="logic", n_args=3)
def if_else(condition, if_true, if_false):
    """Pick ``if_true`` where ``condition`` is non-zero, else ``if_false``.

    This is what the ternary ``a > b ? x : y`` compiles to.  Where the
    condition is NaN the result is NaN, because an unknown condition cannot
    select a branch.
    """
    cond_values, index, columns = _unwrap(condition)
    true_values = _broadcast_like(if_true, cond_values)
    false_values = _broadcast_like(if_false, cond_values)

    out = np.where(cond_values != 0, true_values, false_values)
    out = np.where(np.isnan(cond_values), np.nan, out)
    return _wrap(out, index, columns)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _unwrap(x):
    """Split a DataFrame into values plus labels; pass scalars through."""
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float), x.index, x.columns
    return np.asarray(x, dtype=float), None, None


def _wrap(values, index, columns):
    if index is None:
        return values if values.ndim else float(values)
    return pd.DataFrame(values, index=index, columns=columns)


def _broadcast_like(x, reference: np.ndarray) -> np.ndarray:
    values, _, _ = _unwrap(x)
    if values.ndim == 0:
        return np.full(reference.shape, float(values))
    return values


def _compare(x, y, ufunc):
    """Apply a comparison, propagating NaN instead of returning False."""
    left, index, columns = _unwrap(x)
    right, r_index, r_columns = _unwrap(y)

    if index is None and r_index is not None:
        index, columns = r_index, r_columns

    reference = left if left.ndim else right
    left = _broadcast_like(x, reference)
    right = _broadcast_like(y, reference)

    with np.errstate(invalid="ignore"):
        out = ufunc(left, right).astype(float)

    unknown = np.isnan(left) | np.isnan(right)
    out = np.where(unknown, np.nan, out)
    return _wrap(out, index, columns)
