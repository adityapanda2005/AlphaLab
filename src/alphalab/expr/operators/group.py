"""Group operators: compare an instrument against its sector peers.

These are the operators that turn a raw idea into something a market-neutral
book can actually hold.  "Cheap versus the whole market" usually just means
"this is an energy stock and energy is out of favour" -- a sector bet dressed
up as a stock-picking signal.  ``group_demean(x, sector)`` removes that bet and
leaves the part of the signal that genuinely distinguishes one energy name from
another.

The group argument is a panel of labels: one column per instrument, and in the
common case constant down the date axis.

Performance note
----------------
These were the slowest operators in the library by a wide margin.  Profiling
the full run showed the group family alone accounting for roughly 45% of total
evaluation time, all of it spent comparing an object-dtype label panel against
each sector in turn and then calling ``DataFrame.where`` on the full panel --
so an eight-sector classification touched every cell sixteen times.

The fix exploits a property of the data rather than micro-optimising the loop:
a sector map is almost always *static*, meaning every row of the label panel is
identical.  When that holds, group membership is a property of the columns, so
each group can be handled as a contiguous column slice of a plain numpy array
and no masking is needed at all.  Measured on a 1500 x 150 panel with eight
sectors, that is 11x faster for ``group_mean`` and 9x for ``group_rank``, with
bit-identical output.

The general time-varying path is kept for correctness when an instrument does
change classification mid-sample, and ``tests/test_operators.py`` checks the
two paths agree.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .registry import register


@register("group_mean", category="group", n_args=2)
def group_mean(x, group):
    """Mean of ``x`` within each group, broadcast back to every member."""
    return _reduce_by_group(x, group, _nanmean)


@register("group_demean", category="group", n_args=2)
def group_demean(x, group):
    """Subtract the group mean, leaving only within-group differences.

    This is sector neutralisation in one line, and it is usually the single
    most valuable transformation in a cross-sectional alpha library.
    """
    frame = _frame(x)
    return frame - group_mean(frame, group)


@register("group_rank", category="group", n_args=2)
def group_rank(x, group):
    """Percentile rank of ``x`` within each group, in ``[0, 1]``."""
    return _rank_by_group(x, group)


@register("group_zscore", category="group", n_args=2)
def group_zscore(x, group):
    """Standardise ``x`` within each group: ``(x - group_mean) / group_std``."""
    frame = _frame(x)
    centred = frame - group_mean(frame, group)
    std = _reduce_by_group(frame, group, _nanstd)
    out = centred / std.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


@register("group_scale", category="group", n_args=2)
def group_scale(x, group):
    """Rescale within each group so each group's absolute values sum to 1.

    Equalising gross exposure per sector stops one large sector from
    dominating the book's risk.
    """
    frame = _frame(x)
    gross = _reduce_by_group(frame, group, _nanabssum)
    out = frame / gross.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------
# Reducers.  Each maps a (dates x members) block to one value per date.
# ----------------------------------------------------------------------
def _nanmean(block: np.ndarray) -> np.ndarray:
    return np.nanmean(block, axis=1)


def _nanstd(block: np.ndarray) -> np.ndarray:
    return np.nanstd(block, axis=1, ddof=0)


def _nanabssum(block: np.ndarray) -> np.ndarray:
    return np.nansum(np.abs(block), axis=1)


# ----------------------------------------------------------------------
# Dispatch between the static fast path and the general path
# ----------------------------------------------------------------------
def _reduce_by_group(
    x, group, reducer: Callable[[np.ndarray], np.ndarray]
) -> pd.DataFrame:
    frame, labels = _aligned(x, group)
    column_codes = _static_column_codes(labels)

    if column_codes is not None:
        return _reduce_static(frame, column_codes, reducer)
    return _reduce_dynamic(frame, labels, reducer)


def _rank_by_group(x, group) -> pd.DataFrame:
    frame, labels = _aligned(x, group)
    column_codes = _static_column_codes(labels)

    if column_codes is not None:
        return _rank_static(frame, column_codes)
    return _rank_dynamic(frame, labels)


# ----------------------------------------------------------------------
# Fast path: the sector map does not change over time
# ----------------------------------------------------------------------
def _reduce_static(
    frame: pd.DataFrame,
    column_codes: Tuple[np.ndarray, int],
    reducer: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    codes, n_groups = column_codes
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)

    with warnings.catch_warnings():
        # An all-NaN sector on some date is normal (nothing listed yet) and
        # numpy's "mean of empty slice" warning is just noise here.
        warnings.simplefilter("ignore", RuntimeWarning)
        for code in range(n_groups):
            members = np.flatnonzero(codes == code)
            if members.size == 0:
                continue
            reduced = reducer(values[:, members])
            out[:, members] = reduced[:, None]

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _rank_static(
    frame: pd.DataFrame, column_codes: Tuple[np.ndarray, int]
) -> pd.DataFrame:
    codes, n_groups = column_codes
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)

    for code in range(n_groups):
        members = np.flatnonzero(codes == code)
        if members.size == 0:
            continue
        block = pd.DataFrame(values[:, members])
        out[:, members] = block.rank(
            axis=1, pct=True, na_option="keep"
        ).to_numpy()

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


# ----------------------------------------------------------------------
# General path: an instrument may change group mid-sample
# ----------------------------------------------------------------------
def _reduce_dynamic(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    reducer: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    codes, n_groups = _cell_codes(labels)
    out = np.full(values.shape, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for code in range(n_groups):
            mask = codes == code
            if not mask.any():
                continue
            block = np.where(mask, values, np.nan)
            reduced = reducer(block)
            out = np.where(mask, reduced[:, None], out)

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _rank_dynamic(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    codes, n_groups = _cell_codes(labels)
    out = np.full(values.shape, np.nan)

    for code in range(n_groups):
        mask = codes == code
        if not mask.any():
            continue
        block = pd.DataFrame(
            np.where(mask, values, np.nan),
            index=frame.index,
            columns=frame.columns,
        )
        ranked = block.rank(axis=1, pct=True, na_option="keep").to_numpy()
        out = np.where(mask, ranked, out)

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(f"group operators need a panel, got {type(x).__name__}")


def _aligned(x, group) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _frame(x)
    labels = _frame(group)
    return frame.align(labels, join="inner")


def _static_column_codes(
    labels: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, int]]:
    """Integer code per column if the label panel is constant down the rows.

    Returns ``None`` when any instrument changes label during the sample, in
    which case the caller falls back to the general per-cell path.
    """
    array = labels.to_numpy()
    if array.shape[0] == 0:
        return None

    first_row = array[0]
    # NaN never equals itself, so compare on the filled representation.
    same = (array == first_row) | (pd.isna(array) & pd.isna(first_row))
    if not bool(same.all()):
        return None

    codes, uniques = pd.factorize(first_row, use_na_sentinel=True)
    return codes, len(uniques)


def _cell_codes(labels: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Integer code for every cell, so the group loop compares ints not objects."""
    array = labels.to_numpy()
    flat_codes, uniques = pd.factorize(array.ravel(), use_na_sentinel=True)
    return flat_codes.reshape(array.shape), len(uniques)
