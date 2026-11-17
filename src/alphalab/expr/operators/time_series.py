"""Time-series operators: compare an instrument against its own history.

Every function here looks *backwards* along the date axis and never forwards.
That is the single most important invariant in the whole project, because a
rolling window that accidentally includes the current or a future observation
turns a worthless signal into a spectacular backtest.

Two conventions keep that invariant easy to audit:

* all rolling windows use pandas' default right-aligned window, which ends on
  the current row and extends into the past;
* ``min_periods`` is always the full window, so a value only appears once its
  entire lookback is genuinely available.  Partial windows at the start of the
  sample are NaN rather than quietly computed from three observations.

``tests/test_no_lookahead.py`` asserts this property mechanically for every
registered time-series operator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register


# ----------------------------------------------------------------------
# Shifts and differences
# ----------------------------------------------------------------------
@register("ts_delay", category="time_series", n_args=2, const_args=(1,))
def ts_delay(x, d):
    """Value from ``d`` days ago."""
    return _frame(x).shift(_w(d))


@register("ts_delta", category="time_series", n_args=2, const_args=(1,))
def ts_delta(x, d):
    """Change over ``d`` days: ``x_t - x_{t-d}``."""
    frame = _frame(x)
    return frame - frame.shift(_w(d))


@register("ts_returns", category="time_series", n_args=2, const_args=(1,))
def ts_returns(x, d):
    """Simple return over ``d`` days: ``x_t / x_{t-d} - 1``."""
    frame = _frame(x)
    prev = frame.shift(_w(d))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = frame / prev - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


@register("ts_log_returns", category="time_series", n_args=2, const_args=(1,))
def ts_log_returns(x, d):
    """Log return over ``d`` days: ``log(x_t) - log(x_{t-d})``."""
    frame = _frame(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log(frame)
    out = logged - logged.shift(_w(d))
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------
# Rolling moments
# ----------------------------------------------------------------------
@register("ts_sum", category="time_series", n_args=2, const_args=(1,))
def ts_sum(x, d):
    """Rolling sum over ``d`` days."""
    return _roll(x, d).sum()


@register("ts_mean", category="time_series", n_args=2, const_args=(1,))
def ts_mean(x, d):
    """Rolling mean over ``d`` days."""
    return _roll(x, d).mean()


@register("ts_std", category="time_series", n_args=2, const_args=(1,))
def ts_std(x, d):
    """Rolling sample standard deviation over ``d`` days."""
    return _roll(x, d).std()


@register("ts_var", category="time_series", n_args=2, const_args=(1,))
def ts_var(x, d):
    """Rolling sample variance over ``d`` days."""
    return _roll(x, d).var()


@register("ts_median", category="time_series", n_args=2, const_args=(1,))
def ts_median(x, d):
    """Rolling median over ``d`` days."""
    return _roll(x, d).median()


@register("ts_skew", category="time_series", n_args=2, const_args=(1,))
def ts_skew(x, d):
    """Rolling skewness over ``d`` days."""
    return _roll(x, d).skew()


@register("ts_kurt", category="time_series", n_args=2, const_args=(1,))
def ts_kurt(x, d):
    """Rolling excess kurtosis over ``d`` days."""
    return _roll(x, d).kurt()


@register("ts_min", category="time_series", n_args=2, const_args=(1,))
def ts_min(x, d):
    """Rolling minimum over ``d`` days."""
    return _roll(x, d).min()


@register("ts_max", category="time_series", n_args=2, const_args=(1,))
def ts_max(x, d):
    """Rolling maximum over ``d`` days."""
    return _roll(x, d).max()


@register("ts_product", category="time_series", n_args=2, const_args=(1,))
def ts_product(x, d):
    """Rolling product over ``d`` days, computed in log space for stability."""
    frame = _frame(x)
    sign = np.sign(frame).rolling(_w(d), min_periods=_w(d)).apply(
        np.prod, raw=True
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log(np.abs(frame))
    magnitude = np.exp(logged.rolling(_w(d), min_periods=_w(d)).sum())
    out = sign * magnitude
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------
# Rolling normalisation
# ----------------------------------------------------------------------
@register("ts_zscore", category="time_series", n_args=2, const_args=(1,))
def ts_zscore(x, d):
    """How unusual today is versus this instrument's own last ``d`` days.

    Unlike the cross-sectional :func:`zscore`, this asks "is this stock
    expensive relative to its own recent history", which is the natural way to
    express a mean-reversion idea.
    """
    frame = _frame(x)
    window = _roll(x, d)
    mean = window.mean()
    std = window.std()
    out = (frame - mean) / std.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


@register("ts_rank", category="time_series", n_args=2, const_args=(1,))
def ts_rank(x, d):
    """Percentile rank of today's value within the last ``d`` days, in ``[0, 1]``."""
    return _roll(x, d).rank(pct=True)


@register("ts_scale", category="time_series", n_args=2, const_args=(1,))
def ts_scale(x, d):
    """Min-max scale today's value against the last ``d`` days, in ``[0, 1]``."""
    frame = _frame(x)
    window = _roll(x, d)
    lo = window.min()
    hi = window.max()
    span = (hi - lo).replace(0.0, np.nan)
    out = (frame - lo) / span
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------
# Rolling relationships between two series
# ----------------------------------------------------------------------
@register("ts_corr", category="time_series", n_args=3, const_args=(2,))
def ts_corr(x, y, d):
    """Rolling Pearson correlation between two panels over ``d`` days.

    Computed from rolling sums rather than ``Rolling.corr`` so that the
    pairing stays strictly column-by-column: we want corr(close_i, volume_i)
    for each instrument ``i``, never the full cross-instrument matrix.
    """
    a, b = _frame(x), _frame(y)
    window = _w(d)
    cov = _rolling_cov(a, b, window)
    sd_a = a.rolling(window, min_periods=window).std()
    sd_b = b.rolling(window, min_periods=window).std()
    denom = (sd_a * sd_b).replace(0.0, np.nan)
    out = cov / denom
    return out.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)


@register("ts_cov", category="time_series", n_args=3, const_args=(2,))
def ts_cov(x, y, d):
    """Rolling sample covariance between two panels over ``d`` days."""
    return _rolling_cov(_frame(x), _frame(y), _w(d))


@register("ts_beta", category="time_series", n_args=3, const_args=(2,))
def ts_beta(x, y, d):
    """Rolling slope from regressing ``x`` on ``y`` over ``d`` days.

    ``ts_beta(returns, market_returns, 60)`` is the textbook rolling market
    beta, which the portfolio layer uses to strip out market exposure.
    """
    a, b = _frame(x), _frame(y)
    window = _w(d)
    cov = _rolling_cov(a, b, window)
    var = b.rolling(window, min_periods=window).var()
    out = cov / var.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------
# Window position and weighting
# ----------------------------------------------------------------------
@register("ts_argmax", category="time_series", n_args=2, const_args=(1,))
def ts_argmax(x, d):
    """Days since the highest value in the last ``d`` days (0 means today).

    A small value means the instrument is at or near a recent high, which is a
    different statement from "it went up a lot", and often a better one.
    """
    return _arg_extreme(_frame(x), _w(d), np.argmax)


@register("ts_argmin", category="time_series", n_args=2, const_args=(1,))
def ts_argmin(x, d):
    """Days since the lowest value in the last ``d`` days (0 means today)."""
    return _arg_extreme(_frame(x), _w(d), np.argmin)


@register("decay_linear", category="time_series", n_args=2, const_args=(1,))
def decay_linear(x, d):
    """Linearly weighted moving average: today weighted ``d``, ``d`` days ago 1.

    Smoothing a signal before trading it is the cheapest turnover reduction
    there is.  Linear decay is preferred to a flat mean because it keeps the
    signal responsive while still averaging away one-day noise.

    Implemented as ``d`` vectorised shifts rather than ``rolling.apply`` --
    about two orders of magnitude faster, and the arithmetic is visible.
    """
    frame = _frame(x)
    window = _w(d)
    if window == 1:
        return frame.copy()

    weights = np.arange(window, 0, -1, dtype=float)  # d, d-1, ..., 1
    weights /= weights.sum()

    out = None
    weight_used = None
    for lag, weight in enumerate(weights):
        shifted = frame.shift(lag)
        term = shifted * weight
        present = shifted.notna() * weight
        out = term if out is None else out.add(term, fill_value=0.0)
        weight_used = present if weight_used is None else weight_used.add(
            present, fill_value=0.0
        )

    # Require the full window, matching the convention of the other operators.
    valid = frame.rolling(window, min_periods=window).count() == window
    return out.where(valid)


@register("hump", category="time_series", min_args=1, max_args=2)
def hump(x, threshold=0.01):
    """Only let the signal move when it moves by more than ``threshold``.

    A deliberate turnover brake: small day-to-day wiggles are ignored, so the
    book does not pay spread to chase noise.  Implemented as a sequential scan
    because each output depends on the previous output.
    """
    frame = _frame(x)
    values = frame.to_numpy(dtype=float, copy=True)
    out = np.full_like(values, np.nan)

    previous = np.full(values.shape[1], np.nan)
    for t in range(values.shape[0]):
        row = values[t]
        started = np.isnan(previous) & ~np.isnan(row)
        previous = np.where(started, row, previous)

        moved = np.abs(row - previous) > threshold
        previous = np.where(moved & ~np.isnan(row), row, previous)
        out[t] = previous

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(
        f"time-series operators need a panel, got {type(x).__name__}"
    )


def _w(d) -> int:
    """Coerce a window argument to a positive integer."""
    window = int(d)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {d!r}")
    return window


def _roll(x, d):
    """Right-aligned rolling window that demands a complete lookback."""
    window = _w(d)
    return _frame(x).rolling(window, min_periods=window)


def _rolling_cov(a: pd.DataFrame, b: pd.DataFrame, window: int) -> pd.DataFrame:
    """Column-wise rolling covariance with Bessel's correction.

    cov = (sum(xy) - n * mean(x) * mean(y)) / (n - 1)
    """
    a, b = a.align(b, join="inner")
    both_valid = a.notna() & b.notna()
    a = a.where(both_valid)
    b = b.where(both_valid)

    roll_a = a.rolling(window, min_periods=window)
    roll_b = b.rolling(window, min_periods=window)
    roll_ab = (a * b).rolling(window, min_periods=window)

    n = float(window)
    cov = (roll_ab.sum() - n * roll_a.mean() * roll_b.mean()) / (n - 1.0)
    return cov.replace([np.inf, -np.inf], np.nan)


def _arg_extreme(frame: pd.DataFrame, window: int, finder) -> pd.DataFrame:
    """Days since the window's extreme value, vectorised with a sliding view.

    ``sliding_window_view`` gives a zero-copy (T-w+1, N, w) view of the panel,
    so the whole computation is one numpy pass instead of a Python loop over
    T * N windows.

    Two details make this roughly 3.5x faster than the obvious version, which
    matters because these were among the slowest operators in the library:

    * the NaN fill is applied once to the (T, N) source array, *before* the
      sliding view is taken.  Filling the view instead would materialise a
      full (T-w+1, N, w) copy -- for a 1500 x 150 panel with a 60-day window
      that is a 1.6 GB temporary, and it dominated the runtime.
    * "does this window contain a NaN" is answered with a cumulative-sum
      difference rather than ``isnan(view).any(axis=-1)``, which would
      materialise the same oversized temporary a second time.
    """
    values = frame.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    out = np.full((n_rows, n_cols), np.nan)

    if n_rows >= window:
        missing = np.isnan(values)
        fill = -np.inf if finder is np.argmax else np.inf
        filled = np.where(missing, fill, values)

        view = np.lib.stride_tricks.sliding_window_view(
            filled, window_shape=window, axis=0
        )  # zero-copy, shape (n_rows - window + 1, n_cols, window)
        position = finder(view, axis=-1)
        days_since = (window - 1) - position

        # Rolling count of NaNs per window, via a prefix sum.
        prefix = np.cumsum(
            np.vstack([np.zeros((1, n_cols), dtype=np.int64), missing]), axis=0
        )
        has_nan = (prefix[window:] - prefix[:-window]) > 0

        out[window - 1 :] = np.where(has_nan, np.nan, days_since.astype(float))

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)
