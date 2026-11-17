"""Cross-sectional operators: compare instruments against each other on a date.

Every function here works along ``axis=1`` of the (date x instrument) panel,
so the value for one stock on one day depends only on the other stocks on that
same day.  That property matters for two reasons:

* it is what makes a signal *relative* -- "cheap compared to its peers today"
  rather than "cheap compared to last year", which is what a market-neutral
  book actually trades;
* it uses no information from other dates, so these operators can never leak
  the future into the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register


@register("rank", category="cross_sectional", n_args=1)
def rank(x):
    """Percentile rank across instruments on each date, in ``[0, 1]``.

    Ranking is the workhorse of cross-sectional alpha research: it throws away
    the magnitude of a raw signal and keeps only the ordering, which makes the
    result robust to outliers and to the signal's unknown scale.
    """
    return _as_frame(x).rank(axis=1, pct=True, na_option="keep")


@register("zscore", category="cross_sectional", n_args=1)
def zscore(x):
    """Standardise across instruments: ``(x - mean) / std`` per date."""
    frame = _as_frame(x)
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0)
    out = frame.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    return out.replace([np.inf, -np.inf], np.nan)


@register("demean", category="cross_sectional", n_args=1)
def demean(x):
    """Subtract the cross-sectional mean, leaving a zero-sum signal per date.

    A zero-sum signal is the first step toward a dollar-neutral book: the longs
    and the shorts cancel, so the portfolio carries no net market exposure.
    """
    frame = _as_frame(x)
    return frame.sub(frame.mean(axis=1), axis=0)


@register("scale", category="cross_sectional", min_args=1, max_args=2)
def scale(x, target=1.0):
    """Rescale each date so the absolute values sum to ``target``.

    This fixes the gross exposure of the book.  Without it, a signal whose
    magnitude drifts over time would silently lever up and down.
    """
    frame = _as_frame(x)
    gross = frame.abs().sum(axis=1)
    out = frame.div(gross.replace(0.0, np.nan), axis=0) * target
    return out.replace([np.inf, -np.inf], np.nan)


@register("winsorize", category="cross_sectional", min_args=1, max_args=2)
def winsorize(x, n_std=4.0):
    """Clip values more than ``n_std`` cross-sectional deviations from the mean.

    Winsorising rather than dropping keeps the instrument in the book but stops
    one broken print from dominating the day's positions.
    """
    frame = _as_frame(x)
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0)
    lower = mean - n_std * std
    upper = mean + n_std * std
    return frame.clip(lower=lower, upper=upper, axis=0)


@register("normalize", category="cross_sectional", n_args=1)
def normalize(x):
    """Map each date onto ``[-0.5, +0.5]`` by percentile rank.

    Equivalent to ``rank(x) - 0.5``; provided as one operator because the
    centred rank is such a common building block.
    """
    return _as_frame(x).rank(axis=1, pct=True, na_option="keep") - 0.5


@register("cs_mean", category="cross_sectional", n_args=1)
def cs_mean(x):
    """Cross-sectional mean, broadcast back to every instrument."""
    frame = _as_frame(x)
    return _broadcast(frame.mean(axis=1), frame)


@register("cs_std", category="cross_sectional", n_args=1)
def cs_std(x):
    """Cross-sectional standard deviation, broadcast back to every instrument."""
    frame = _as_frame(x)
    return _broadcast(frame.std(axis=1, ddof=0), frame)


@register("cs_min", category="cross_sectional", n_args=1)
def cs_min(x):
    """Cross-sectional minimum, broadcast back to every instrument."""
    frame = _as_frame(x)
    return _broadcast(frame.min(axis=1), frame)


@register("cs_max", category="cross_sectional", n_args=1)
def cs_max(x):
    """Cross-sectional maximum, broadcast back to every instrument."""
    frame = _as_frame(x)
    return _broadcast(frame.max(axis=1), frame)


@register("quantile", category="cross_sectional", min_args=1, max_args=2)
def quantile(x, buckets=5.0):
    """Assign each instrument to one of ``buckets`` groups, numbered from 0.

    Bucketing is how a continuous signal becomes a tradeable portfolio: long
    the top bucket, short the bottom one.
    """
    n = max(int(buckets), 1)
    pct = _as_frame(x).rank(axis=1, pct=True, na_option="keep")
    # Multiply by n and floor, but keep the top edge inside the last bucket.
    out = np.minimum(np.floor(pct * n), n - 1)
    return out


@register("truncate", category="cross_sectional", min_args=1, max_args=2)
def truncate(x, max_weight=0.05):
    """Cap any single instrument's share of the book's gross exposure.

    Applied after :func:`scale`, this is a crude but effective concentration
    limit: no one name may exceed ``max_weight`` of total gross.
    """
    frame = _as_frame(x)
    capped = frame.clip(lower=-abs(max_weight), upper=abs(max_weight))
    gross = capped.abs().sum(axis=1)
    return capped.div(gross.replace(0.0, np.nan), axis=0)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(
        "cross-sectional operators need a panel of instruments, "
        f"got {type(x).__name__}"
    )


def _broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """Turn a per-date Series back into a full panel."""
    return pd.DataFrame(
        np.repeat(series.values[:, None], like.shape[1], axis=1),
        index=like.index,
        columns=like.columns,
    )
