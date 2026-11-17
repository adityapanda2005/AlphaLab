"""How good is a signal?  The measurements that decide whether to keep it.

The central statistic in cross-sectional alpha research is the **information
coefficient** (IC): the correlation, computed across instruments on a single
date, between the signal and the return that followed.  One IC per date gives a
time series, and the question "is this alpha real" becomes the much more
tractable "is the mean of this time series reliably different from zero".

Rank (Spearman) correlation is the default rather than Pearson because a raw
signal's distribution is arbitrary -- some are bounded, some have fat tails,
some are dominated by a handful of extreme values -- and the portfolio will be
built from the *ordering* anyway.

Scale intuition, worth keeping in mind before reading any number below: a daily
IC of 0.02 is a perfectly respectable equity signal, 0.05 is strong, and
anything above 0.15 on daily data almost certainly means a bug -- most often
look-ahead, where the signal has been given information it could not have had.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Trading days per year, used to annualise.
TRADING_DAYS = 252

#: Horizons (in days) at which signal decay is measured.
DEFAULT_DECAY_HORIZONS = (1, 2, 3, 5, 10, 20)


@dataclass
class AlphaStats:
    """Everything measured about one candidate alpha."""

    name: str
    ic_mean: float = np.nan
    ic_std: float = np.nan
    ic_ir: float = np.nan           # annualised information ratio of the IC
    ic_t_stat: float = np.nan
    ic_positive_rate: float = np.nan
    ic_skew: float = np.nan
    n_observations: int = 0
    coverage: float = np.nan        # average instruments scored per date
    autocorrelation: float = np.nan  # 1-day signal autocorrelation
    turnover: float = np.nan        # average daily weight churn, 0 to 2
    quantile_spread: float = np.nan  # top minus bottom bucket, annualised
    decay: Dict[int, float] = field(default_factory=dict)
    family: str = ""

    @property
    def half_life(self) -> float:
        """Roughly how many days until the IC falls to half its one-day value.

        Reported as NaN when the signal does not decay monotonically, which is
        common for noisy alphas and is itself informative.
        """
        if not self.decay or 1 not in self.decay:
            return float("nan")
        start = self.decay[1]
        if not np.isfinite(start) or abs(start) < 1e-12:
            return float("nan")
        for horizon in sorted(self.decay):
            if abs(self.decay[horizon]) <= abs(start) / 2.0:
                return float(horizon)
        return float("nan")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["half_life"] = self.half_life
        out.pop("decay")
        for horizon, value in sorted(self.decay.items()):
            out[f"ic_h{horizon}"] = value
        return out


# ----------------------------------------------------------------------
# Information coefficient
# ----------------------------------------------------------------------
def information_coefficient(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    min_instruments: int = 10,
) -> pd.Series:
    """Per-date cross-sectional correlation between signal and forward return.

    Implemented as a vectorised matrix computation rather than a per-date
    ``corr`` call: with ten years of daily data that is thousands of tiny
    correlations, and looping over them is a hundred times slower than doing
    the arithmetic in one pass.

    Dates where fewer than ``min_instruments`` names have both a signal and a
    return are dropped, because a correlation across four stocks is noise.
    """
    signal, forward_returns = signal.align(forward_returns, join="inner")

    valid = signal.notna() & forward_returns.notna()
    counts = valid.sum(axis=1)

    left = signal.where(valid)
    right = forward_returns.where(valid)

    if method == "spearman":
        # Rank within each date, so the result is a Spearman correlation.
        left = left.rank(axis=1, na_option="keep")
        right = right.rank(axis=1, na_option="keep")
    elif method != "pearson":
        raise ValueError("method must be 'spearman' or 'pearson'")

    left = left.sub(left.mean(axis=1), axis=0)
    right = right.sub(right.mean(axis=1), axis=0)

    numerator = (left * right).sum(axis=1, min_count=1)
    denominator = np.sqrt(
        (left**2).sum(axis=1, min_count=1)
        * (right**2).sum(axis=1, min_count=1)
    )

    ic = numerator / denominator.replace(0.0, np.nan)
    ic = ic.where(counts >= min_instruments)
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def ic_summary(ic: pd.Series) -> dict:
    """Mean, volatility, annualised IR and t-statistic of an IC series.

    The t-statistic treats daily ICs as independent, which is approximately
    true for a signal with low autocorrelation and optimistic for a slow one.
    It is a screening tool, not a p-value to be taken literally -- and the
    whole point of the deflated Sharpe ratio later in the pipeline is that
    these per-alpha statistics cannot be trusted once hundreds of alphas have
    been examined.
    """
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "ic_t_stat": float("nan"),
            "ic_positive_rate": float("nan"),
            "ic_skew": float("nan"),
            "n_observations": n,
        }

    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std * np.sqrt(TRADING_DAYS) if std > 0 else float("nan"),
        "ic_t_stat": mean / std * np.sqrt(n) if std > 0 else float("nan"),
        "ic_positive_rate": float((ic > 0).mean()),
        "ic_skew": float(ic.skew()),
        "n_observations": n,
    }


def ic_decay(
    signal: pd.DataFrame,
    panel,
    horizons: Sequence[int] = DEFAULT_DECAY_HORIZONS,
    **kwargs,
) -> Dict[int, float]:
    """Mean IC against returns measured over increasingly distant horizons.

    How fast a signal decays decides how fast it must be traded, and therefore
    how much of it survives transaction costs.  A signal whose IC is gone by
    day two has to be traded daily; one that persists for a month can be held.
    """
    out: Dict[int, float] = {}
    for horizon in horizons:
        forward = panel.forward_returns(horizon)
        ic = information_coefficient(signal, forward, **kwargs)
        out[int(horizon)] = float(ic.mean()) if len(ic) else float("nan")
    return out


# ----------------------------------------------------------------------
# Trading characteristics
# ----------------------------------------------------------------------
def to_weights(
    signal: pd.DataFrame, *, mask: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Turn a raw signal into dollar-neutral, unit-gross weights.

    Demean so longs and shorts cancel, then scale so the absolute weights sum
    to one.  This is the simplest honest mapping from signal to portfolio, and
    it makes turnover and quantile statistics comparable across alphas whose
    raw scales differ by orders of magnitude.
    """
    if mask is not None:
        signal = signal.where(mask)
    centred = signal.sub(signal.mean(axis=1), axis=0)
    gross = centred.abs().sum(axis=1)
    weights = centred.div(gross.replace(0.0, np.nan), axis=0)
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def turnover(weights: pd.DataFrame) -> float:
    """Average daily sum of absolute weight changes.

    Zero means the book never moves; two means it is fully rebuilt every day
    (every long becomes a short).  This is the quantity transaction costs are
    charged on, so it is the number that decides whether a fast signal is
    tradeable.
    """
    if len(weights) < 2:
        return float("nan")
    changes = weights.diff().abs().sum(axis=1)
    return float(changes.iloc[1:].mean())


def autocorrelation(signal: pd.DataFrame, lag: int = 1) -> float:
    """Average cross-sectional correlation between the signal and its own lag.

    A cheap proxy for turnover that needs no portfolio construction: a signal
    correlated 0.95 with yesterday barely trades, one correlated 0.1 churns.
    """
    lagged = signal.shift(lag)
    valid = signal.notna() & lagged.notna()

    left = signal.where(valid).rank(axis=1, na_option="keep")
    right = lagged.where(valid).rank(axis=1, na_option="keep")
    left = left.sub(left.mean(axis=1), axis=0)
    right = right.sub(right.mean(axis=1), axis=0)

    numerator = (left * right).sum(axis=1, min_count=1)
    denominator = np.sqrt(
        (left**2).sum(axis=1, min_count=1)
        * (right**2).sum(axis=1, min_count=1)
    )
    series = (numerator / denominator.replace(0.0, np.nan)).dropna()
    return float(series.mean()) if len(series) else float("nan")


def quantile_returns(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    buckets: int = 5,
    mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Mean forward return of each signal bucket, per date.

    The bucket profile is the sanity check that a headline IC cannot give you:
    a signal with a good IC but a non-monotonic bucket profile is usually
    driven by its extremes, and will behave unpredictably once position limits
    stop you from holding those extremes.
    """
    if mask is not None:
        signal = signal.where(mask)
    signal, forward_returns = signal.align(forward_returns, join="inner")

    valid = signal.notna() & forward_returns.notna()
    ranked = signal.where(valid).rank(axis=1, pct=True, na_option="keep")
    labels = np.minimum(np.floor(ranked * buckets), buckets - 1)

    out = {}
    for bucket in range(buckets):
        in_bucket = labels == bucket
        out[bucket] = forward_returns.where(in_bucket & valid).mean(axis=1)
    return pd.DataFrame(out)


def quantile_spread(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    buckets: int = 5,
    mask: Optional[pd.DataFrame] = None,
    annualise: bool = True,
) -> float:
    """Mean return of the top bucket minus the bottom bucket."""
    profile = quantile_returns(
        signal, forward_returns, buckets=buckets, mask=mask
    )
    if profile.empty:
        return float("nan")
    spread = (profile[buckets - 1] - profile[0]).dropna()
    if not len(spread):
        return float("nan")
    daily = float(spread.mean())
    return daily * TRADING_DAYS if annualise else daily


# ----------------------------------------------------------------------
# One alpha, end to end
# ----------------------------------------------------------------------
def evaluate_alpha(
    name: str,
    signal: pd.DataFrame,
    panel,
    *,
    forward_returns: Optional[pd.DataFrame] = None,
    mask: Optional[pd.DataFrame] = None,
    decay_horizons: Sequence[int] = DEFAULT_DECAY_HORIZONS,
    family: str = "",
) -> AlphaStats:
    """Measure one alpha against one panel and return its full statistics."""
    if mask is not None:
        signal = signal.where(mask)
    if forward_returns is None:
        forward_returns = panel.forward_returns(1)

    ic = information_coefficient(signal, forward_returns)
    stats = AlphaStats(name=name, family=family, **ic_summary(ic))

    stats.coverage = float(signal.notna().sum(axis=1).mean())
    stats.autocorrelation = autocorrelation(signal)
    stats.turnover = turnover(to_weights(signal, mask=mask))
    stats.quantile_spread = quantile_spread(
        signal, forward_returns, mask=mask
    )
    stats.decay = ic_decay(signal, panel, decay_horizons)
    return stats


def evaluate_library(
    signals: Dict[str, pd.DataFrame],
    panel,
    *,
    mask: Optional[pd.DataFrame] = None,
    families: Optional[Dict[str, str]] = None,
    decay_horizons: Sequence[int] = (),
    progress: bool = False,
) -> pd.DataFrame:
    """Measure every alpha and return a tidy table, best first.

    ``decay_horizons`` defaults to empty, which skips the decay profile.  That
    is a deliberate split of work: measuring decay at six horizons means six
    more cross-sectional rankings per alpha, and across a full library that
    dominates the runtime.  Decay only changes decisions for alphas that
    survive screening, so the pipeline runs the cheap sweep over everything and
    then computes full decay for the handful that are kept.

    The forward-return panel is computed once per horizon and reused across
    every alpha, which is the difference between this step taking seconds and
    taking minutes.
    """
    families = families or {}
    forward_cache = {h: panel.forward_returns(h) for h in decay_horizons}
    base_forward = forward_cache.get(1, panel.forward_returns(1))

    rows: List[dict] = []
    for position, (name, signal) in enumerate(signals.items(), start=1):
        if progress and position % 25 == 0:
            print(f"  evaluated {position}/{len(signals)} alphas")

        masked = signal.where(mask) if mask is not None else signal
        ic = information_coefficient(masked, base_forward)
        stats = AlphaStats(
            name=name, family=families.get(name, ""), **ic_summary(ic)
        )
        stats.coverage = float(masked.notna().sum(axis=1).mean())
        stats.autocorrelation = autocorrelation(masked)
        stats.turnover = turnover(to_weights(masked))
        stats.quantile_spread = quantile_spread(masked, base_forward)
        stats.decay = {
            int(h): float(
                information_coefficient(masked, forward_cache[h]).mean()
            )
            for h in decay_horizons
        }
        rows.append(stats.to_dict())

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            "ic_ir", ascending=False, key=lambda s: s.abs()
        ).reset_index(drop=True)
    return table
