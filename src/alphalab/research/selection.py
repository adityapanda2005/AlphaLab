"""Choosing which alphas to keep: screening, then de-duplication.

Two filters run in sequence, and they reject for different reasons.

**Screening** drops alphas that are not good enough on their own: weak IC, too
little history, or turnover so high that no realistic cost assumption leaves
anything behind.

**Correlation pruning** drops alphas that are good but *redundant*.  This is
the step people skip, and skipping it quietly wrecks everything downstream.
A library built by sweeping windows contains near-duplicates by construction:
``rev_simple_5`` and ``rev_simple_10`` are nearly the same signal.  Feed thirty
copies of one idea into a combiner and three things go wrong: the model's
effective number of independent inputs is a fraction of what it appears to be,
regularisation gets spread across duplicates instead of selecting among ideas,
and the apparent diversification of the final portfolio is an illusion.

The pruning rule is greedy and deliberately simple: sort by standalone quality,
walk down the list, and keep an alpha only if it is not too correlated with
anything already kept.  Greedy selection is not optimal -- choosing the best
subset is combinatorial -- but it is predictable, explainable in one sentence,
and robust to the noise in the correlation estimates themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScreenConfig:
    """Thresholds for the standalone quality screen.

    The defaults are loose on purpose.  A tight screen applied to the same
    sample that produced the statistics is itself a form of overfitting: the
    survivors are the ones that got lucky.  The job here is to remove the
    clearly unusable, and to let the walk-forward model and the deflated
    Sharpe ratio handle the harder question of what is real.
    """

    min_abs_ic_t_stat: float = 2.0
    min_observations: int = 250
    max_turnover: float = 2.0
    min_coverage: float = 20.0
    max_abs_ic_mean: float = 0.25  # above this, suspect look-ahead, not alpha


@dataclass
class SelectionResult:
    """What survived, what did not, and why."""

    kept: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def rejection_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for reason in self.rejected.values():
            key = reason.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        return counts

    def summary(self) -> str:
        reasons = ", ".join(
            f"{reason} {count}"
            for reason, count in sorted(
                self.rejection_counts().items(), key=lambda kv: -kv[1]
            )
        )
        return (
            f"kept {self.n_kept}, rejected {self.n_rejected}"
            + (f" ({reasons})" if reasons else "")
        )


def screen_alphas(
    stats: pd.DataFrame, config: Optional[ScreenConfig] = None
) -> SelectionResult:
    """Apply the standalone quality thresholds, recording why each alpha fell."""
    config = config or ScreenConfig()
    result = SelectionResult()

    for row in stats.itertuples():
        name = row.name
        reason = _rejection_reason(row, config)
        if reason is None:
            result.kept.append(name)
        else:
            result.rejected[name] = reason

    return result


def _rejection_reason(row, config: ScreenConfig) -> Optional[str]:
    t_stat = getattr(row, "ic_t_stat", np.nan)
    ic_mean = getattr(row, "ic_mean", np.nan)

    if not np.isfinite(t_stat):
        return "degenerate: IC could not be computed"
    if getattr(row, "n_observations", 0) < config.min_observations:
        return (
            f"short history: {getattr(row, 'n_observations', 0)} observations "
            f"< {config.min_observations}"
        )
    if getattr(row, "coverage", np.nan) < config.min_coverage:
        return f"thin coverage: {getattr(row, 'coverage', float('nan')):.1f} names"
    if abs(ic_mean) > config.max_abs_ic_mean:
        # Not a compliment.  An IC this large on daily data is a bug report.
        return (
            f"implausible: |IC| {abs(ic_mean):.3f} exceeds "
            f"{config.max_abs_ic_mean}, suspect look-ahead"
        )
    if abs(t_stat) < config.min_abs_ic_t_stat:
        return f"weak: |t| {abs(t_stat):.2f} < {config.min_abs_ic_t_stat}"
    if getattr(row, "turnover", np.nan) > config.max_turnover:
        return (
            f"too fast: turnover {getattr(row, 'turnover', float('nan')):.2f} "
            f"> {config.max_turnover}"
        )
    return None


# ----------------------------------------------------------------------
# Correlation between alphas
# ----------------------------------------------------------------------
def signal_correlation_matrix(
    signals: Dict[str, pd.DataFrame],
    *,
    names: Optional[Sequence[str]] = None,
    mask: Optional[pd.DataFrame] = None,
    sample_dates: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Correlation between alphas, measured the way they will be traded.

    Each signal is cross-sectionally ranked and demeaned per date before being
    flattened, so the correlation reflects agreement about *relative ordering*
    -- which is what the portfolio acts on -- rather than agreement about raw
    magnitudes, which may differ by orders of magnitude between alphas.

    Dates are subsampled because the full panel is large and the correlation
    between two signals is stable long before every date has been used.
    """
    names = list(names) if names is not None else list(signals)
    if not names:
        return pd.DataFrame()

    reference = signals[names[0]]
    dates = reference.index
    if sample_dates and len(dates) > sample_dates:
        rng = np.random.default_rng(seed)
        chosen = np.sort(
            rng.choice(len(dates), size=sample_dates, replace=False)
        )
        dates = dates[chosen]

    flattened = {}
    for name in names:
        frame = signals[name].reindex(index=dates)
        if mask is not None:
            frame = frame.where(mask.reindex(index=dates))
        ranked = frame.rank(axis=1, pct=True, na_option="keep")
        ranked = ranked.sub(ranked.mean(axis=1), axis=0)
        flattened[name] = ranked.to_numpy().ravel()

    matrix = pd.DataFrame(flattened).corr(method="pearson", min_periods=100)
    return matrix


def prune_correlated(
    stats: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    candidates: Optional[Sequence[str]] = None,
    max_correlation: float = 0.7,
    rank_by: str = "ic_t_stat",
    max_keep: Optional[int] = None,
) -> SelectionResult:
    """Greedily keep the best alphas that are not near-duplicates of each other.

    Ordering is by standalone quality; an alpha joins the keep-list only if its
    absolute correlation with every already-kept alpha is below
    ``max_correlation``.

    Absolute correlation matters because a perfectly *negatively* correlated
    alpha carries no more independent information than a positive one -- it is
    the same signal with the sign flipped, and the combiner can flip signs by
    itself.
    """
    result = SelectionResult(correlation_matrix=correlation)

    ordering = stats.copy()
    if candidates is not None:
        ordering = ordering[ordering["name"].isin(list(candidates))]
    if ordering.empty:
        return result

    ordering["_rank_key"] = ordering[rank_by].abs()
    ordering = ordering.sort_values("_rank_key", ascending=False)

    kept: List[str] = []
    for name in ordering["name"]:
        if max_keep is not None and len(kept) >= max_keep:
            result.rejected[name] = f"capped: already kept {max_keep}"
            continue
        if name not in correlation.columns:
            kept.append(name)
            continue

        clash, value = _worst_correlation(name, kept, correlation)
        if clash is not None and abs(value) > max_correlation:
            result.rejected[name] = (
                f"redundant: |corr| {abs(value):.2f} with {clash}"
            )
            continue
        kept.append(name)

    result.kept = kept
    return result


def _worst_correlation(
    name: str, kept: Sequence[str], correlation: pd.DataFrame
) -> Tuple[Optional[str], float]:
    """The already-kept alpha this one most resembles."""
    worst_name: Optional[str] = None
    worst_value = 0.0

    for other in kept:
        if other not in correlation.index:
            continue
        value = correlation.at[name, other]
        if not np.isfinite(value):
            continue
        if abs(value) > abs(worst_value):
            worst_name, worst_value = other, float(value)

    return worst_name, worst_value


def select_alphas(
    stats: pd.DataFrame,
    signals: Dict[str, pd.DataFrame],
    *,
    screen: Optional[ScreenConfig] = None,
    max_correlation: float = 0.7,
    max_keep: Optional[int] = 40,
    mask: Optional[pd.DataFrame] = None,
    seed: int = 0,
) -> Tuple[SelectionResult, SelectionResult]:
    """Run both filters in order and return each stage's result.

    Returning both stages rather than only the survivors is deliberate: the
    report shows how many alphas died of weakness versus how many died of
    redundancy, and those two numbers say very different things about the
    library.
    """
    screened = screen_alphas(stats, screen)
    if not screened.kept:
        return screened, SelectionResult()

    correlation = signal_correlation_matrix(
        signals, names=screened.kept, mask=mask, seed=seed
    )
    pruned = prune_correlated(
        stats,
        correlation,
        candidates=screened.kept,
        max_correlation=max_correlation,
        max_keep=max_keep,
    )
    return screened, pruned
