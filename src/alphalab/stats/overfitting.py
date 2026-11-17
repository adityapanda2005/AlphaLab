"""Deflated Sharpe Ratio and Probability of Backtest Overfitting.

Why this module exists
----------------------
Everything up to this point in the pipeline -- the IC scores, the correlation
pruning, the walk-forward model, the backtest Sharpe -- has one deep problem:
*selection bias*.  We looked at 142 alphas.  Even if every one of them were
pure noise, the best-looking one would have a positive IC and a positive
Sharpe.  Multiply by hundreds and the best random portfolio can easily reach
a Sharpe of 1 or higher.

The two statistics here are the corrective lens.  They answer different
variants of the same question:

*Deflated Sharpe Ratio (DSR).*  "Given that I tried N strategies and the best
one achieved Sharpe S, what is the probability that this Sharpe is real rather
than the expected maximum of N independent draws from a zero-mean
distribution?"

*Probability of Backtest Overfitting (PBO).*  "If I split the data into many
sub-periods, build an optimal strategy on each half, and test it on the other,
how often does the in-sample winner *lose* out of sample?"

A DSR p-value above 0.05 means you cannot reject the hypothesis that the
Sharpe is explained by search.  A PBO above 0.5 means more than half of data
splits resulted in an overfit outcome.  The pipeline reports both numbers, and
the README explains what they mean.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
*Journal of Portfolio Management*, 40(5), 94-107.

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017).
"The Probability of Backtest Overfitting."
*Journal of Computational Finance*, 20(4), 39-69.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

TRADING_DAYS = 252


@dataclass
class OverfitReport:
    """Combined overfitting diagnostics for the pipeline."""

    deflated_sharpe_ratio: float
    dsr_p_value: float
    dsr_haircut: float                # how much the raw Sharpe is worth, corrected
    n_trials: int
    raw_sharpe: float
    skewness: float
    kurtosis: float
    n_observations: int
    pbo: Optional[float] = None       # None if not computed
    pbo_n_combinations: int = 0
    pbo_logit_mean: Optional[float] = None

    @property
    def is_significant(self) -> bool:
        return self.dsr_p_value < 0.05

    def summary(self) -> str:
        lines = [
            f"Deflated Sharpe Ratio: {self.deflated_sharpe_ratio:.3f} "
            f"(p={self.dsr_p_value:.3f})",
            f"  raw Sharpe {self.raw_sharpe:.3f}, after deflation "
            f"{self.dsr_haircut:.3f}",
            f"  {self.n_trials} strategies tried, "
            f"{self.n_observations} daily observations",
            f"  skewness {self.skewness:.2f}, excess kurtosis {self.kurtosis:.2f}",
        ]
        if self.pbo is not None:
            lines.append(
                f"Probability of Backtest Overfitting: {self.pbo:.1%} "
                f"(over {self.pbo_n_combinations} combinations)"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Deflated Sharpe Ratio
# ----------------------------------------------------------------------
def deflated_sharpe(
    returns: pd.Series,
    *,
    n_trials: int,
    annualise: bool = True,
) -> OverfitReport:
    """Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    returns:
        Daily returns of the selected strategy.
    n_trials:
        How many strategies were examined before this one was chosen.  In this
        pipeline that is the number of alphas evaluated in the library, not
        just the survivors, because the screening process itself is what
        introduces the selection bias.

    Returns
    -------
    An :class:`OverfitReport` with the DSR fields filled in and ``pbo=None``.

    The mathematics
    ~~~~~~~~~~~~~~~
    Under the null hypothesis that the true Sharpe is zero, the expected
    maximum Sharpe of ``n_trials`` independent draws is approximately::

        E[max SR] ≈ √(2 * log(n_trials)) - (γ + log(π/2)) / (2 * √(2 * log(n_trials)))

    where γ is the Euler-Mascheroni constant.  The DSR asks whether the
    observed Sharpe is significantly above this benchmark, correcting for
    skewness and kurtosis of the return distribution (because the standard
    error of the Sharpe ratio depends on the shape of the distribution, not
    just its first two moments).
    """
    r = returns.dropna()
    n = len(r)
    if n < 10 or n_trials < 1:
        return OverfitReport(
            deflated_sharpe_ratio=float("nan"),
            dsr_p_value=float("nan"),
            dsr_haircut=float("nan"),
            n_trials=n_trials,
            raw_sharpe=float("nan"),
            skewness=float("nan"),
            kurtosis=float("nan"),
            n_observations=n,
        )

    # Sample statistics.
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    if std == 0:
        return OverfitReport(
            deflated_sharpe_ratio=0.0,
            dsr_p_value=1.0,
            dsr_haircut=0.0,
            n_trials=n_trials,
            raw_sharpe=0.0,
            skewness=0.0,
            kurtosis=0.0,
            n_observations=n,
        )

    sr = mean / std
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r))  # excess kurtosis

    if annualise:
        raw_sharpe = sr * np.sqrt(TRADING_DAYS)
    else:
        raw_sharpe = sr

    # Expected maximum Sharpe under the null.
    sr0 = _expected_max_sr(n_trials, n)

    # Standard error of the Sharpe estimator, correcting for higher moments.
    # From Lo (2002) and Bailey & López de Prado (2014).
    se_sr = np.sqrt(
        (1 - skew * sr + (kurt - 1) / 4.0 * sr**2) / (n - 1)
    )
    if se_sr <= 0:
        se_sr = 1.0 / np.sqrt(n)

    # The Deflated Sharpe Ratio is a one-sided z-test of SR against SR0.
    dsr = (sr - sr0) / se_sr
    p_value = float(scipy_stats.norm.sf(dsr))  # right tail

    # The "haircut" is the Sharpe that remains after subtracting what selection
    # bias alone can explain.
    haircut = (sr - sr0) * np.sqrt(TRADING_DAYS) if annualise else (sr - sr0)

    return OverfitReport(
        deflated_sharpe_ratio=float(dsr),
        dsr_p_value=p_value,
        dsr_haircut=max(float(haircut), 0.0),
        n_trials=n_trials,
        raw_sharpe=float(raw_sharpe),
        skewness=skew,
        kurtosis=kurt,
        n_observations=n,
    )


def _expected_max_sr(n_trials: int, n_obs: int) -> float:
    """Bailey & López de Prado (2014), Equation 2.

    The expected maximum daily Sharpe of ``n_trials`` independent strategies
    when the true Sharpe is zero.
    """
    if n_trials <= 1:
        return 0.0
    z = np.sqrt(2.0 * np.log(n_trials))
    euler_mascheroni = 0.5772156649015329
    correction = (euler_mascheroni + np.log(np.pi / 2.0)) / (2.0 * z)
    return float((z - correction) / np.sqrt(n_obs))


# ----------------------------------------------------------------------
# Probability of Backtest Overfitting (combinatorial purged CV)
# ----------------------------------------------------------------------
def probability_of_backtest_overfitting(
    returns_by_alpha: Dict[str, pd.Series],
    *,
    n_partitions: int = 16,
    seed: int = 42,
    max_combinations: int = 500,
) -> OverfitReport:
    """Combinatorial PBO: how often the in-sample winner fails out of sample.

    Parameters
    ----------
    returns_by_alpha:
        Daily return series for every candidate strategy (usually just the
        final combiner variants, not the raw alphas).
    n_partitions:
        Split the timeline into this many equal-length blocks.  Half form the
        "in-sample", the other half "out-of-sample".  With 16 blocks that is
        C(16,8) = 12 870 possible splits, far more than any sequential test,
        and the symmetry of the construction means every block serves equally
        often in and out of sample.
    max_combinations:
        Cap the number of splits actually evaluated, since C(16,8) is fast but
        a larger partition count could blow up.  Randomly sampled combinations
        still produce a stable estimate of PBO.

    The procedure
    ~~~~~~~~~~~~~
    For each half-and-half split:

    1.  Measure the Sharpe of every alpha on the in-sample half.
    2.  Pick the in-sample winner.
    3.  Measure the in-sample winner's Sharpe on the out-of-sample half.
    4.  Rank the winner among all alphas on the out-of-sample half.
    5.  Record ``logit(rank / n_alphas)``.  If the in-sample winner is below
        median out of sample, the logit is negative.

    PBO = fraction of combinations where the logit is negative.
    PBO > 0.5 means overfitting is the norm, not the exception.
    """
    from itertools import combinations

    names = sorted(returns_by_alpha)
    n_alphas = len(names)
    if n_alphas < 2 or n_partitions < 4:
        return _empty_overfit_report()

    # Align all series to the same date index.
    combined = pd.DataFrame(returns_by_alpha)
    combined = combined.dropna(how="all")
    n_dates = len(combined)

    if n_dates < n_partitions * 20:
        return _empty_overfit_report()

    # Split the timeline into equal blocks.
    block_size = n_dates // n_partitions
    block_sharpes = np.zeros((n_partitions, n_alphas))
    for block in range(n_partitions):
        start = block * block_size
        end = start + block_size
        chunk = combined.iloc[start:end]
        for j, name in enumerate(names):
            series = chunk[name].dropna()
            if len(series) > 1 and series.std() > 0:
                block_sharpes[block, j] = series.mean() / series.std()

    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))

    rng = np.random.default_rng(seed)
    if len(all_combos) > max_combinations:
        indices = rng.choice(len(all_combos), size=max_combinations, replace=False)
        selected = [all_combos[i] for i in indices]
    else:
        selected = all_combos

    logits: List[float] = []
    for in_sample_blocks in selected:
        out_sample_blocks = tuple(
            b for b in range(n_partitions) if b not in in_sample_blocks
        )

        # Sharpe of each alpha on the in-sample and out-of-sample halves.
        is_sharpes = block_sharpes[list(in_sample_blocks)].mean(axis=0)
        oos_sharpes = block_sharpes[list(out_sample_blocks)].mean(axis=0)

        # Which alpha wins in-sample?
        winner = int(np.argmax(is_sharpes))

        # Where does the winner rank out-of-sample?
        oos_rank = int((oos_sharpes >= oos_sharpes[winner]).sum())
        relative_rank = oos_rank / n_alphas

        # Logit of the relative rank.  Clamp to avoid log(0).
        relative_rank = np.clip(relative_rank, 0.001, 0.999)
        logits.append(float(np.log(relative_rank / (1.0 - relative_rank))))

    n_negative = sum(1 for val in logits if val < 0)
    pbo = float(n_negative / len(logits)) if logits else float("nan")
    logit_mean = float(np.mean(logits)) if logits else float("nan")

    report = _empty_overfit_report()
    report.pbo = pbo
    report.pbo_n_combinations = len(selected)
    report.pbo_logit_mean = logit_mean
    return report


def full_overfit_report(
    daily_returns: pd.Series,
    *,
    n_trials: int,
    returns_by_strategy: Optional[Dict[str, pd.Series]] = None,
    n_partitions: int = 16,
) -> OverfitReport:
    """Run both the DSR and PBO analyses and merge the results."""
    report = deflated_sharpe(daily_returns, n_trials=n_trials)

    if returns_by_strategy and len(returns_by_strategy) >= 3:
        pbo_report = probability_of_backtest_overfitting(
            returns_by_strategy, n_partitions=n_partitions
        )
        report.pbo = pbo_report.pbo
        report.pbo_n_combinations = pbo_report.pbo_n_combinations
        report.pbo_logit_mean = pbo_report.pbo_logit_mean

    return report


def _empty_overfit_report() -> OverfitReport:
    return OverfitReport(
        deflated_sharpe_ratio=float("nan"),
        dsr_p_value=float("nan"),
        dsr_haircut=float("nan"),
        n_trials=0,
        raw_sharpe=float("nan"),
        skewness=float("nan"),
        kurtosis=float("nan"),
        n_observations=0,
    )
