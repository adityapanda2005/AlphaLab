"""Walk-forward cross-validation with purging and embargo.

This is the most important file in the machine-learning layer, and the reason
is worth stating plainly: **ordinary k-fold cross-validation is invalid on this
problem, and using it produces backtests that look wonderful and are worthless.**

Three separate things go wrong with a naive shuffle-and-split.

1. *Training on the future.*  A random fold puts 2022 in the training set and
   2019 in the test set.  The model learns relationships that had not been
   discovered yet, and the "out-of-sample" score measures nothing.

2. *Label overlap -- the subtle one.*  The target is a forward return.  With a
   five-day horizon, the label attached to Monday is the return earned through
   Friday.  If training ends on Monday and testing begins on Tuesday, the
   Monday training label already contains four days of the test period.  The
   model has been shown the answer.  **Purging** fixes this by deleting the
   last ``horizon`` observations from the training set, so no surviving
   training label can reach into the test window.

3. *Serial correlation across the boundary.*  Volatility, volume and the
   signals themselves are highly autocorrelated, so the days immediately after
   a test fold still resemble it closely.  Training on them leaks information
   about the test period through the back door.  An **embargo** -- a further
   gap of a few days that belongs to neither set -- removes that.

The construction follows Marcos López de Prado's treatment in *Advances in
Financial Machine Learning* (2018), chapter 7.

Anchored or rolling
-------------------
``expanding=True`` (the default) grows the training window forward: every fold
trains on all history up to its cut.  ``expanding=False`` keeps a fixed-length
window, which adapts faster if the market's behaviour changes but throws away
data.  Both are offered because which is right is an empirical question, not a
matter of principle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/test split, expressed as integer positions into the dates."""

    index: int
    train_start: int
    train_end: int   # exclusive
    test_start: int  # exclusive of the purge and embargo gap before it
    test_end: int    # exclusive

    @property
    def n_train(self) -> int:
        return max(0, self.train_end - self.train_start)

    @property
    def n_test(self) -> int:
        return max(0, self.test_end - self.test_start)

    @property
    def gap(self) -> int:
        """Days discarded between the end of training and the start of testing."""
        return max(0, self.test_start - self.train_end)

    def describe(self, dates: Optional[pd.Index] = None) -> str:
        if dates is None:
            return (
                f"fold {self.index}: train [{self.train_start}:{self.train_end}) "
                f"gap {self.gap} test [{self.test_start}:{self.test_end})"
            )
        return (
            f"fold {self.index}: train {dates[self.train_start].date()} to "
            f"{dates[self.train_end - 1].date()} | gap {self.gap}d | "
            f"test {dates[self.test_start].date()} to "
            f"{dates[self.test_end - 1].date()}"
        )


class PurgedWalkForward:
    """Generate forward-moving folds separated by a purge-and-embargo gap.

    Parameters
    ----------
    n_splits:
        How many test folds to produce.
    horizon:
        The label's forward-looking horizon, in days.  This is what sets the
        size of the purge: a label computed over ``horizon`` days must be
        removed if its window would reach into the test set.
    embargo:
        Extra days dropped after the purge, to break serial correlation across
        the boundary.
    min_train:
        The first fold is only emitted once this many training days exist, so
        no model is fitted on a handful of observations.
    expanding:
        Anchor the training window at the start of the sample (``True``) or
        keep it a fixed length (``False``).
    """

    def __init__(
        self,
        n_splits: int = 6,
        *,
        horizon: int = 1,
        embargo: int = 5,
        min_train: int = 504,
        expanding: bool = True,
        train_window: Optional[int] = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")

        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        self.min_train = min_train
        self.expanding = expanding
        self.train_window = train_window

    # ------------------------------------------------------------------
    @property
    def total_gap(self) -> int:
        """Days between the last training observation and the first test one.

        ``horizon`` days of purge, because a label spanning ``horizon`` days
        from the final training date would otherwise overlap the test window,
        plus ``embargo`` days to break serial correlation.
        """
        return self.horizon + self.embargo

    def split(self, dates: Sequence) -> List[Fold]:
        """Produce the folds for a given date index."""
        n_dates = len(dates)
        gap = self.total_gap

        usable = n_dates - self.min_train - gap
        if usable <= 0:
            raise ValueError(
                f"only {n_dates} dates: too few for min_train={self.min_train} "
                f"plus a {gap}-day purge and embargo gap"
            )

        test_size = usable // self.n_splits
        if test_size < 1:
            raise ValueError(
                f"{n_dates} dates cannot be divided into {self.n_splits} folds "
                f"after reserving {self.min_train} for training"
            )

        folds: List[Fold] = []
        for i in range(self.n_splits):
            test_start = self.min_train + gap + i * test_size
            test_end = (
                n_dates if i == self.n_splits - 1 else test_start + test_size
            )
            if test_start >= n_dates:
                break

            train_end = test_start - gap
            if self.expanding or self.train_window is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_window)

            if train_end - train_start < 2:
                continue

            folds.append(
                Fold(
                    index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=min(test_end, n_dates),
                )
            )

        if not folds:
            raise ValueError("no valid folds could be constructed")
        return folds

    def __iter__(self) -> Iterator:  # pragma: no cover - convenience only
        raise TypeError("call split(dates) to generate folds")


def assert_no_leakage(folds: Sequence[Fold], horizon: int) -> None:
    """Fail loudly if any fold could let a training label see its test window.

    Called by the pipeline and asserted in ``tests/test_splits.py``.  A
    silently broken splitter is the single most expensive bug available in
    this domain, so it is checked rather than trusted: every training
    observation's label must finish strictly before its test fold begins.
    """
    for fold in folds:
        if fold.train_end > fold.test_start:
            raise AssertionError(
                f"fold {fold.index}: training overlaps the test window"
            )

        last_train_label_ends = (fold.train_end - 1) + horizon
        if last_train_label_ends >= fold.test_start:
            raise AssertionError(
                f"fold {fold.index}: the last training label spans to position "
                f"{last_train_label_ends}, which reaches the test set starting "
                f"at {fold.test_start}. The purge is too small for a "
                f"{horizon}-day horizon."
            )


def describe_folds(folds: Sequence[Fold], dates: Optional[pd.Index] = None) -> str:
    return "\n".join(fold.describe(dates) for fold in folds)
