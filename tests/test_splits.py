"""Tests for the purged walk-forward cross-validation splitter."""

import pandas as pd
import pytest

from alphalab.ml.splits import Fold, PurgedWalkForward, assert_no_leakage


@pytest.fixture
def dates():
    return pd.bdate_range("2015-01-01", periods=1500)


class TestPurgedWalkForward:
    def test_no_overlap(self, dates):
        """Train and test sets must never share a date."""
        splitter = PurgedWalkForward(n_splits=5, horizon=1, embargo=5)
        folds = splitter.split(dates)
        for fold in folds:
            assert fold.train_end <= fold.test_start

    def test_purge_gap(self, dates):
        """The gap between training and testing must be at least horizon + embargo."""
        splitter = PurgedWalkForward(n_splits=5, horizon=5, embargo=10)
        folds = splitter.split(dates)
        for fold in folds:
            assert fold.gap >= 15  # 5 + 10

    def test_no_leakage_assertion(self, dates):
        """The leakage assertion must pass for a correctly constructed split."""
        splitter = PurgedWalkForward(n_splits=4, horizon=5, embargo=5)
        folds = splitter.split(dates)
        assert_no_leakage(folds, horizon=5)  # should not raise

    def test_leakage_detected(self):
        """A manually broken fold should trigger the assertion."""
        bad_fold = Fold(index=0, train_start=0, train_end=100, test_start=100, test_end=200)
        with pytest.raises(AssertionError, match="label spans"):
            assert_no_leakage([bad_fold], horizon=5)

    def test_expanding_window(self, dates):
        """Expanding mode: every fold trains from the start."""
        splitter = PurgedWalkForward(n_splits=3, expanding=True)
        folds = splitter.split(dates)
        for fold in folds:
            assert fold.train_start == 0

    def test_folds_cover_test_period(self, dates):
        """Test folds should tile the period after the warmup without gaps."""
        splitter = PurgedWalkForward(n_splits=5, horizon=1, embargo=5)
        folds = splitter.split(dates)
        for i in range(len(folds) - 1):
            assert folds[i].test_end == folds[i + 1].test_start

    def test_too_few_dates_raises(self):
        short = pd.bdate_range("2020-01-01", periods=50)
        splitter = PurgedWalkForward(n_splits=5, min_train=504)
        with pytest.raises(ValueError, match="too few"):
            splitter.split(short)
