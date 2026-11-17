"""Tests for the operator library."""

import numpy as np
import pandas as pd
import pytest

from alphalab.expr import Evaluator
from alphalab.expr.operators import count, get
from alphalab.expr.operators.registry import ArityError, OperatorError


@pytest.fixture
def toy_data():
    """8 instruments, 60 days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=60)
    cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 8)), axis=0)),
        index=dates, columns=cols,
    )
    volume = pd.DataFrame(rng.lognormal(10, 0.3, (60, 8)), index=dates, columns=cols)
    sector = pd.DataFrame(
        np.tile(["A", "A", "B", "B", "C", "C", "A", "B"], (60, 1)),
        index=dates, columns=cols,
    )
    return {
        "close": close, "open": close * 0.999, "high": close * 1.005,
        "low": close * 0.995, "volume": volume, "sector": sector,
        "returns": close.pct_change(fill_method=None),
        "market": pd.DataFrame(
            np.tile(close.pct_change(fill_method=None).mean(axis=1).values[:, None], 8),
            index=dates, columns=cols,
        ),
        "dollar_volume": close * volume,
        "vwap": (close * 1.005 + close * 0.995 + close) / 3,
    }


class TestRegistry:
    def test_all_registered(self):
        assert count() >= 50

    def test_lookup(self):
        spec = get("rank")
        assert spec.category == "cross_sectional"

    def test_unknown_raises(self):
        with pytest.raises(OperatorError):
            get("nonexistent_op")

    def test_arity_check(self):
        spec = get("ts_mean")
        with pytest.raises(ArityError):
            spec.check_arity(1)


class TestArithmetic:
    def test_add(self, toy_data):
        ev = Evaluator(toy_data)
        result = ev.evaluate("close + volume")
        expected = toy_data["close"] + toy_data["volume"]
        pd.testing.assert_frame_equal(result, expected)

    def test_divide_by_zero(self, toy_data):
        """Division by zero produces NaN, not inf."""
        ev = Evaluator(toy_data)
        result = ev.evaluate("close / (close - close)")
        assert result.notna().sum().sum() == 0  # all NaN


class TestCrossSectional:
    def test_rank_range(self, toy_data):
        ev = Evaluator(toy_data)
        result = ev.evaluate("rank(close)")
        assert result.min().min() >= 0
        assert result.max().max() <= 1

    def test_demean_sums_to_zero(self, toy_data):
        ev = Evaluator(toy_data)
        result = ev.evaluate("demean(close)")
        row_sums = result.sum(axis=1).dropna()
        np.testing.assert_allclose(row_sums, 0, atol=1e-10)


class TestTimeSeries:
    def test_ts_delay(self, toy_data):
        ev = Evaluator(toy_data)
        delayed = ev.evaluate("ts_delay(close, 1)")
        expected = toy_data["close"].shift(1)
        pd.testing.assert_frame_equal(delayed, expected)

    def test_ts_mean_window(self, toy_data):
        ev = Evaluator(toy_data)
        result = ev.evaluate("ts_mean(close, 20)")
        # First 19 rows should be NaN (full window required)
        assert result.iloc[:19].isna().all().all()
        assert result.iloc[19:].notna().any().any()


class TestNoLookahead:
    """The most important invariant: time-series operators never see the future.

    For every registered ts_* operator, the value at date t must depend only on
    data at dates <= t.  We test this by zeroing out all future data and
    checking that the output at t is unchanged.
    """

    def test_no_lookahead(self, toy_data):
        ev = Evaluator(toy_data)
        formulas = [
            "ts_mean(close, 10)",
            "ts_std(close, 10)",
            "ts_delta(close, 5)",
            "ts_rank(close, 10)",
            "ts_argmax(close, 10)",
            "decay_linear(close, 5)",
        ]
        for formula in formulas:
            full = ev.evaluate(formula)
            # Pick a date in the middle
            test_row = 40
            test_date = full.index[test_row]

            # Truncate data to dates <= test_date
            truncated = {
                name: frame.iloc[: test_row + 1]
                for name, frame in toy_data.items()
            }
            ev_trunc = Evaluator(truncated)
            partial = ev_trunc.evaluate(formula)

            np.testing.assert_array_equal(
                full.loc[test_date].values,
                partial.iloc[-1].values,
                err_msg=f"lookahead detected in {formula}",
            )


class TestGroup:
    def test_group_demean(self, toy_data):
        ev = Evaluator(toy_data)
        result = ev.evaluate("group_demean(close, sector)")
        # Within each sector, the mean should be ~0
        sector = toy_data["sector"]
        for label in ["A", "B", "C"]:
            mask = sector == label
            sector_mean = result.where(mask).mean(axis=1).dropna()
            np.testing.assert_allclose(
                sector_mean, 0, atol=1e-8,
                err_msg=f"group_demean not zero-mean in sector {label}",
            )

    def test_static_vs_dynamic_agree(self, toy_data):
        """The static fast path must produce identical results to the general path."""
        from alphalab.expr.operators.group import (
            _aligned,
            _nanmean,
            _rank_dynamic,
            _rank_static,
            _reduce_dynamic,
            _reduce_static,
            _static_column_codes,
        )
        frame, labels = _aligned(toy_data["close"], toy_data["sector"])
        codes = _static_column_codes(labels)
        assert codes is not None, "toy data should be detected as static"

        fast = _reduce_static(frame, codes, _nanmean)
        slow = _reduce_dynamic(frame, labels, _nanmean)
        pd.testing.assert_frame_equal(fast, slow)

        fast_rank = _rank_static(frame, codes)
        slow_rank = _rank_dynamic(frame, labels)
        pd.testing.assert_frame_equal(fast_rank, slow_rank)
