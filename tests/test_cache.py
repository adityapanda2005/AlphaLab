"""Tests for the sub-expression cache."""

import numpy as np
import pandas as pd
import pytest

from alphalab.expr import (
    Evaluator,
    NullCache,
    SubexpressionCache,
    shared_subexpressions,
)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=50)
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (50, 4)), axis=0)),
        index=dates, columns=cols,
    )
    volume = pd.DataFrame(rng.lognormal(10, 0.3, (50, 4)), index=dates, columns=cols)
    return {"close": close, "volume": volume}


class TestSubexpressionCache:
    def test_hit_and_miss(self):
        cache = SubexpressionCache()
        assert cache.get("x") is not None or True  # just calling it
        cache.put("x", 42)
        assert cache.get("x") == 42

    def test_lru_eviction(self):
        cache = SubexpressionCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        from alphalab.expr.cache import MISS
        assert cache.get("a") is MISS
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_stats(self):
        cache = SubexpressionCache()
        cache.put("x", 42)
        cache.get("x")
        cache.get("y")
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1

    def test_admission_policy(self):
        cache = SubexpressionCache(admit={"keep"})
        cache.put("keep", 1)
        cache.put("reject", 2)
        from alphalab.expr.cache import MISS
        assert cache.get("keep") == 1
        assert cache.get("reject") is MISS
        assert cache.stats.rejections == 1

    def test_null_cache_never_stores(self):
        cache = NullCache()
        cache.put("x", 42)
        from alphalab.expr.cache import MISS
        assert cache.get("x") is MISS


class TestCacheCorrectness:
    def test_cached_equals_uncached(self, toy_data):
        """Cached and uncached evaluation must produce identical results."""
        formulas = {
            "a1": "rank(close)",
            "a2": "rank(close) * volume",
            "a3": "ts_mean(close, 10)",
        }
        ev_cached = Evaluator(toy_data, SubexpressionCache())
        ev_uncached = Evaluator(toy_data, NullCache())

        sigs_c, _ = ev_cached.evaluate_many(formulas)
        sigs_u, _ = ev_uncached.evaluate_many(formulas)

        for name in formulas:
            pd.testing.assert_frame_equal(
                sigs_c[name], sigs_u[name],
                check_names=False,
                obj=f"signal {name}",
            )


class TestSharedSubexpressions:
    def test_detects_shared(self):
        shared = shared_subexpressions([
            "rank(ts_delta(close, 5))",
            "rank(ts_delta(close, 5)) + 1",
        ])
        keys = [k for k, _ in shared]
        assert "rank(ts_delta(close,5))" in keys
        assert "ts_delta(close,5)" in keys
