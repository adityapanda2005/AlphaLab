"""Integration test: the full pipeline on a small panel."""

import warnings
import pytest
from alphalab.data import MarketConfig
from alphalab.pipeline import PipelineConfig, run_pipeline
from alphalab.portfolio import PortfolioConfig
from alphalab.research import ScreenConfig


class TestPipeline:
    @pytest.fixture(scope="class")
    def result(self):
        """Run the pipeline once for the whole class."""
        cfg = PipelineConfig(
            source="synthetic",
            market=MarketConfig(n_instruments=50, n_days=600, seed=7),
            screen=ScreenConfig(min_abs_ic_t_stat=1.5, min_observations=100),
            portfolio=PortfolioConfig(cost_bps=3.0),
            top_n_universe=40,
            max_keep=15,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return run_pipeline(cfg, verbose=False)

    def test_all_alphas_evaluated(self, result):
        assert result.n_alphas_total >= 100

    def test_some_survived_screen(self, result):
        assert result.n_alphas_screened > 0

    def test_some_survived_pruning(self, result):
        assert result.n_alphas_pruned > 0

    def test_backtest_produced(self, result):
        assert len(result.backtests) > 0

    def test_sharpe_is_finite(self, result):
        import math
        for name, bt in result.backtests.items():
            assert math.isfinite(bt.sharpe), f"{name} Sharpe is not finite"

    def test_overfit_report_exists(self, result):
        assert result.overfit is not None

    def test_dsr_is_finite(self, result):
        import math
        assert math.isfinite(result.overfit.dsr_p_value)

    def test_cache_reuse(self, result):
        """The cache should reuse at least some nodes."""
        assert result.eval_reuse_ratio > 0.3
