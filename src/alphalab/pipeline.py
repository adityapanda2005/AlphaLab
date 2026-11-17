"""End-to-end alpha research pipeline.

This is the entry point that ties every layer together: generate data, parse
and evaluate the formula library, measure and screen the results, combine the
survivors with machine learning, build a portfolio, and assess whether the
result is real or overfit.

Usage::

    python -m alphalab.pipeline          # run with default synthetic data
    python -m alphalab.pipeline --source yfinance   # run on real prices

The pipeline is also importable::

    from alphalab.pipeline import run_pipeline
    results = run_pipeline()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .alphas import ALPHA_LIBRARY, FAMILY_OF
from .data import MarketConfig, MarketPanel, load_panel
from .expr import Evaluator, NullCache, SubexpressionCache
from .ml import CombinerConfig, CombinerResult, combine_signals
from .portfolio import BacktestResult, PortfolioConfig, construct_portfolio
from .research import ScreenConfig, evaluate_library, select_alphas
from .stats import OverfitReport, full_overfit_report

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Every knob of the pipeline in one place."""

    source: str = "synthetic"
    market: MarketConfig = field(default_factory=MarketConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    combiner: CombinerConfig = field(default_factory=CombinerConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    max_correlation: float = 0.7
    max_keep: int = 40
    top_n_universe: int = 150
    output_dir: str = "artifacts"
    seed: int = 7


@dataclass
class PipelineResult:
    """Everything the pipeline produced."""

    panel: MarketPanel
    n_alphas_total: int = 0
    n_alphas_screened: int = 0
    n_alphas_pruned: int = 0
    eval_seconds: float = 0.0
    eval_reuse_ratio: float = 0.0
    eval_cache_peak_mb: float = 0.0
    alpha_stats: Optional[pd.DataFrame] = None
    selected_names: List[str] = field(default_factory=list)
    combiner_results: Dict[str, CombinerResult] = field(default_factory=dict)
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)
    overfit: Optional[OverfitReport] = None
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def best_model(self) -> str:
        if not self.backtests:
            return ""
        return max(self.backtests, key=lambda k: abs(self.backtests[k].sharpe))

    @property
    def best_sharpe(self) -> float:
        bm = self.best_model
        return self.backtests[bm].sharpe if bm else float("nan")


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full research pipeline and return the results."""
    config = config or PipelineConfig()
    result = PipelineResult(panel=MarketPanel())

    def log(msg: str) -> None:
        if verbose:
            print(msg)
        logger.info(msg)

    # ------------------------------------------------------------------
    # Stage 1: Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    log("=" * 70)
    log("AlphaLab: Alpha Research & Backtesting Engine")
    log("=" * 70)

    log("\n[1/7] Loading market data...")
    panel = load_panel(config.source, config=config.market)
    result.panel = panel
    result.timings["data"] = time.perf_counter() - t0
    log(f"  {panel.describe()}")

    mask = panel.liquid_universe(top_n=config.top_n_universe)
    log(f"  universe: top {config.top_n_universe} by ADV")

    # ------------------------------------------------------------------
    # Stage 2: Evaluate the alpha library
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    log(f"\n[2/7] Evaluating {len(ALPHA_LIBRARY)} alpha formulas...")
    evaluator = Evaluator(panel, SubexpressionCache())
    signals, eval_report = evaluator.evaluate_many(ALPHA_LIBRARY)
    result.timings["evaluate"] = time.perf_counter() - t1
    result.n_alphas_total = len(signals)
    result.eval_seconds = eval_report.seconds
    result.eval_reuse_ratio = eval_report.reuse_ratio
    result.eval_cache_peak_mb = eval_report.cache_peak_megabytes
    log(f"  {eval_report.summary()}")
    log(f"  cache: {evaluator.cache.stats.summary()}")

    # Benchmark: evaluate without cache for comparison
    time.perf_counter()
    evaluator_nc = Evaluator(panel, NullCache())
    _, rep_nc = evaluator_nc.evaluate_many(ALPHA_LIBRARY, admission="all")
    no_cache_seconds = rep_nc.seconds
    result.timings["evaluate_no_cache"] = no_cache_seconds
    log(f"  without cache: {no_cache_seconds:.2f}s "
        f"(speedup: {no_cache_seconds / eval_report.seconds:.2f}x)")

    # ------------------------------------------------------------------
    # Stage 3: Measure and screen
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    log("\n[3/7] Measuring alpha quality...")
    stats = evaluate_library(signals, panel, mask=mask, families=FAMILY_OF)
    result.alpha_stats = stats
    result.timings["measure"] = time.perf_counter() - t2

    log("  top 5 by |IC-IR|:")
    for _, row in stats.head(5).iterrows():
        log(f"    {row['name']:30s}  IC={row['ic_mean']:+.4f}  "
            f"IR={row['ic_ir']:+.2f}  t={row['ic_t_stat']:+.1f}  "
            f"TO={row['turnover']:.2f}")

    t3 = time.perf_counter()
    log("\n[4/7] Screening and de-duplicating...")
    screened, pruned = select_alphas(
        stats, signals, mask=mask,
        screen=config.screen,
        max_correlation=config.max_correlation,
        max_keep=config.max_keep,
        seed=config.seed,
    )
    result.n_alphas_screened = screened.n_kept
    result.n_alphas_pruned = pruned.n_kept
    result.selected_names = pruned.kept
    result.timings["select"] = time.perf_counter() - t3
    log(f"  screen: {screened.summary()}")
    log(f"  prune:  {pruned.summary()}")
    log(f"  kept ({pruned.n_kept}): {', '.join(pruned.kept[:8])}...")

    if not pruned.kept:
        log("\n  !! No alphas survived. Pipeline stops here.")
        return result

    # ------------------------------------------------------------------
    # Stage 4: Combine with ML
    # ------------------------------------------------------------------
    t4 = time.perf_counter()
    log(f"\n[5/7] Combining {pruned.n_kept} alphas with ML (walk-forward)...")

    for model_name in ("equal_weight", "ridge", "lightgbm"):
        log(f"  fitting: {model_name}...")
        cfg = CombinerConfig(**{**asdict(config.combiner), "model": model_name})
        try:
            cr = combine_signals(
                signals, pruned.kept, panel,
                config=cfg, mask=mask, verbose=verbose,
            )
            result.combiner_results[model_name] = cr
            log(f"    {cr.describe()}")
        except Exception as exc:
            log(f"    FAILED: {exc}")

    result.timings["combine"] = time.perf_counter() - t4

    # ------------------------------------------------------------------
    # Stage 5: Backtest
    # ------------------------------------------------------------------
    t5 = time.perf_counter()
    log("\n[6/7] Backtesting...")

    for model_name, cr in result.combiner_results.items():
        bt = construct_portfolio(
            cr.predictions, panel,
            config=config.portfolio, mask=mask,
        )
        result.backtests[model_name] = bt
        summary = bt.summary()
        log(f"  {model_name:15s}  Sharpe={summary['sharpe']:.3f}  "
            f"Return={summary['total_return']}  DD={summary['max_drawdown']}  "
            f"TO={summary['mean_turnover']:.3f}  names={summary['mean_names']:.0f}")

    result.timings["backtest"] = time.perf_counter() - t5

    # ------------------------------------------------------------------
    # Stage 6: Overfitting check
    # ------------------------------------------------------------------
    t6 = time.perf_counter()
    log("\n[7/7] Overfitting assessment...")

    best = result.best_model
    if best:
        bt = result.backtests[best]
        strategies = {
            name: result.backtests[name].daily_returns
            for name in result.backtests
        }
        report = full_overfit_report(
            bt.daily_returns,
            n_trials=len(ALPHA_LIBRARY),
            returns_by_strategy=strategies,
        )
        result.overfit = report
        log(f"\n{report.summary()}")

    result.timings["overfit"] = time.perf_counter() - t6
    result.timings["total"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log(f"\n{'=' * 70}")
    log(f"Pipeline completed in {result.timings['total']:.1f}s")
    log(f"  Best model: {best} (Sharpe {result.best_sharpe:.3f})")
    log(f"  Deflated Sharpe p-value: {report.dsr_p_value:.3f}"
        if result.overfit else "")
    log(f"{'=' * 70}")

    return result


def save_artifacts(result: PipelineResult, output_dir: str = "artifacts") -> None:
    """Write the pipeline's output to disk for the README and reports."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    if result.alpha_stats is not None:
        result.alpha_stats.to_csv(path / "alpha_stats.csv", index=False)

    for name, bt in result.backtests.items():
        eq = bt.equity_curve()
        eq.to_csv(path / f"equity_{name}.csv")

    if result.overfit:
        (path / "overfit_report.json").write_text(
            json.dumps(asdict(result.overfit), indent=2, default=str)
        )

    # Summary for the README.
    summary: Dict[str, Any] = {
        "panel": result.panel.describe(),
        "n_alphas_total": result.n_alphas_total,
        "n_alphas_screened": result.n_alphas_screened,
        "n_alphas_pruned": result.n_alphas_pruned,
        "evaluator": {
            "seconds": round(result.eval_seconds, 2),
            "reuse_ratio": f"{result.eval_reuse_ratio:.1%}",
            "cache_peak_mb": round(result.eval_cache_peak_mb, 1),
        },
        "backtests": {
            name: bt.summary() for name, bt in result.backtests.items()
        },
        "timings": {k: round(v, 2) for k, v in result.timings.items()},
    }
    if result.overfit:
        summary["overfit"] = {
            "deflated_sharpe": round(result.overfit.deflated_sharpe_ratio, 3),
            "dsr_p_value": round(result.overfit.dsr_p_value, 3),
            "dsr_haircut": round(result.overfit.dsr_haircut, 3),
            "n_trials": result.overfit.n_trials,
        }
        if result.overfit.pbo is not None:
            summary["overfit"]["pbo"] = round(result.overfit.pbo, 3)

    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


# Allow running as a module: python -m alphalab.pipeline
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaLab research pipeline")
    parser.add_argument(
        "--source", default="synthetic",
        choices=["synthetic", "yfinance", "csv"],
    )
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(source=args.source, output_dir=args.output)
    res = run_pipeline(cfg, verbose=not args.quiet)
    save_artifacts(res, args.output)
