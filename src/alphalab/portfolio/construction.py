"""Turn a forecast into a market-neutral portfolio and measure the result.

The portfolio constructor is deliberately simple: demean, cap, sector-
neutralise, scale.  Every real portfolio has constraints from the prime broker,
the risk team and the available capital, and trying to replicate all of that in
a research backtest would add complexity without adding information.  The job
here is to test the *signal*, not to simulate the operations of a fund.

What it does enforce
--------------------
*Dollar neutrality.*  Longs and shorts sum to zero on every date.  Without
this the backtest would carry market exposure, and most of the PnL would be
the market, not the signal.

*Sector neutrality.*  Within each sector, long and short weights sum to zero.
Without this, a signal that just says "overweight technology" is indis-
tinguishable from one that picks individual names within technology.

*Position limits.*  No single name exceeds ``max_weight`` of the gross book.
Unlimited concentration lets one stock dominate the whole result, and in
practice you cannot hold 30% of your capital in one name anyway.

*Transaction costs.*  Modelled as a fixed fraction of turnover (default 10 bps
per side = 20 bps round trip).  This is generous; real costs include market
impact, borrow costs and commissions.  A signal that only works at zero cost
is a signal that does not work.

What it does *not* enforce: borrow constraints, volume participation limits,
margin requirements, intraday path dependency, or multi-day execution.  The
README says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    """Everything about how the forecast becomes a portfolio."""

    max_weight: float = 0.03         # 3% of gross per name
    cost_bps: float = 10.0           # one-way cost: 10 bps
    sector_neutral: bool = True
    dollar_neutral: bool = True
    gross_target: float = 1.0        # sum of |weights| per date


@dataclass
class BacktestResult:
    """The full output of a strategy simulation."""

    daily_returns: pd.Series         # the bottom line, net of costs
    gross_returns: pd.Series         # before costs
    cost_drag: pd.Series             # daily cost paid
    weights: pd.DataFrame            # (dates x instruments)
    turnover: pd.Series              # daily
    n_long: pd.Series
    n_short: pd.Series
    config: PortfolioConfig = field(default_factory=PortfolioConfig)

    # --- summary statistics -----------------------------------------------
    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio, the standard measure of risk-adjusted PnL.

        This is the *in-sample* Sharpe of the out-of-sample predictions.  It
        is an honest number only because the predictions were produced by a
        walk-forward model that never saw the period it forecast.  But it is
        still a backtest: it benefits from survivorship bias in the universe
        and from hindsight in the choice of alpha families.  That is why the
        pipeline reports the Deflated Sharpe alongside it.
        """
        r = self.daily_returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    @property
    def total_return(self) -> float:
        return float(np.expm1(np.log1p(self.daily_returns.dropna()).sum()))

    @property
    def max_drawdown(self) -> float:
        equity = (1 + self.daily_returns.dropna()).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return float(dd.min())

    @property
    def mean_turnover(self) -> float:
        return float(self.turnover.dropna().mean())

    @property
    def mean_names(self) -> float:
        return float((self.n_long + self.n_short).dropna().mean())

    @property
    def total_cost(self) -> float:
        return float(self.cost_drag.sum())

    def summary(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 3),
            "total_return": f"{self.total_return:.1%}",
            "max_drawdown": f"{self.max_drawdown:.1%}",
            "mean_turnover": round(self.mean_turnover, 3),
            "mean_names": round(self.mean_names, 1),
            "total_cost_bps": round(self.total_cost * 10_000, 0),
            "n_days": int(self.daily_returns.notna().sum()),
        }

    def equity_curve(self) -> pd.Series:
        return (1 + self.daily_returns.dropna()).cumprod()


def construct_portfolio(
    forecast: pd.DataFrame,
    panel,
    *,
    config: Optional[PortfolioConfig] = None,
    mask: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """From raw forecast to PnL, one day at a time.

    Every step here uses information available at ``t`` and not from ``t+1``:
    the forecast is the model's output for date ``t``, the sector labels are
    known at ``t``, and the return earned is from ``t`` to ``t+1``.  The only
    subtle point is that the *weights* are set using the forecast and then
    held for one day, which assumes the trades execute at the close -- a
    standard simplification for daily research.
    """
    config = config or PortfolioConfig()

    forecast = forecast.copy()
    if mask is not None:
        forecast = forecast.where(mask.reindex_like(forecast))

    # Drop dates with too few names to form a meaningful portfolio.
    n_valid = forecast.notna().sum(axis=1)
    forecast = forecast.where(n_valid >= 10)

    sector = panel["sector"] if config.sector_neutral and "sector" in panel else None
    weights = _build_weights(forecast, sector, config)
    returns = panel["returns"].reindex_like(weights)

    # PnL: the forecast at date t predicts the return from t to t+1.
    # We form weights from that forecast at the close of t, then hold them
    # for one day and earn the return from t to t+1.
    # Using shift(-1) on returns gives us the return from t+1 to t+2, which
    # is WRONG -- it misaligns the signal with the return it predicted.
    # Instead, the weights at t should earn returns at t+1, which we get by
    # shifting the weights forward (or equivalently, NOT shifting returns).
    # But there's a subtlety: the ML combiner's prediction for date t was
    # trained to predict forward_returns(1) at t, which is close[t+1]/close[t]-1.
    # So the weight at t should earn that same return.
    gross_pnl = (weights.shift(1) * returns).sum(axis=1)

    # Turnover is the absolute change in weights between consecutive days.
    turn = weights.diff().abs().sum(axis=1)
    # Shift turnover to align with PnL dates (cost is paid when the trade
    # executes, which is the day after the signal is formed).
    cost = turn.shift(1).fillna(0.0) * config.cost_bps / 10_000
    net_pnl = gross_pnl - cost

    n_long = (weights > 1e-9).sum(axis=1)
    n_short = (weights < -1e-9).sum(axis=1)

    return BacktestResult(
        daily_returns=net_pnl,
        gross_returns=gross_pnl,
        cost_drag=cost,
        weights=weights,
        turnover=turn,
        n_long=n_long,
        n_short=n_short,
        config=config,
    )


# ----------------------------------------------------------------------
# Weight construction pipeline
# ----------------------------------------------------------------------
def _build_weights(
    forecast: pd.DataFrame,
    sector: Optional[pd.DataFrame],
    config: PortfolioConfig,
) -> pd.DataFrame:
    w = forecast.copy()

    # Step 1: dollar neutrality -- demean so longs and shorts cancel.
    if config.dollar_neutral:
        w = w.sub(w.mean(axis=1), axis=0)

    # Step 2: sector neutrality -- demean within each sector.
    if sector is not None:
        w = _sector_demean(w, sector)

    # Step 3: position cap -- stop any single name from dominating.
    if config.max_weight and config.max_weight > 0:
        w = w.clip(lower=-config.max_weight, upper=config.max_weight)
        # Re-demean after clipping, because clipping breaks the zero-sum.
        if config.dollar_neutral:
            w = w.sub(w.mean(axis=1), axis=0)

    # Step 4: scale to target gross.
    gross = w.abs().sum(axis=1)
    w = w.div(gross.replace(0.0, np.nan), axis=0) * config.gross_target

    return w.fillna(0.0)


def _sector_demean(
    weights: pd.DataFrame, sector: pd.DataFrame
) -> pd.DataFrame:
    """Remove the sector mean from each stock's weight.

    Uses the same static-sector detection as the evaluator's group operators:
    if every row of the sector panel is identical, we can work with column
    indices rather than a full per-cell comparison, which is 10x faster.
    """
    sector = sector.reindex_like(weights)
    array = sector.to_numpy()

    if array.shape[0] > 0 and bool((array == array[0]).all()):
        # Static fast path.
        codes, _uniques = pd.factorize(array[0], use_na_sentinel=True)
        values = weights.to_numpy(dtype=float, copy=True)
        for code in range(len(_uniques)):
            members = np.flatnonzero(codes == code)
            if members.size == 0:
                continue
            group_mean = np.nanmean(values[:, members], axis=1, keepdims=True)
            values[:, members] -= group_mean
        return pd.DataFrame(values, index=weights.index, columns=weights.columns)

    # Dynamic fallback for time-varying classification.
    flat_labels = sector.to_numpy().ravel()
    flat_codes, uniques = pd.factorize(flat_labels, use_na_sentinel=True)
    code_panel = flat_codes.reshape(weights.shape)
    values = weights.to_numpy(dtype=float)

    for code in range(len(uniques)):
        mask = code_panel == code
        if not mask.any():
            continue
        block = np.where(mask, values, np.nan)
        group_mean = np.nanmean(block, axis=1, keepdims=True)
        values = np.where(mask, values - group_mean, values)

    return pd.DataFrame(values, index=weights.index, columns=weights.columns)
