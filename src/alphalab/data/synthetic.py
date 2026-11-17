"""A synthetic equity market with known, deliberately planted effects.

Why a simulator at all?
-----------------------
Two reasons, and the second is the interesting one.

1. *Reproducibility.*  Free vendor data changes under you, needs network
   access, and cannot be committed to a repository.  A seeded simulator means
   ``make pipeline`` produces the same numbers on any machine, forever, which
   is what makes the benchmarks in the README checkable rather than merely
   claimed.

2. *Ground truth.*  On real data nobody knows what the true signal is, so a
   backtest can only ever tell you that something worked -- never that your
   machinery is correct.  Here the predictable structure is planted by hand at
   a known strength, so the pipeline can be asked a much sharper question:
   *does it recover the effect I put in, at roughly the size I put in?*  A
   research stack that cannot pass that test has a bug, however good its
   backtest looks.  ``tests/test_recovers_planted_signal.py`` makes exactly
   this assertion.

What is planted
---------------
The return of instrument ``i`` on day ``t`` is built as::

    r_it = beta_i * market_t + sector_loading_i * sector_t + idio_it

    idio_it = theta_it                      <- slow, persistent: momentum
            + eps_it                        <- today's shock
            - phi_it * eps_i,t-1            <- yesterday's shock, reversed

* ``theta`` is a highly persistent AR(1), so past returns carry a little
  information about future returns: a weak **momentum** effect.
* The ``- phi * eps_{t-1}`` term is an MA(1) with a negative coefficient,
  which is a **short-term reversal** effect: what went up yesterday gives a
  little back today.
* ``phi`` itself rises with yesterday's **volume** shock, reproducing the
  well-documented fact that reversal is stronger after unusually heavy
  trading.  This is what gives volume-conditioned alphas something real to
  find.

Realism beyond the planted signal: Student-t shocks (fat tails), GARCH-style
volatility clustering, persistent and return-correlated volume, a market
factor, sector factors, and OHLC bars whose highs and lows are consistent with
the day's close.

Everything here is a simulation.  No number produced from it is a claim about
any real market, and the README says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .panel import MarketPanel

#: Sector names used for the simulated classification.
SECTORS = (
    "Technology",
    "Financials",
    "Healthcare",
    "Energy",
    "Industrials",
    "ConsumerStaples",
    "ConsumerDiscretionary",
    "Utilities",
)


@dataclass(frozen=True)
class MarketConfig:
    """Every knob of the simulator, in one auditable place.

    The defaults are chosen so the planted edge is *weak* -- a daily
    information coefficient in the low hundredths.  That is roughly what a real
    cross-sectional equity signal looks like, and it keeps the project honest:
    a simulator tuned to produce a Sharpe of 5 would make the whole pipeline
    look good regardless of whether it works.
    """

    n_instruments: int = 200
    n_days: int = 2520  # about ten trading years
    start_date: str = "2014-01-02"
    seed: int = 20240115

    # --- factor structure -------------------------------------------------
    market_vol: float = 0.009          # daily stdev of the market factor
    sector_vol: float = 0.006          # daily stdev of each sector factor
    beta_mean: float = 1.0
    beta_dispersion: float = 0.25

    # --- idiosyncratic risk ----------------------------------------------
    idio_vol_mean: float = 0.016
    idio_vol_dispersion: float = 0.005
    garch_alpha: float = 0.08          # weight on yesterday's squared shock
    garch_beta: float = 0.90           # weight on yesterday's variance
    t_degrees_of_freedom: float = 4.0  # fat tails; lower = fatter

    # --- planted predictability ------------------------------------------
    reversal_strength: float = 0.040   # base MA(1) coefficient, phi0
    reversal_volume_sensitivity: float = 0.045  # how much phi rises with volume
    momentum_persistence: float = 0.990         # AR(1) rho for theta
    # Shock stdev for theta.  The *stationary* stdev is this divided by
    # sqrt(1 - rho^2), i.e. about 7x larger, which is the number that actually
    # determines how strong momentum is.  It is deliberately tiny: at 0.00008
    # the spread in expected daily return across the universe is a few basis
    # points, giving a 12-1 momentum IC in the low hundredths, as in reality.
    momentum_scale: float = 0.00008

    # --- volume -----------------------------------------------------------
    log_volume_mean: float = 13.8
    log_volume_dispersion: float = 0.6
    volume_persistence: float = 0.85
    volume_return_sensitivity: float = 0.55  # |return| -> volume
    volume_noise: float = 0.35

    # --- prices and bars ---------------------------------------------------
    initial_price_low: float = 15.0
    initial_price_high: float = 350.0
    overnight_gap_share: float = 0.35  # fraction of daily move that gaps
    # Overnight moves are not merely a fixed slice of the daily move: news
    # arrives between the close and the next open and is not fully reversed
    # intraday.  Without this independent component the gap and the intraday
    # move would both be deterministic multiples of the daily return, hence
    # perfectly rank-correlated with it and with each other -- which would
    # make three genuinely different alphas look identical.
    overnight_noise_share: float = 0.45  # as a fraction of idiosyncratic vol
    intraday_range_multiple: float = 1.6

    # --- data quality ------------------------------------------------------
    missing_data_rate: float = 0.001   # sporadic gaps, as in real vendor data

    def __post_init__(self) -> None:
        if self.n_instruments < 2:
            raise ValueError("need at least 2 instruments")
        if self.n_days < 60:
            raise ValueError("need at least 60 days for rolling windows")
        if not 0.0 <= self.garch_alpha + self.garch_beta < 1.0:
            raise ValueError("garch_alpha + garch_beta must be in [0, 1)")


def generate_market(
    config: Optional[MarketConfig] = None, **overrides
) -> MarketPanel:
    """Simulate a market and return it as a :class:`MarketPanel`.

    >>> panel = generate_market(MarketConfig(n_instruments=10, n_days=120))
    >>> panel.shape
    (120, 10)
    """
    config = config or MarketConfig()
    if overrides:
        config = MarketConfig(**{**config.__dict__, **overrides})

    rng = np.random.default_rng(config.seed)
    n, t = config.n_instruments, config.n_days

    dates = pd.bdate_range(config.start_date, periods=t)
    tickers = _make_tickers(n)
    sector_ids = rng.integers(0, len(SECTORS), size=n)

    # ------------------------------------------------------------------
    # Static instrument characteristics
    # ------------------------------------------------------------------
    betas = rng.normal(config.beta_mean, config.beta_dispersion, size=n)
    sector_loadings = rng.uniform(0.6, 1.4, size=n)
    base_idio_vol = np.clip(
        rng.normal(config.idio_vol_mean, config.idio_vol_dispersion, size=n),
        0.004,
        0.060,
    )
    base_log_volume = rng.normal(
        config.log_volume_mean, config.log_volume_dispersion, size=n
    )

    # ------------------------------------------------------------------
    # Factor paths
    # ------------------------------------------------------------------
    market_factor = rng.standard_t(
        config.t_degrees_of_freedom, size=t
    ) * config.market_vol / _t_scale(config.t_degrees_of_freedom)
    sector_factor = rng.standard_t(
        config.t_degrees_of_freedom, size=(t, len(SECTORS))
    ) * config.sector_vol / _t_scale(config.t_degrees_of_freedom)

    # ------------------------------------------------------------------
    # Sequential simulation of idiosyncratic returns and volume
    # ------------------------------------------------------------------
    returns = np.zeros((t, n))
    log_volume = np.zeros((t, n))

    variance = base_idio_vol**2                      # GARCH state
    previous_shock = np.zeros(n)                     # eps_{t-1}
    volume_state = np.zeros(n)                       # AR(1) volume state

    # Start theta at its stationary distribution rather than at zero, so the
    # momentum effect has its full strength from day one instead of warming up
    # over the first few hundred rows.
    theta_stationary_sd = config.momentum_scale / np.sqrt(
        1.0 - config.momentum_persistence**2
    )
    theta = rng.normal(0.0, theta_stationary_sd, size=n)  # momentum state

    t_scale = _t_scale(config.t_degrees_of_freedom)

    for day in range(t):
        # --- volatility clustering (GARCH(1,1)) ---
        if day > 0:
            variance = (
                base_idio_vol**2 * (1 - config.garch_alpha - config.garch_beta)
                + config.garch_alpha * previous_shock**2
                + config.garch_beta * variance
            )
        volatility = np.sqrt(variance)

        # --- today's idiosyncratic shock, fat tailed ---
        shock = (
            rng.standard_t(config.t_degrees_of_freedom, size=n)
            / t_scale
            * volatility
        )

        # --- slow, persistent expected-return component => momentum ---
        theta = config.momentum_persistence * theta + rng.normal(
            0.0, config.momentum_scale, size=n
        )

        # --- reversal, amplified by yesterday's volume surprise ---
        # volume_state is yesterday's volume shock at this point in the loop,
        # so nothing here uses information from today or later.
        phi = config.reversal_strength + (
            config.reversal_volume_sensitivity * np.clip(volume_state, -2.0, 3.0)
        )
        phi = np.clip(phi, 0.0, 0.35)

        idio = theta + shock - phi * previous_shock

        # --- assemble the total return from factors plus idiosyncratic ---
        returns[day] = (
            betas * market_factor[day]
            + sector_loadings * sector_factor[day, sector_ids]
            + idio
        )

        # --- volume: persistent, and heavier when the move is large ---
        volume_state = (
            config.volume_persistence * volume_state
            + config.volume_return_sensitivity
            * (np.abs(idio) / np.maximum(volatility, 1e-9) - 0.8)
            + rng.normal(0.0, config.volume_noise, size=n)
        )
        log_volume[day] = base_log_volume + volume_state

        previous_shock = shock

    # ------------------------------------------------------------------
    # Prices and OHLC bars
    # ------------------------------------------------------------------
    initial_price = rng.uniform(
        config.initial_price_low, config.initial_price_high, size=n
    )
    close = initial_price * np.exp(np.cumsum(returns, axis=0))

    previous_close = np.vstack([initial_price, close[:-1]])
    # The overnight move carries a share of the day's return plus its own
    # independent news shock, so gap and intraday are distinct signals rather
    # than two rescalings of the same number.
    overnight_shock = rng.normal(
        0.0, 1.0, size=(t, n)
    ) * base_idio_vol * config.overnight_noise_share
    gap = returns * config.overnight_gap_share + overnight_shock
    open_ = previous_close * np.exp(gap)

    # Intraday range scaled to the size of the day's move, never inverted.
    daily_range = (
        np.abs(returns) * config.intraday_range_multiple
        + rng.gamma(2.0, 0.004, size=(t, n))
    )
    mid_high = np.maximum(open_, close)
    mid_low = np.minimum(open_, close)
    high = mid_high * np.exp(daily_range * rng.uniform(0.2, 0.8, size=(t, n)))
    low = mid_low * np.exp(-daily_range * rng.uniform(0.2, 0.8, size=(t, n)))

    volume = np.exp(log_volume)

    # ------------------------------------------------------------------
    # Sporadic missing data, as any real vendor feed has
    # ------------------------------------------------------------------
    if config.missing_data_rate > 0:
        holes = rng.random((t, n)) < config.missing_data_rate
        holes[:25] = False  # keep the warm-up period clean
        for array in (close, open_, high, low, volume):
            array[holes] = np.nan

    frames = {
        "open": pd.DataFrame(open_, index=dates, columns=tickers),
        "high": pd.DataFrame(high, index=dates, columns=tickers),
        "low": pd.DataFrame(low, index=dates, columns=tickers),
        "close": pd.DataFrame(close, index=dates, columns=tickers),
        "volume": pd.DataFrame(volume, index=dates, columns=tickers),
        "sector": pd.DataFrame(
            np.tile(np.array(SECTORS, dtype=object)[sector_ids], (t, 1)),
            index=dates,
            columns=tickers,
        ),
    }

    return MarketPanel.from_frames(
        frames, source=f"synthetic(seed={config.seed})", derive=True
    )


def planted_effects(config: Optional[MarketConfig] = None) -> dict:
    """Describe what was planted, for tests and for the report to check against.

    Having the simulator state its own ground truth means the validation test
    cannot silently drift away from the generator.
    """
    config = config or MarketConfig()
    return {
        "short_term_reversal": {
            "present": config.reversal_strength > 0,
            "mechanism": "MA(1) with negative coefficient on yesterday's shock",
            "base_coefficient": config.reversal_strength,
            "expected_sign_of_ic": -1,  # ts_delta(close, 1) should predict negatively
        },
        "volume_conditioned_reversal": {
            "present": config.reversal_volume_sensitivity > 0,
            "mechanism": "reversal coefficient rises with yesterday's volume shock",
            "sensitivity": config.reversal_volume_sensitivity,
        },
        "momentum": {
            "present": config.momentum_scale > 0,
            "mechanism": "persistent AR(1) expected-return component",
            "persistence": config.momentum_persistence,
            "expected_sign_of_ic": +1,
        },
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_tickers(n: int) -> List[str]:
    """Deterministic pseudo-tickers: SIM0001 ... SIMnnnn."""
    width = max(4, len(str(n)))
    return [f"SIM{i:0{width}d}" for i in range(1, n + 1)]


def _t_scale(degrees_of_freedom: float) -> float:
    """Standard deviation of a Student-t, used to rescale to unit variance.

    Without this a t(4) draw would carry roughly 1.41x the intended volatility,
    so the configured ``idio_vol_mean`` would not mean what it says.
    """
    if degrees_of_freedom <= 2:
        raise ValueError("t degrees of freedom must exceed 2 for finite variance")
    return float(np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0)))
