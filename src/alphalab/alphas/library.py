"""The candidate alpha library: formulas to be tested, not formulas that work.

This is the most misunderstood file in the project, so it is worth being
explicit about what it is.

These ~120 formulas are *candidates*.  The overwhelming majority of them are
expected to be worthless.  That is not a flaw in the library, it is the entire
premise of the research process: you generate a wide set of economically
motivated ideas, measure them honestly, discard the ones that do not survive,
and then correct your final result for the fact that you looked at so many.
A library where every formula worked would be a library that had been fitted
to the sample.

Design notes
------------
*Economic motivation first.*  Each family below starts from a reason a signal
might predict returns -- investors overreact to news, liquidity providers
demand compensation for absorbing one-sided flow, and so on -- rather than
from a search over random formula shapes.  Families are then swept over
plausible lookback windows, because the economically right horizon is rarely
known in advance.

*Shared building blocks are deliberate.*  The constants below (daily returns,
twenty-day volume, trailing volatility) appear across many families, exactly as
they would in a real research group where everyone builds on the same
primitives.  That redundancy is what the sub-expression cache exploits, and
``alphalab plan`` will show it: roughly half the node evaluations in this
library are repeats.

*Sector-neutral variants are separate entries.*  Whether a signal survives
sector neutralisation is an empirical question, and often the neutral version
is the only one with any content, so both go into the pool and the screen
decides.
"""

from __future__ import annotations

from typing import Dict, List

# ----------------------------------------------------------------------
# Shared building blocks.
#
# Defining these once, as strings, is what creates the overlapping structure
# the evaluator's cache is built to exploit: every formula that mentions
# DAILY_RETURN produces the identical sub-tree, hence the identical cache key.
# ----------------------------------------------------------------------
DAILY_RETURN = "ts_returns(close,1)"
VOLATILITY_20 = f"ts_std({DAILY_RETURN},20)"
VOLATILITY_60 = f"ts_std({DAILY_RETURN},60)"
VOLUME_MEAN_20 = "ts_mean(volume,20)"
VOLUME_SURPRISE = f"log(volume/{VOLUME_MEAN_20})"
DOLLAR_VOLUME_MEAN_20 = "ts_mean(dollar_volume,20)"
TRUE_RANGE = "(high-low)/close"
CLOSE_LOCATION = "(close-low)/(high-low)"
GAP = "(open-ts_delay(close,1))/ts_delay(close,1)"
INTRADAY_MOVE = "(close-open)/open"


def _reversal_family() -> Dict[str, str]:
    """Short-horizon reversal.

    Economic story: a burst of one-sided demand pushes a price away from fair
    value, and the liquidity providers who absorbed that flow are compensated
    as it comes back.  The effect is strongest over days, not months, and
    should be stronger when the move came on heavy volume (the flow was large)
    and in volatile names (the push moved the price further).
    """
    alphas: Dict[str, str] = {}

    for window in (1, 2, 3, 5, 10, 20):
        # Plain reversal: short what went up.
        alphas[f"rev_simple_{window}"] = (
            f"-rank(ts_returns(close,{window}))"
        )
        # Risk-adjusted: a 3% move in a quiet name is a bigger surprise than
        # the same move in a volatile one.
        alphas[f"rev_volscaled_{window}"] = (
            f"-rank(ts_returns(close,{window})/{VOLATILITY_20})"
        )

    for window in (1, 2, 3, 5):
        # Volume-conditioned: reversal should be stronger after heavy trading.
        alphas[f"rev_volume_{window}"] = (
            f"-rank(ts_returns(close,{window})*{VOLUME_SURPRISE})"
        )
        # Rank the two legs separately, then multiply: more robust to the
        # scale of either leg than multiplying the raw values.
        alphas[f"rev_volume_ranked_{window}"] = (
            f"-rank(ts_returns(close,{window}))*rank({VOLUME_SURPRISE})"
        )

    # Reversal measured against the stock's own recent range rather than a
    # fixed lookback return.
    for window in (5, 10, 20):
        alphas[f"rev_zscore_{window}"] = f"-rank(ts_zscore(close,{window}))"
        alphas[f"rev_tsrank_{window}"] = f"-rank(ts_rank(close,{window}))"
        alphas[f"rev_scaled_{window}"] = f"-rank(ts_scale(close,{window}))"

    # Overnight versus intraday: these are different flows and often reverse
    # on different horizons.
    alphas["rev_gap"] = f"-rank({GAP})"
    alphas["rev_intraday"] = f"-rank({INTRADAY_MOVE})"
    alphas["rev_gap_volscaled"] = f"-rank({GAP}/{VOLATILITY_20})"

    return alphas


def _momentum_family() -> Dict[str, str]:
    """Medium-horizon momentum.

    Economic story: information diffuses slowly and investors under-react, so
    trends persist over months.  The standard construction skips the most
    recent month because that horizon is dominated by the reversal effect
    above -- including it mixes two opposing signals and usually cancels both.
    """
    alphas: Dict[str, str] = {}

    # The classic 12-1: twelve-month return, skipping the last month.
    alphas["mom_12_1"] = (
        "rank(ts_delay(close,21)/ts_delay(close,252)-1)"
    )
    alphas["mom_6_1"] = "rank(ts_delay(close,21)/ts_delay(close,126)-1)"
    alphas["mom_3_1"] = "rank(ts_delay(close,21)/ts_delay(close,63)-1)"

    for window in (20, 40, 60, 120, 252):
        alphas[f"mom_total_{window}"] = f"rank(ts_returns(close,{window}))"
        # Sharpe-like momentum: trend divided by the noise around it.
        alphas[f"mom_sharpe_{window}"] = (
            f"rank(ts_mean({DAILY_RETURN},{window})/ts_std({DAILY_RETURN},{window}))"
        )

    # Trend measured as price versus its own moving average.
    for window in (20, 50, 200):
        alphas[f"mom_vs_ma_{window}"] = (
            f"rank(close/ts_mean(close,{window})-1)"
        )

    # Position within the recent range: near a 52-week high is a different
    # statement from a large trailing return.
    for window in (60, 252):
        alphas[f"mom_near_high_{window}"] = f"rank(close/ts_max(close,{window}))"
        alphas[f"mom_days_since_high_{window}"] = (
            f"-rank(ts_argmax(close,{window}))"
        )
        alphas[f"mom_days_since_low_{window}"] = (
            f"rank(ts_argmin(close,{window}))"
        )

    return alphas


def _volume_family() -> Dict[str, str]:
    """Volume, liquidity and turnover.

    Economic story: volume measures attention and the intensity of flow.
    Unusual volume marks a crowd arriving; persistently high turnover relative
    to size marks a name whose price is set by traders rather than by holders.
    """
    alphas: Dict[str, str] = {}

    for window in (5, 10, 20, 60):
        alphas[f"vol_surprise_{window}"] = (
            f"rank(volume/ts_mean(volume,{window}))"
        )
        alphas[f"vol_trend_{window}"] = f"rank(ts_delta(log(volume),{window}))"

    alphas["vol_surprise_log"] = f"rank({VOLUME_SURPRISE})"
    alphas["vol_surprise_z"] = "rank(ts_zscore(log(volume),60))"

    # Amihud illiquidity: how far the price moves per dollar traded.  Illiquid
    # names earn a premium for being hard to get out of.
    alphas["illiquidity_amihud"] = (
        f"rank(abs({DAILY_RETURN})/log(dollar_volume))"
    )
    alphas["illiquidity_amihud_20"] = (
        f"rank(ts_mean(abs({DAILY_RETURN})/log(dollar_volume),20))"
    )

    # Size, proxied by traded value.  Small names have historically earned
    # more, and also cost more to trade.
    alphas["size_small"] = f"-rank({DOLLAR_VOLUME_MEAN_20})"
    alphas["size_small_log"] = f"-rank(log({DOLLAR_VOLUME_MEAN_20}))"

    # Price-volume agreement.  A rising price on rising volume is a different
    # animal from a rising price on fading volume.
    for window in (10, 20, 60):
        alphas[f"pv_corr_{window}"] = (
            f"-rank(ts_corr(close,log(volume),{window}))"
        )
        alphas[f"pv_corr_returns_{window}"] = (
            f"-rank(ts_corr({DAILY_RETURN},log(volume),{window}))"
        )

    return alphas


def _volatility_family() -> Dict[str, str]:
    """Risk, and the low-volatility anomaly.

    Economic story: leverage-constrained investors reach for risk by buying
    volatile stocks rather than by levering safe ones, which bids up the
    volatile names and depresses their returns.  The documented effect is that
    *low* volatility outperforms on a risk-adjusted basis, so these are signed
    accordingly.
    """
    alphas: Dict[str, str] = {}

    for window in (20, 60, 120):
        alphas[f"lowvol_{window}"] = f"-rank(ts_std({DAILY_RETURN},{window}))"
        alphas[f"lowrange_{window}"] = f"-rank(ts_mean({TRUE_RANGE},{window}))"

    # Idiosyncratic volatility: the part of risk left after stripping out the
    # market, which is the part the anomaly is usually attributed to.
    alphas["lowvol_idio_60"] = (
        f"-rank(ts_std({DAILY_RETURN}-ts_beta({DAILY_RETURN},market,60)*market,60))"
    )
    alphas["beta_low_60"] = f"-rank(ts_beta({DAILY_RETURN},market,60))"
    alphas["beta_low_120"] = f"-rank(ts_beta({DAILY_RETURN},market,120))"

    # Volatility of volatility, and volatility trend: a regime change in risk
    # is itself informative.
    alphas["vol_of_vol_60"] = f"-rank(ts_std({VOLATILITY_20},60))"
    alphas["vol_trend_ratio"] = f"-rank({VOLATILITY_20}/{VOLATILITY_60})"

    # Higher moments: lottery-like stocks (positive skew, big maximum day)
    # are over-bought by retail investors and under-perform.
    alphas["skew_low_60"] = f"-rank(ts_skew({DAILY_RETURN},60))"
    alphas["kurt_low_60"] = f"-rank(ts_kurt({DAILY_RETURN},60))"
    alphas["max_return_lottery"] = f"-rank(ts_max({DAILY_RETURN},20))"

    return alphas


def _range_family() -> Dict[str, str]:
    """Where the close sits inside the day's range, and gap behaviour.

    Economic story: a close at the top of the day's range means buyers held
    control into the bell, which carries short-horizon information about
    whether the flow is finished.
    """
    alphas: Dict[str, str] = {}

    alphas["close_location"] = f"rank({CLOSE_LOCATION})"
    alphas["close_location_rev"] = f"-rank({CLOSE_LOCATION})"
    for window in (5, 10, 20):
        alphas[f"close_location_mean_{window}"] = (
            f"-rank(ts_mean({CLOSE_LOCATION},{window}))"
        )

    alphas["range_expansion"] = f"-rank({TRUE_RANGE}/ts_mean({TRUE_RANGE},20))"
    alphas["high_low_position_20"] = (
        "rank((close-ts_min(low,20))/(ts_max(high,20)-ts_min(low,20)))"
    )
    alphas["high_low_position_60_rev"] = (
        "-rank((close-ts_min(low,60))/(ts_max(high,60)-ts_min(low,60)))"
    )

    # Volume-weighted average price versus the close: a crude read on whether
    # the day's flow was accumulating or distributing.
    alphas["vwap_premium"] = "-rank(close/vwap-1)"
    alphas["vwap_premium_smoothed"] = (
        "-rank(decay_linear(close/vwap-1,5))"
    )

    return alphas


def _sector_neutral_family() -> Dict[str, str]:
    """Sector-neutral versions of the most promising raw signals.

    Economic story: half of what a raw cross-sectional signal picks up is
    usually just a sector tilt.  Removing the sector mean asks the sharper
    question -- is this name cheap *relative to its peers*?  Often the neutral
    version is the only one with any content left in it.
    """
    alphas: Dict[str, str] = {}

    neutral_targets = {
        "rev_simple_1": f"-ts_returns(close,1)",
        "rev_simple_5": f"-ts_returns(close,5)",
        "rev_volscaled_5": f"-ts_returns(close,5)/{VOLATILITY_20}",
        "rev_volume_1": f"-ts_returns(close,1)*{VOLUME_SURPRISE}",
        "mom_12_1": "ts_delay(close,21)/ts_delay(close,252)-1",
        "mom_total_60": "ts_returns(close,60)",
        "lowvol_60": f"-ts_std({DAILY_RETURN},60)",
        "vol_surprise": VOLUME_SURPRISE,
        "close_location": CLOSE_LOCATION,
        "illiquidity": f"abs({DAILY_RETURN})/log(dollar_volume)",
    }

    for name, expression in neutral_targets.items():
        alphas[f"sn_{name}"] = f"group_rank({expression},sector)-0.5"
        alphas[f"sd_{name}"] = f"rank(group_demean({expression},sector))"

    return alphas


def _smoothed_family() -> Dict[str, str]:
    """Turnover-reduced versions of fast signals.

    Economic story: none -- this family exists for a purely practical reason.
    A one-day reversal signal rebuilds the book every day, and after costs the
    best raw signal in the library can easily be the worst net of fees.
    Smoothing with a linear decay trades a little signal for a lot of
    turnover, and ``hump`` refuses to trade at all until the signal has moved
    enough to be worth the spread.

    Whether that trade is worth making is exactly what the cost-aware backtest
    in :mod:`alphalab.portfolio` is there to answer.
    """
    alphas: Dict[str, str] = {}

    base_signals = {
        "rev1": f"-rank(ts_returns(close,1))",
        "rev5": f"-rank(ts_returns(close,5))",
        "revvol1": f"-rank(ts_returns(close,1)*{VOLUME_SURPRISE})",
        "volsurp": f"rank({VOLUME_SURPRISE})",
    }

    for name, expression in base_signals.items():
        for window in (3, 5, 10):
            alphas[f"smooth_{name}_d{window}"] = (
                f"decay_linear({expression},{window})"
            )
        alphas[f"hump_{name}"] = f"hump({expression},0.05)"

    return alphas


def _composite_family() -> Dict[str, str]:
    """Hand-built interactions between families.

    Economic story: the families above are not independent.  Reversal is
    stronger in small, volatile, heavily-traded names, so the interaction can
    carry information that neither leg does alone.  These are included partly
    to give the correlation-pruning step something genuinely redundant to
    reject.
    """
    alphas: Dict[str, str] = {}

    alphas["combo_rev_small"] = (
        f"-rank(ts_returns(close,5))*(1-rank({DOLLAR_VOLUME_MEAN_20}))"
    )
    alphas["combo_rev_volatile"] = (
        f"-rank(ts_returns(close,5))*rank({VOLATILITY_20})"
    )
    # Note: this deliberately uses ts_rank (where the name sits in its *own*
    # recent volume history) rather than a cross-sectional rank of the volume
    # ratio.  An earlier version used rank(volume/ts_mean(volume,20)), which is
    # exactly identical to rev_volume_ranked_1 -- rank() is invariant to the
    # monotone log() that distinguished them -- and the correlation pruner duly
    # flagged the pair at 1.00.
    alphas["combo_rev_highvolume"] = (
        "-rank(ts_returns(close,1))*ts_rank(volume,20)"
    )
    alphas["combo_mom_lowvol"] = (
        f"rank(ts_returns(close,120))*(1-rank({VOLATILITY_60}))"
    )
    alphas["combo_mom_minus_rev"] = (
        "rank(ts_delay(close,21)/ts_delay(close,252)-1)"
        "-rank(ts_returns(close,5))"
    )

    # Conditional signals: trade reversal only in the names where the story
    # applies most strongly.  Exercises the ternary operator end to end.
    alphas["cond_rev_if_highvol"] = (
        f"{VOLUME_SURPRISE} > 0 ? -rank(ts_returns(close,1)) : 0*close"
    )
    alphas["cond_mom_if_trending"] = (
        f"close > ts_mean(close,200) ? rank(ts_returns(close,60)) : 0*close"
    )
    alphas["cond_rev_if_wide_range"] = (
        f"{TRUE_RANGE} > ts_mean({TRUE_RANGE},20) ? "
        f"-rank(ts_returns(close,1)) : 0*close"
    )

    return alphas


def build_library() -> Dict[str, str]:
    """Assemble the full candidate library as ``{name: formula}``."""
    library: Dict[str, str] = {}
    for family in (
        _reversal_family,
        _momentum_family,
        _volume_family,
        _volatility_family,
        _range_family,
        _sector_neutral_family,
        _smoothed_family,
        _composite_family,
    ):
        produced = family()
        overlap = set(produced) & set(library)
        if overlap:
            raise ValueError(f"duplicate alpha names: {sorted(overlap)}")
        library.update(produced)
    return library


#: The candidate library, built once at import time.
ALPHA_LIBRARY: Dict[str, str] = build_library()

#: Which family each alpha came from, for grouped reporting.
FAMILY_OF: Dict[str, str] = {}
for _family_name, _builder in (
    ("reversal", _reversal_family),
    ("momentum", _momentum_family),
    ("volume", _volume_family),
    ("volatility", _volatility_family),
    ("range", _range_family),
    ("sector_neutral", _sector_neutral_family),
    ("smoothed", _smoothed_family),
    ("composite", _composite_family),
):
    for _name in _builder():
        FAMILY_OF[_name] = _family_name


def family_names() -> List[str]:
    return sorted(set(FAMILY_OF.values()))


def alphas_in_family(family: str) -> Dict[str, str]:
    return {
        name: formula
        for name, formula in ALPHA_LIBRARY.items()
        if FAMILY_OF.get(name) == family
    }
