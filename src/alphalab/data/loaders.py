"""Loading market data: real vendor prices, or the bundled simulator.

``load_panel`` is the single entry point the pipeline uses.  It takes a source
name and returns a :class:`~alphalab.data.panel.MarketPanel`, handling the
parquet cache so that a second run does no network I/O at all.

On data sources
---------------
The default source is ``synthetic`` because it needs no network, is identical
on every machine, and comes with ground truth (see
:mod:`alphalab.data.synthetic`).  The ``yfinance`` source fetches real daily
bars and is the right choice when you want to see how the stack behaves on
real prices.

Real free data carries caveats that are stated loudly rather than buried,
because pretending otherwise is how backtests end up lying:

* **Survivorship bias.**  A universe built from today's index membership omits
  every company that was delisted, acquired or dropped from the index.  Those
  are disproportionately the losers, so any backtest on such a universe is
  biased upward.  Fixing it properly needs point-in-time constituent history,
  which free sources do not provide.
* **No point-in-time fundamentals.**  Only prices and volumes are available
  here, so the alpha library is restricted to price and volume signals.
* **Adjusted prices are restated.**  Vendors apply today's split and dividend
  adjustments to the whole history, so a backtest sees prices nobody could
  have observed at the time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .panel import MarketPanel
from .synthetic import MarketConfig, generate_market

logger = logging.getLogger(__name__)

#: Where cached panels are written by default.
DEFAULT_CACHE_DIR = Path("data/cache")

VALID_SOURCES = ("synthetic", "yfinance", "csv")


class DataSourceError(RuntimeError):
    """Raised when a data source cannot produce a usable panel."""


def load_panel(
    source: str = "synthetic",
    *,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    config: Optional[MarketConfig] = None,
    tickers: Optional[Sequence[str]] = None,
    start: str = "2014-01-01",
    end: Optional[str] = None,
    csv_dir: Optional[Path] = None,
) -> MarketPanel:
    """Load a market panel from ``source``, using the parquet cache if present.

    Parameters
    ----------
    source:
        ``"synthetic"`` (default), ``"yfinance"`` or ``"csv"``.
    refresh:
        Ignore any cached copy and rebuild from the underlying source.
    """
    if source not in VALID_SOURCES:
        raise DataSourceError(
            f"unknown source {source!r}; expected one of {VALID_SOURCES}"
        )

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path = cache_dir / _cache_key(source, config, tickers, start, end)

    if not refresh and (cache_path / "manifest.json").exists():
        logger.info("loading cached panel from %s", cache_path)
        return MarketPanel.from_parquet(cache_path)

    if source == "synthetic":
        panel = generate_market(config or MarketConfig())
    elif source == "yfinance":
        panel = _load_yfinance(tickers, start, end)
    else:
        panel = _load_csv(csv_dir)

    try:
        panel.to_parquet(cache_path)
        logger.info("cached panel at %s", cache_path)
    except OSError as exc:  # pragma: no cover - cache is best effort
        logger.warning("could not write cache: %s", exc)

    return panel


# ----------------------------------------------------------------------
# yfinance
# ----------------------------------------------------------------------
def _load_yfinance(
    tickers: Optional[Sequence[str]],
    start: str,
    end: Optional[str],
) -> MarketPanel:
    """Download daily bars for ``tickers`` and assemble a panel.

    Requires network access and the optional ``yfinance`` dependency
    (``pip install -e ".[realdata]"``).
    """
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DataSourceError(
            "the yfinance source needs the optional dependency: "
            'pip install -e ".[realdata]"'
        ) from exc

    symbols = list(tickers) if tickers else list(DEFAULT_UNIVERSE)
    logger.info("downloading %d symbols from yfinance", len(symbols))

    raw = yfinance.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        raise DataSourceError(
            "yfinance returned no data. This usually means the network is "
            "unavailable or the symbols are wrong. Use source='synthetic' to "
            "run the pipeline without network access."
        )

    frames = {}
    for field, column in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if column not in raw.columns.get_level_values(0):
            raise DataSourceError(f"yfinance response is missing {column!r}")
        frame = raw[column].copy()
        frame.index = pd.to_datetime(frame.index)
        frames[field] = frame.astype(float)

    # Drop instruments with very little history: rolling windows cannot use
    # them and they only add NaN noise to cross-sectional statistics.
    close = frames["close"]
    keep = close.notna().sum() >= min(252, int(0.5 * len(close)))
    if not keep.any():
        raise DataSourceError("no symbol had enough history to be usable")
    for field in list(frames):
        frames[field] = frames[field].loc[:, keep[keep].index]

    frames["sector"] = _sector_frame(frames["close"])

    panel = MarketPanel.from_frames(
        frames, source=f"yfinance({len(frames['close'].columns)} symbols)"
    )
    logger.info("%s", panel.describe())
    return panel


def _sector_frame(close: pd.DataFrame) -> pd.DataFrame:
    """Attach sector labels, falling back to a single bucket if unknown.

    Sector lookups are a per-symbol network call in yfinance, so this uses the
    static map below and labels anything unmapped ``Unknown``.  A single
    bucket still works -- ``group_demean`` then simply reduces to ``demean``.
    """
    labels = [SECTOR_MAP.get(symbol, "Unknown") for symbol in close.columns]
    return pd.DataFrame(
        np.tile(np.array(labels, dtype=object), (len(close.index), 1)),
        index=close.index,
        columns=close.columns,
    )


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------
def _load_csv(csv_dir: Optional[Path]) -> MarketPanel:
    """Load a panel from one CSV per field, each dates x instruments.

    The escape hatch for your own data: export ``close.csv``, ``volume.csv``
    and friends with dates down the first column and tickers across the header.
    """
    if csv_dir is None:
        raise DataSourceError("source='csv' needs csv_dir")

    directory = Path(csv_dir)
    if not directory.is_dir():
        raise DataSourceError(f"{directory} is not a directory")

    frames = {}
    for path in sorted(directory.glob("*.csv")):
        field = path.stem.lower()
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frames[field] = frame

    if "close" not in frames:
        raise DataSourceError(f"no close.csv found in {directory}")
    if "sector" not in frames:
        frames["sector"] = _sector_frame(frames["close"])

    return MarketPanel.from_frames(frames, source=f"csv({directory})")


# ----------------------------------------------------------------------
# Cache keys and the default real-data universe
# ----------------------------------------------------------------------
def _cache_key(
    source: str,
    config: Optional[MarketConfig],
    tickers: Optional[Sequence[str]],
    start: str,
    end: Optional[str],
) -> str:
    """A short, deterministic directory name describing this request."""
    import hashlib

    if source == "synthetic":
        config = config or MarketConfig()
        payload = repr(sorted(config.__dict__.items()))
        digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
        return f"synthetic-{config.n_instruments}x{config.n_days}-{digest}"

    payload = repr((sorted(tickers) if tickers else "default", start, end))
    digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
    return f"{source}-{digest}"


#: A liquid, sector-diverse large-cap universe for the yfinance source.
#: Current index membership, so see the survivorship-bias warning above.
DEFAULT_UNIVERSE: tuple = (
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO",
    "ACN", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU",
    # Communication services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "GM",
    # Consumer staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "TGT",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "AXP",
    "BLK", "C", "SPGI", "CB", "PGR",
    # Health care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "CVS", "MDT", "GILD", "ISRG",
    # Industrials
    "CAT", "RTX", "HON", "UNP", "BA", "GE", "DE", "LMT", "UPS", "ADP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    # Utilities, real estate, materials
    "NEE", "DUK", "SO", "D", "AEP", "AMT", "PLD", "EQIX", "LIN", "APD",
    "SHW", "FCX", "NEM",
)

#: Static sector map for the default universe.
SECTOR_MAP = {}
for _symbols, _sector in (
    (("AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO",
      "ACN", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU"),
     "Technology"),
    (("GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS"),
     "CommunicationServices"),
    (("AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "GM"),
     "ConsumerDiscretionary"),
    (("WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "TGT"),
     "ConsumerStaples"),
    (("BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "AXP",
      "BLK", "C", "SPGI", "CB", "PGR"),
     "Financials"),
    (("UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
      "BMY", "CVS", "MDT", "GILD", "ISRG"),
     "HealthCare"),
    (("CAT", "RTX", "HON", "UNP", "BA", "GE", "DE", "LMT", "UPS", "ADP"),
     "Industrials"),
    (("XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY"),
     "Energy"),
    (("NEE", "DUK", "SO", "D", "AEP"), "Utilities"),
    (("AMT", "PLD", "EQIX"), "RealEstate"),
    (("LIN", "APD", "SHW", "FCX", "NEM"), "Materials"),
):
    for _symbol in _symbols:
        SECTOR_MAP[_symbol] = _sector
