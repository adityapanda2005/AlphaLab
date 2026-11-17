"""The in-memory market panel that every other layer reads from.

A :class:`MarketPanel` is a mapping from field name to a (date x instrument)
DataFrame, with every field sharing one index and one column set.  Holding the
data this way -- rather than as a long "tidy" table of (date, instrument, field,
value) rows -- is a deliberate choice:

* cross-sectional operations become single ``axis=1`` calls instead of
  ``groupby(date)``, which is roughly two orders of magnitude faster;
* rolling time-series operations become single ``axis=0`` calls;
* the shape of the data mirrors the shape of the problem, so the code reads
  like the mathematics it implements.

The cost is memory: a dense panel stores a value for every instrument on every
date, including the ones that were not listed yet.  For a daily equity universe
in the low thousands that cost is trivial, and the ``available`` mask records
which cells were genuinely tradeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional

import numpy as np
import pandas as pd

#: Fields a complete price panel is expected to carry.
CORE_FIELDS = ("open", "high", "low", "close", "volume")

#: Fields the panel derives from the core ones when they are not supplied.
DERIVED_FIELDS = ("returns", "vwap", "dollar_volume", "adv20", "market")


class PanelError(ValueError):
    """Raised when the supplied frames do not form a consistent panel."""


@dataclass
class MarketPanel(Mapping):
    """A dict-like bundle of aligned (date x instrument) frames.

    Implements :class:`collections.abc.Mapping`, so a panel can be handed
    straight to :class:`~alphalab.expr.evaluator.Evaluator` as its data source.
    """

    fields: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Optional human-readable note about where the data came from.
    source: str = "unknown"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        *,
        source: str = "unknown",
        derive: bool = True,
    ) -> "MarketPanel":
        """Build a panel, aligning every frame and deriving the usual extras."""
        if not frames:
            raise PanelError("a panel needs at least one field")

        cleaned = {
            name: _standardise(name, frame) for name, frame in frames.items()
        }
        index, columns = _common_axes(cleaned)
        aligned = {
            name: frame.reindex(index=index, columns=columns)
            for name, frame in cleaned.items()
        }

        panel = cls(fields=aligned, source=source)
        if derive:
            panel.add_derived_fields()
        return panel

    def add_derived_fields(self) -> "MarketPanel":
        """Fill in returns, vwap, dollar volume, ADV and the market return.

        Each is only computed if it is missing, so a loader that already has a
        true VWAP from the exchange keeps it rather than having it overwritten
        by the ``(h+l+c)/3`` approximation.
        """
        if "close" in self.fields:
            close = self.fields["close"]

            if "returns" not in self.fields:
                # One-day simple return.  This is a *backward* looking field:
                # returns.loc[t] uses close[t] and close[t-1] only.
                self.fields["returns"] = close.pct_change(fill_method=None)

            if "vwap" not in self.fields and {"high", "low"} <= self.fields.keys():
                # Typical price: a reasonable stand-in when true VWAP is absent.
                self.fields["vwap"] = (
                    self.fields["high"] + self.fields["low"] + close
                ) / 3.0

            if "volume" in self.fields:
                if "dollar_volume" not in self.fields:
                    self.fields["dollar_volume"] = (
                        self.fields["volume"] * close
                    )
                if "adv20" not in self.fields:
                    self.fields["adv20"] = (
                        self.fields["dollar_volume"]
                        .rolling(20, min_periods=20)
                        .mean()
                    )

            if "market" not in self.fields and "returns" in self.fields:
                # Equal-weighted market return, broadcast to every column, so
                # formulas can write ts_beta(returns, market, 60).
                breadth = self.fields["returns"].mean(axis=1)
                self.fields["market"] = pd.DataFrame(
                    np.repeat(breadth.to_numpy()[:, None], close.shape[1], axis=1),
                    index=close.index,
                    columns=close.columns,
                )

        return self

    # ------------------------------------------------------------------
    # Mapping protocol -- lets the Evaluator treat a panel as its data dict
    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.fields[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def dates(self) -> pd.Index:
        return next(iter(self.fields.values())).index

    @property
    def instruments(self) -> pd.Index:
        return next(iter(self.fields.values())).columns

    @property
    def shape(self) -> tuple:
        return (len(self.dates), len(self.instruments))

    @property
    def field_names(self) -> List[str]:
        return sorted(self.fields.keys())

    def forward_returns(self, horizon: int = 1) -> pd.DataFrame:
        """The return earned by holding from ``t`` to ``t + horizon``.

        This is the prediction target, and it is the one place in the codebase
        where a *negative* shift is correct: ``shift(-horizon)`` deliberately
        pulls the future back to the present so it can be lined up against a
        signal known at ``t``.  Everything downstream must therefore be careful
        that a model trained on these labels never sees a label whose window
        overlaps its test period -- which is what the purge-and-embargo
        splitter in :mod:`alphalab.ml.splits` exists to guarantee.
        """
        if horizon < 1:
            raise PanelError(f"horizon must be >= 1, got {horizon}")
        close = self.fields["close"]
        return (close.shift(-horizon) / close - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )

    def liquid_universe(
        self,
        *,
        min_adv: float = 0.0,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """A boolean panel marking which names are tradeable on each date.

        Built from ADV known *at* ``t`` (a trailing twenty-day average), never
        from future volume, so applying it introduces no look-ahead.
        """
        adv = self.fields.get("adv20")
        if adv is None:
            raise PanelError("liquid_universe needs an 'adv20' field")

        mask = adv.notna()
        if min_adv > 0:
            mask &= adv >= min_adv
        if top_n is not None:
            ranked = adv.rank(axis=1, ascending=False, method="first")
            mask &= ranked <= top_n
        return mask

    def subset(self, fields: List[str]) -> "MarketPanel":
        missing = [name for name in fields if name not in self.fields]
        if missing:
            raise PanelError(f"panel has no field(s): {', '.join(missing)}")
        return MarketPanel(
            fields={name: self.fields[name] for name in fields},
            source=self.source,
        )

    def describe(self) -> str:
        n_dates, n_instruments = self.shape
        start, end = self.dates[0].date(), self.dates[-1].date()
        coverage = float(self.fields["close"].notna().to_numpy().mean()) \
            if "close" in self.fields else float("nan")
        return (
            f"MarketPanel({self.source}): {n_dates} dates x {n_instruments} "
            f"instruments, {start} to {end}, "
            f"{len(self.fields)} fields, {coverage:.1%} close coverage"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_parquet(self, directory) -> None:
        """Write one parquet file per field, plus a small manifest."""
        import json
        from pathlib import Path

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, frame in self.fields.items():
            frame.to_parquet(path / f"{name}.parquet")
        (path / "manifest.json").write_text(
            json.dumps(
                {"source": self.source, "fields": sorted(self.fields)},
                indent=2,
            )
        )

    @classmethod
    def from_parquet(cls, directory) -> "MarketPanel":
        import json
        from pathlib import Path

        path = Path(directory)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise PanelError(f"no panel manifest found in {path}")
        manifest = json.loads(manifest_path.read_text())

        frames = {
            name: pd.read_parquet(path / f"{name}.parquet")
            for name in manifest["fields"]
        }
        return cls(fields=frames, source=manifest.get("source", str(path)))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _standardise(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise PanelError(f"field {name!r} must be a DataFrame")
    out = frame.sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    # Sector labels stay as objects; everything else is numeric.
    if name != "sector":
        out = out.astype(float)
    return out


def _common_axes(frames: Mapping[str, pd.DataFrame]):
    """Intersect dates and union instruments across every supplied field."""
    iterator = iter(frames.values())
    first = next(iterator)
    index = first.index
    columns = first.columns

    for frame in iterator:
        index = index.intersection(frame.index)
        columns = columns.union(frame.columns)

    if len(index) == 0:
        raise PanelError("fields share no common dates")
    return index.sort_values(), columns.sort_values()
