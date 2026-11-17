"""Memoisation of evaluated sub-expressions.

Why this exists
---------------
An alpha library is not a set of unrelated formulas.  Researchers build on each
other's ideas, so the same fragments appear again and again::

    rank(ts_delta(close, 5)) / ts_std(returns, 20)
    rank(ts_delta(close, 5)) * rank(volume / ts_mean(volume, 20))
    ts_corr(rank(ts_delta(close, 5)), rank(returns), 10)

Evaluated naively, ``rank(ts_delta(close, 5))`` is computed three times, and a
rolling operation over a (dates x instruments) panel is not cheap.  Across a
library of a hundred formulas the duplication compounds badly.

This is the classic *common sub-expression elimination* problem from compilers,
and the classic fix applies: give every sub-tree a canonical name, compute each
distinct name once, and reuse the result.  The canonical name is
:attr:`~alphalab.expr.ast_nodes.Node.key`, which is identical for structurally
identical sub-trees, so the cache is correct by construction rather than by
hoping two formulas were typed the same way.

``scripts/benchmark_cache.py`` measures the resulting speedup end to end.

Safety invariant
----------------
Cached values are handed out by reference, not copied, because copying a panel
would undo the saving.  That is safe only because every operator in
:mod:`alphalab.expr.operators` returns a new object and never mutates its
inputs.  ``tests/test_cache.py`` asserts this invariant directly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


class _Miss:
    """Sentinel distinguishing "absent" from a cached ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<cache miss>"


MISS = _Miss()


@dataclass(frozen=True)
class CacheStats:
    """A snapshot of cache activity, used by tests and the benchmark report."""

    hits: int
    misses: int
    entries: int
    evictions: int
    bytes_held: int
    rejections: int = 0
    peak_bytes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def peak_megabytes(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def megabytes_held(self) -> float:
        return self.bytes_held / (1024 * 1024)

    def summary(self) -> str:
        return (
            f"{self.hits} hits / {self.lookups} lookups "
            f"({self.hit_rate:.1%}), {self.entries} entries, "
            f"{self.megabytes_held:.1f} MB held"
        )


class SubexpressionCache:
    """An LRU cache keyed by canonical sub-expression string.

    Admission policy
    ----------------
    Caching *every* node turned out to be a bad trade.  Measured on the
    bundled library, roughly 70% of distinct sub-expressions are referenced
    exactly once, so storing them held hundreds of megabytes of panels that
    were never read again.

    The fix is admission control: before evaluating a batch,
    :meth:`~alphalab.expr.evaluator.Evaluator.evaluate_many` counts how often
    each sub-expression key appears across the whole batch and admits only the
    keys seen more than once.  Single-use nodes still flow through the
    evaluator, they simply are not retained.  This keeps the full speedup --
    nothing reusable is dropped -- while cutting peak memory several-fold.
    ``scripts/benchmark_cache.py`` measures all three policies side by side.

    Parameters
    ----------
    max_entries:
        Evict the least recently used entry once this many are held.  ``None``
        means unbounded, which is the right default once admission control is
        doing the real work.
    admit:
        If given, only these keys are retained.  ``None`` means cache
        everything, which is the honest "naive memoisation" baseline.
    """

    def __init__(
        self,
        max_entries: Optional[int] = None,
        admit: Optional[set] = None,
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 or None")
        self._max_entries = max_entries
        self._admit = admit
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rejections = 0
        # Memory is tracked incrementally rather than recomputed on demand.
        # Summing every entry's footprint on each insert turns the cache into
        # an accidental O(n^2) loop, which showed up immediately as a
        # pathological slowdown once the library grew past a hundred formulas.
        self._sizes: dict[str, int] = {}
        self._bytes_total = 0
        self._peak_bytes = 0

    # ------------------------------------------------------------------
    # Admission control
    # ------------------------------------------------------------------
    def set_admission(self, keys: Optional[set]) -> None:
        """Restrict what may be retained; ``None`` lifts the restriction."""
        self._admit = keys

    @property
    def admission(self) -> Optional[set]:
        return self._admit

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def get(self, key: str) -> Any:
        """Return the cached value for ``key``, or :data:`MISS`."""
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return MISS

    def put(self, key: str, value: Any) -> Any:
        """Store ``value`` under ``key`` and return it unchanged.

        Values rejected by the admission policy are returned untouched, so the
        caller cannot tell the difference and correctness never depends on
        whether a particular node was retained.
        """
        if self._admit is not None and key not in self._admit:
            self._rejections += 1
            return value

        if key in self._sizes:
            self._bytes_total -= self._sizes.pop(key)

        self._store[key] = value
        self._store.move_to_end(key)

        size = _sizeof(value)
        self._sizes[key] = size
        self._bytes_total += size

        if self._max_entries is not None:
            while len(self._store) > self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                self._bytes_total -= self._sizes.pop(evicted_key, 0)
                self._evictions += 1

        if self._bytes_total > self._peak_bytes:
            self._peak_bytes = self._bytes_total
        return value

    def clear(self) -> None:
        """Drop every entry but keep the accumulated counters."""
        self._store.clear()
        self._sizes.clear()
        self._bytes_total = 0

    def reset_stats(self) -> None:
        self._hits = self._misses = self._evictions = self._rejections = 0
        self._peak_bytes = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            entries=len(self._store),
            evictions=self._evictions,
            bytes_held=self._bytes_total,
            rejections=self._rejections,
            peak_bytes=max(self._peak_bytes, self._bytes_total),
        )

    def keys(self):
        """Cached keys, least recently used first."""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SubexpressionCache({self.stats.summary()})"


def _sizeof(value: Any) -> int:
    """Approximate in-memory footprint of a cached value, in bytes.

    Uses the block values rather than ``memory_usage(deep=True)`` because the
    panels are numeric and this needs to be cheap: it runs on every insert.
    """
    if isinstance(value, pd.DataFrame):
        return int(value.shape[0]) * int(value.shape[1]) * 8
    if isinstance(value, pd.Series):
        return int(value.shape[0]) * 8
    return 8


class NullCache(SubexpressionCache):
    """A cache that never retains anything.

    Used by the benchmark as the honest "no optimisation" baseline, and by
    tests that need to prove two code paths agree.  Subclassing keeps the
    evaluator free of ``if self._cache is not None`` branches.
    """

    def __init__(self) -> None:
        super().__init__(max_entries=None)

    def get(self, key: str) -> Any:
        self._misses += 1
        return MISS

    def put(self, key: str, value: Any) -> Any:
        return value
