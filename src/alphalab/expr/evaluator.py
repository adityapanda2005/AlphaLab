"""Evaluate alpha expression trees against a panel of market data.

The evaluator is a straightforward post-order walk of the AST with one
addition: before computing a node it asks the cache whether an identical
sub-tree has already been evaluated.  See :mod:`alphalab.expr.cache` for why
that matters and how correctness is guaranteed.

Value model
-----------
Exactly two kinds of value flow through the tree:

``float``
    A numeric literal, e.g. the ``20`` in ``ts_mean(close, 20)``.
``pandas.DataFrame``
    A panel indexed by date with one column per instrument.

Operators therefore never need to care about the AST, and the engine never
needs to care about what an operator does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import operators as ops
from .ast_nodes import BinOp, Call, Node, Num, UnaryOp, Var
from .cache import MISS, SubexpressionCache
from .parser import parse


class EvaluationError(RuntimeError):
    """Raised when a formula cannot be evaluated against the given data."""


class UnknownFieldError(EvaluationError):
    """Raised when a formula references a data field that was not supplied."""


@dataclass
class EvaluationReport:
    """Timing and cache accounting for one :meth:`Evaluator.evaluate_many` call."""

    n_expressions: int = 0
    n_nodes_total: int = 0
    n_nodes_distinct: int = 0
    n_nodes_shared: int = 0
    seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0
    cache_peak_megabytes: float = 0.0
    failures: Dict[str, str] = field(default_factory=dict)

    @property
    def reuse_ratio(self) -> float:
        """How many node evaluations were saved, as a fraction of the total.

        1 - distinct/total.  A library of completely unrelated formulas scores
        0; a library where every formula shares its building blocks scores
        close to 1.
        """
        if not self.n_nodes_total:
            return 0.0
        return 1.0 - (self.n_nodes_distinct / self.n_nodes_total)

    def summary(self) -> str:
        return (
            f"{self.n_expressions} expressions, "
            f"{self.n_nodes_distinct}/{self.n_nodes_total} distinct nodes "
            f"({self.reuse_ratio:.1%} reused), {self.seconds:.3f}s"
        )


class Evaluator:
    """Turns expression trees into signal panels.

    Parameters
    ----------
    data:
        Mapping from field name (``close``, ``volume``, ``sector`` ...) to a
        (date x instrument) DataFrame.  Usually a
        :class:`~alphalab.data.panel.MarketPanel`, which is such a mapping.
    cache:
        Sub-expression cache.  Pass :class:`~alphalab.expr.cache.NullCache` to
        disable reuse, which is what the benchmark baseline does.
    """

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        cache: Optional[SubexpressionCache] = None,
    ) -> None:
        self._data = data
        self._cache = cache if cache is not None else SubexpressionCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def cache(self) -> SubexpressionCache:
        return self._cache

    def evaluate(self, expression) -> pd.DataFrame:
        """Evaluate one formula (a string or an AST) into a signal panel."""
        node = parse(expression) if isinstance(expression, str) else expression
        validate(node)
        value = self._eval_node(node)
        if not isinstance(value, pd.DataFrame):
            raise EvaluationError(
                f"formula {node.key!r} evaluated to a scalar, not a panel; "
                "an alpha must produce one value per instrument per date"
            )
        return value

    def evaluate_many(
        self,
        expressions: Mapping[str, str] | Sequence[str],
        *,
        skip_failures: bool = True,
        admission: str = "shared",
    ) -> Tuple[Dict[str, pd.DataFrame], EvaluationReport]:
        """Evaluate a whole library in one pass, sharing the cache.

        Evaluating together rather than one call at a time is the entire point
        of the cache: fragments shared between formulas are computed once for
        the batch.

        Parameters
        ----------
        admission:
            ``"shared"`` (default) first counts how often each sub-expression
            occurs across the batch and retains only those occurring more than
            once.  ``"all"`` retains every node, which is naive memoisation and
            holds several times the memory for the same speedup.  See
            :class:`~alphalab.expr.cache.SubexpressionCache`.

        Returns the signals plus an :class:`EvaluationReport`.  By default a
        formula that fails is recorded in ``report.failures`` and skipped, so
        one bad entry cannot abort a long research run.
        """
        if admission not in ("shared", "all"):
            raise ValueError("admission must be 'shared' or 'all'")
        items = (
            list(expressions.items())
            if isinstance(expressions, Mapping)
            else [(src, src) for src in expressions]
        )

        parsed: List[Tuple[str, Node]] = []
        report = EvaluationReport(n_expressions=len(items))

        for name, source in items:
            try:
                node = parse(source) if isinstance(source, str) else source
                validate(node)
                parsed.append((name, node))
            except Exception as exc:
                if not skip_failures:
                    raise
                report.failures[name] = f"{type(exc).__name__}: {exc}"

        # One pass over the batch to find out what is worth keeping.  This is
        # cheap -- it walks the trees, not the data -- and it is what lets the
        # cache hold only the nodes that will actually be read again.
        use_counts: Dict[str, int] = {}
        for _, node in parsed:
            for sub in node.walk():
                if not isinstance(sub, Num):
                    use_counts[sub.key] = use_counts.get(sub.key, 0) + 1

        report.n_nodes_total = sum(node.size() for _, node in parsed)
        report.n_nodes_distinct = len(use_counts)
        report.n_nodes_shared = sum(1 for n in use_counts.values() if n > 1)

        previous_admission = self._cache.admission
        if admission == "shared":
            self._cache.set_admission(
                {key for key, n in use_counts.items() if n > 1}
            )
        else:
            self._cache.set_admission(None)

        hits_before = self._cache.stats.hits
        misses_before = self._cache.stats.misses

        signals: Dict[str, pd.DataFrame] = {}
        started = time.perf_counter()
        try:
            for name, node in parsed:
                try:
                    value = self._eval_node(node)
                    if not isinstance(value, pd.DataFrame):
                        raise EvaluationError(
                            f"{name!r} evaluated to a scalar, not a panel"
                        )
                    signals[name] = value
                except Exception as exc:
                    if not skip_failures:
                        raise
                    report.failures[name] = f"{type(exc).__name__}: {exc}"
        finally:
            report.seconds = time.perf_counter() - started
            stats = self._cache.stats
            report.cache_hits = stats.hits - hits_before
            report.cache_misses = stats.misses - misses_before
            report.cache_entries = stats.entries
            report.cache_peak_megabytes = stats.peak_megabytes
            self._cache.set_admission(previous_admission)

        return signals, report

    # ------------------------------------------------------------------
    # Recursive walk
    # ------------------------------------------------------------------
    def _eval_node(self, node: Node):
        # Literals are too cheap to be worth a dictionary lookup.
        if isinstance(node, Num):
            return node.value

        key = node.key
        cached = self._cache.get(key)
        if cached is not MISS:
            return cached

        if isinstance(node, Var):
            value = self._lookup_field(node.name)
        elif isinstance(node, UnaryOp):
            name = ops.UNARY_OPERATOR_NAMES[node.op]
            value = self._call(name, [self._eval_node(node.operand)])
        elif isinstance(node, BinOp):
            name = ops.BINARY_OPERATOR_NAMES[node.op]
            value = self._call(
                name,
                [self._eval_node(node.left), self._eval_node(node.right)],
            )
        elif isinstance(node, Call):
            args = [self._eval_node(arg) for arg in node.args]
            value = self._call(node.name, args)
        else:  # pragma: no cover - guards against a new node type
            raise EvaluationError(f"cannot evaluate node type {type(node).__name__}")

        return self._cache.put(key, value)

    def _call(self, name: str, args: List):
        spec = ops.get(name)
        try:
            return spec.fn(*args)
        except Exception as exc:
            raise EvaluationError(
                f"operator {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _lookup_field(self, name: str) -> pd.DataFrame:
        try:
            value = self._data[name]
        except KeyError:
            available = ", ".join(sorted(self._data.keys()))
            raise UnknownFieldError(
                f"unknown data field {name!r}; available fields: {available}"
            ) from None
        if not isinstance(value, pd.DataFrame):
            raise EvaluationError(
                f"data field {name!r} must be a DataFrame, "
                f"got {type(value).__name__}"
            )
        return value


# ----------------------------------------------------------------------
# Static validation
# ----------------------------------------------------------------------
def validate(node: Node) -> None:
    """Check a tree before any data is touched.

    Catches unknown operators, wrong argument counts and non-literal window
    arguments.  Running this first means a typo in formula 87 of the library
    surfaces immediately rather than after eighty-six panels have been
    computed.
    """
    for sub in node.walk():
        if isinstance(sub, Call):
            spec = ops.get(sub.name)
            spec.check_arity(len(sub.args))
            for index in spec.const_args:
                if index < len(sub.args) and not isinstance(sub.args[index], Num):
                    raise ops.ArityError(
                        f"argument {index + 1} of {sub.name!r} must be a number "
                        f"literal (a lookback window), got {sub.args[index].key!r}"
                    )
        elif isinstance(sub, BinOp):
            if sub.op not in ops.BINARY_OPERATOR_NAMES:
                raise ops.OperatorError(f"unknown infix operator {sub.op!r}")
        elif isinstance(sub, UnaryOp):
            if sub.op not in ops.UNARY_OPERATOR_NAMES:
                raise ops.OperatorError(f"unknown prefix operator {sub.op!r}")


def required_fields(node: Node) -> List[str]:
    """Every raw data field a tree reads, sorted and de-duplicated."""
    return sorted({sub.name for sub in node.walk() if isinstance(sub, Var)})


def shared_subexpressions(
    expressions: Iterable[str], *, min_uses: int = 2
) -> List[Tuple[str, int]]:
    """Report sub-expressions appearing in more than one place.

    Used by ``alphalab plan`` and by the README to show concretely where the
    cache earns its keep.  Returns ``(key, use_count)`` ordered by count.
    """
    counts: Dict[str, int] = {}
    for source in expressions:
        node = parse(source) if isinstance(source, str) else source
        for sub in node.walk():
            if isinstance(sub, Num):
                continue
            counts[sub.key] = counts.get(sub.key, 0) + 1

    shared = [(key, n) for key, n in counts.items() if n >= min_uses]
    shared.sort(key=lambda item: (-item[1], len(item[0])))
    return shared
