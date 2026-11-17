"""Central registry mapping operator names to their implementations.

Every operator the expression language understands is registered here through
the :func:`register` decorator.  Keeping one registry means that:

* the evaluator has a single lookup path and no ``if name == ...`` ladder;
* ``alphalab ops`` can print self-documenting help straight from the registry;
* adding an operator is a single decorated function, with no other file to
  touch.

Operators receive already-evaluated arguments.  A numeric literal in a formula
arrives as a plain ``float`` and a data field arrives as a ``pandas.DataFrame``
indexed by date with one column per instrument, so an operator never has to
know anything about the AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


class OperatorError(KeyError):
    """Raised when a formula references an operator that does not exist."""


class ArityError(ValueError):
    """Raised when an operator is called with the wrong number of arguments."""


@dataclass(frozen=True)
class OperatorSpec:
    """Everything the engine knows about one operator."""

    name: str
    fn: Callable
    min_args: int
    max_args: int
    category: str
    doc: str
    #: Indices of arguments that must be numeric literals in the source, e.g.
    #: the lookback window of ``ts_mean(close, 20)``.  Validated at parse time
    #: so that a bad formula fails before any data is touched.
    const_args: Tuple[int, ...] = ()

    def check_arity(self, n_args: int) -> None:
        if not (self.min_args <= n_args <= self.max_args):
            if self.min_args == self.max_args:
                expected = f"{self.min_args}"
            else:
                expected = f"{self.min_args} to {self.max_args}"
            raise ArityError(
                f"operator {self.name!r} takes {expected} arguments, "
                f"got {n_args}"
            )


_REGISTRY: Dict[str, OperatorSpec] = {}


def register(
    name: str,
    *,
    category: str,
    n_args: Optional[int] = None,
    min_args: Optional[int] = None,
    max_args: Optional[int] = None,
    const_args: Tuple[int, ...] = (),
) -> Callable:
    """Decorator that adds a function to the operator registry.

    Either pass ``n_args`` for a fixed arity, or ``min_args``/``max_args`` for
    an operator with optional parameters.
    """
    if n_args is not None:
        lo = hi = n_args
    else:
        if min_args is None or max_args is None:
            raise ValueError("provide n_args, or both min_args and max_args")
        lo, hi = min_args, max_args

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"operator {name!r} is already registered")
        _REGISTRY[name] = OperatorSpec(
            name=name,
            fn=fn,
            min_args=lo,
            max_args=hi,
            category=category,
            doc=(fn.__doc__ or "").strip().splitlines()[0]
            if fn.__doc__
            else "",
            const_args=const_args,
        )
        return fn

    return decorator


def get(name: str) -> OperatorSpec:
    """Look up an operator, raising a helpful error if it is unknown."""
    try:
        return _REGISTRY[name]
    except KeyError:
        close = _suggest(name)
        hint = f"; did you mean {close!r}?" if close else ""
        raise OperatorError(f"unknown operator {name!r}{hint}") from None


def exists(name: str) -> bool:
    return name in _REGISTRY


def all_specs() -> Dict[str, OperatorSpec]:
    """Return a copy of the registry, sorted by category then name."""
    return dict(
        sorted(_REGISTRY.items(), key=lambda kv: (kv[1].category, kv[0]))
    )


def count() -> int:
    return len(_REGISTRY)


def _suggest(name: str) -> Optional[str]:
    """Cheap nearest-name hint for typos, using difflib."""
    import difflib

    matches = difflib.get_close_matches(name, _REGISTRY.keys(), n=1, cutoff=0.7)
    return matches[0] if matches else None
