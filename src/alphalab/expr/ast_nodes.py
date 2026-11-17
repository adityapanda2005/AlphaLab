"""Abstract syntax tree nodes for the alpha expression language.

Every node exposes a ``key`` property that returns a canonical string form of
the sub-tree it roots.  Two structurally identical sub-trees always produce the
same key, which is what makes common sub-expression caching possible in
:mod:`alphalab.expr.evaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


class Node:
    """Base class for all AST nodes."""

    __slots__ = ()

    @property
    def key(self) -> str:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def children(self) -> Sequence["Node"]:
        return ()

    def walk(self):
        """Yield this node and every descendant, depth first."""
        yield self
        for child in self.children():
            yield from child.walk()

    def depth(self) -> int:
        kids = self.children()
        return 1 + max((c.depth() for c in kids), default=0)

    def size(self) -> int:
        return sum(1 for _ in self.walk())

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True)
class Num(Node):
    """A numeric literal, e.g. ``5`` or ``0.5``."""

    value: float

    @property
    def key(self) -> str:
        # Render integers without a trailing ".0" so that ts_mean(close, 20)
        # and ts_mean(close, 20.0) share a cache entry.
        if float(self.value).is_integer():
            return str(int(self.value))
        return repr(self.value)


@dataclass(frozen=True)
class Var(Node):
    """A reference to a raw data field, e.g. ``close`` or ``volume``."""

    name: str

    @property
    def key(self) -> str:
        return self.name


@dataclass(frozen=True)
class UnaryOp(Node):
    """A prefix operator, currently ``-`` and ``+``."""

    op: str
    operand: Node

    @property
    def key(self) -> str:
        return f"({self.op}{self.operand.key})"

    def children(self) -> Sequence[Node]:
        return (self.operand,)


@dataclass(frozen=True)
class BinOp(Node):
    """An infix operator such as ``+``, ``*`` or ``>``."""

    op: str
    left: Node
    right: Node

    @property
    def key(self) -> str:
        return f"({self.left.key}{self.op}{self.right.key})"

    def children(self) -> Sequence[Node]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Call(Node):
    """A function call such as ``ts_mean(close, 20)``."""

    name: str
    args: tuple = field(default_factory=tuple)

    @property
    def key(self) -> str:
        inner = ",".join(a.key for a in self.args)
        return f"{self.name}({inner})"

    def children(self) -> Sequence[Node]:
        return self.args
