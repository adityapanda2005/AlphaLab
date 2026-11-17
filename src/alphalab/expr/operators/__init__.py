"""The operator library for the alpha expression language.

Importing this package populates the registry as a side effect: each module
below decorates its functions with ``@register``, so simply importing them is
what makes ``ts_mean``, ``rank`` and friends available to the evaluator.

The infix operators in a formula map onto registered functions too, so there
is exactly one implementation of ``+`` in the codebase rather than one in the
evaluator and one in the operator library.
"""

from __future__ import annotations

from . import arithmetic, cross_sectional, group, logic, time_series  # noqa: F401
from .registry import (
    ArityError,
    OperatorError,
    OperatorSpec,
    all_specs,
    count,
    exists,
    get,
    register,
)

#: Infix spelling -> registered operator name.
BINARY_OPERATOR_NAMES = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "%": "modulo",
    "^": "power",
    ">": "greater",
    ">=": "greater_equal",
    "<": "less",
    "<=": "less_equal",
    "==": "equal",
    "!=": "not_equal",
    "&&": "logical_and",
    "||": "logical_or",
}

#: Prefix spelling -> registered operator name.
UNARY_OPERATOR_NAMES = {
    "-": "negate",
    "!": "logical_not",
}

__all__ = [
    "ArityError",
    "OperatorError",
    "OperatorSpec",
    "BINARY_OPERATOR_NAMES",
    "UNARY_OPERATOR_NAMES",
    "all_specs",
    "count",
    "exists",
    "get",
    "register",
]
