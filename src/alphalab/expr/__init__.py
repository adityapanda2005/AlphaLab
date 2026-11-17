"""The alpha expression language: tokenizer, parser, operators and evaluator.

A formula is a string such as::

    -rank(ts_delta(close, 5)) * rank(volume / ts_mean(volume, 20))

which means, in words: go short the stocks that rose most over the last five
days, weighted by how unusually heavily they are trading versus their own
twenty-day average.  The pipeline that turns that string into a panel of
numbers is:

    text --tokenize--> tokens --parse--> AST --evaluate--> (dates x instruments)

Keeping alphas as data rather than as Python functions is what makes the rest
of the project possible: formulas can be stored in config, generated
programmatically, counted, diffed, and -- crucially -- analysed for shared
sub-structure so the engine computes each distinct fragment only once.
"""

from __future__ import annotations

from .ast_nodes import BinOp, Call, Node, Num, UnaryOp, Var
from .cache import CacheStats, NullCache, SubexpressionCache
from .evaluator import (
    EvaluationError,
    EvaluationReport,
    Evaluator,
    UnknownFieldError,
    required_fields,
    shared_subexpressions,
    validate,
)
from .parser import ParseError, parse
from .tokenizer import TokenizeError, tokenize

__all__ = [
    "BinOp",
    "CacheStats",
    "Call",
    "EvaluationError",
    "EvaluationReport",
    "Evaluator",
    "Node",
    "NullCache",
    "Num",
    "ParseError",
    "SubexpressionCache",
    "TokenizeError",
    "UnaryOp",
    "UnknownFieldError",
    "Var",
    "parse",
    "required_fields",
    "shared_subexpressions",
    "tokenize",
    "validate",
]
