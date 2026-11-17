"""Recursive-descent parser for the alpha expression language.

The parser turns the flat token list produced by :mod:`alphalab.expr.tokenizer`
into the AST defined in :mod:`alphalab.expr.ast_nodes`.

Grammar (lowest precedence first)::

    expression  := ternary
    ternary     := logic_or ( "?" expression ":" ternary )?      # right assoc
    logic_or    := logic_and ( "||" logic_and )*
    logic_and   := equality ( "&&" equality )*
    equality    := comparison ( ("==" | "!=") comparison )*
    comparison  := additive ( ("<" | ">" | "<=" | ">=") additive )*
    additive    := multiplicative ( ("+" | "-") multiplicative )*
    multiplicative := power ( ("*" | "/" | "%") power )*
    power       := unary ( "^" power )?                          # right assoc
    unary       := ("-" | "+") unary | primary
    primary     := NUMBER
                 | IDENT "(" arguments ")"
                 | IDENT
                 | "(" expression ")"
    arguments   := ( expression ("," expression)* )?

Why hand-written rather than a parser generator?  The grammar is twenty lines
long, a hand-written parser has no build step and no runtime dependency, and
the error messages can point at the exact character that broke.  It is also the
part of the project most worth being able to explain line by line.

The ternary ``cond ? a : b`` is desugared into ``if_else(cond, a, b)`` so that
the evaluator and the operator registry only ever deal with named functions.
"""

from __future__ import annotations

from typing import List, Optional

from .ast_nodes import BinOp, Call, Node, Num, UnaryOp, Var
from .tokenizer import Token, TokenType, tokenize


class ParseError(ValueError):
    """Raised when the token stream does not match the grammar."""


# Binary operator groups, from lowest binding power to highest.  Each entry is
# a tuple of operator spellings that share a precedence level.
_LOGIC_OR = ("||",)
_LOGIC_AND = ("&&",)
_EQUALITY = ("==", "!=")
_COMPARISON = ("<", ">", "<=", ">=")
_ADDITIVE = ("+", "-")
_MULTIPLICATIVE = ("*", "/", "%")


class Parser:
    """Consumes a token list and produces a single expression tree.

    A fresh instance is cheap; :func:`parse` creates one per formula, so the
    parser holds no state between calls and is safe to use from threads.
    """

    def __init__(self, tokens: List[Token], source: str = "") -> None:
        self._tokens = tokens
        self._source = source
        self._pos = 0

    # ------------------------------------------------------------------
    # Token stream helpers
    # ------------------------------------------------------------------
    @property
    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        if token.type is not TokenType.EOF:
            self._pos += 1
        return token

    def _check_op(self, spellings: tuple) -> bool:
        token = self._current
        return token.type is TokenType.OP and token.value in spellings

    def _match_op(self, spellings: tuple) -> Optional[Token]:
        if self._check_op(spellings):
            return self._advance()
        return None

    def _expect(self, token_type: TokenType, what: str) -> Token:
        token = self._current
        if token.type is not token_type:
            raise ParseError(
                f"expected {what} at position {token.pos} but found "
                f"{token.value!r} in {self._source!r}"
            )
        return self._advance()

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------
    def parse_expression(self) -> Node:
        return self._ternary()

    def _ternary(self) -> Node:
        condition = self._binary(0)
        if not self._check_op(("?",)):
            return condition
        self._advance()  # consume "?"
        if_true = self.parse_expression()
        if not self._check_op((":",)):
            token = self._current
            raise ParseError(
                f"expected ':' to close the '?' at position {token.pos} "
                f"in {self._source!r}"
            )
        self._advance()  # consume ":"
        # Right associative: a ? b : c ? d : e parses as a ? b : (c ? d : e).
        if_false = self._ternary()
        return Call("if_else", (condition, if_true, if_false))

    # The binary levels share one driver so that adding a precedence level is
    # a one-line change to _BINARY_LEVELS rather than a new method.
    _BINARY_LEVELS = (
        _LOGIC_OR,
        _LOGIC_AND,
        _EQUALITY,
        _COMPARISON,
        _ADDITIVE,
        _MULTIPLICATIVE,
    )

    def _binary(self, level: int) -> Node:
        if level >= len(self._BINARY_LEVELS):
            return self._power()

        node = self._binary(level + 1)
        spellings = self._BINARY_LEVELS[level]
        while self._check_op(spellings):
            op = self._advance().value
            right = self._binary(level + 1)
            node = BinOp(op, node, right)
        return node

    def _power(self) -> Node:
        base = self._unary()
        if self._check_op(("^",)):
            self._advance()
            # Right associative: 2 ^ 3 ^ 2 == 2 ^ (3 ^ 2).
            exponent = self._power()
            return BinOp("^", base, exponent)
        return base

    def _unary(self) -> Node:
        token = self._match_op(("-", "+"))
        if token is not None:
            operand = self._unary()
            if token.value == "+":
                # Unary plus is a no-op; dropping it keeps cache keys canonical
                # so that "+close" and "close" share a cache entry.
                return operand
            return UnaryOp("-", operand)
        return self._primary()

    def _primary(self) -> Node:
        token = self._current

        if token.type is TokenType.NUMBER:
            self._advance()
            return Num(float(token.value))

        if token.type is TokenType.IDENT:
            self._advance()
            if self._current.type is TokenType.LPAREN:
                return self._finish_call(token.value)
            return Var(token.value)

        if token.type is TokenType.LPAREN:
            self._advance()
            inner = self.parse_expression()
            self._expect(TokenType.RPAREN, "')'")
            return inner

        raise ParseError(
            f"unexpected {token.value!r} at position {token.pos} "
            f"in {self._source!r}"
        )

    def _finish_call(self, name: str) -> Call:
        self._expect(TokenType.LPAREN, "'('")
        args: List[Node] = []
        if self._current.type is not TokenType.RPAREN:
            args.append(self.parse_expression())
            while self._current.type is TokenType.COMMA:
                self._advance()
                args.append(self.parse_expression())
        self._expect(TokenType.RPAREN, "')'")
        return Call(name, tuple(args))


def parse(source: str) -> Node:
    """Parse ``source`` into an AST.

    >>> parse("rank(close)").key
    'rank(close)'
    >>> parse("-ts_delta(close, 5)").key
    '(-ts_delta(close,5))'

    Raises :class:`ParseError` if the formula is malformed, or
    :class:`~alphalab.expr.tokenizer.TokenizeError` if it contains a character
    outside the language.
    """
    tokens = tokenize(source)
    parser = Parser(tokens, source)
    node = parser.parse_expression()

    trailing = parser._current
    if trailing.type is not TokenType.EOF:
        raise ParseError(
            f"unexpected trailing {trailing.value!r} at position "
            f"{trailing.pos} in {source!r}"
        )
    return node
