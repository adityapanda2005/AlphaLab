"""Hand-written tokenizer for the alpha expression language.

The grammar is small enough that a regex-free character scanner is both the
fastest and the clearest implementation.  Keeping it hand-written (rather than
reaching for ``ast.literal_eval`` or ``eval``) is deliberate: the expression
language must never be able to execute arbitrary Python, because formulas are
treated as untrusted configuration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    NUMBER = auto()
    IDENT = auto()
    OP = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    pos: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.type.name}, {self.value!r}, @{self.pos})"


class TokenizeError(ValueError):
    """Raised when the source string contains a character we cannot lex."""


# Multi-character operators must be tried before their single-character
# prefixes, otherwise ">=" would lex as ">" followed by "=".
_TWO_CHAR_OPS = ("<=", ">=", "==", "!=", "&&", "||")
_ONE_CHAR_OPS = "+-*/%^<>?:"


def tokenize(source: str) -> List[Token]:
    """Turn ``source`` into a flat list of tokens, ending with an EOF token."""
    tokens: List[Token] = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "(":
            tokens.append(Token(TokenType.LPAREN, ch, i))
            i += 1
            continue

        if ch == ")":
            tokens.append(Token(TokenType.RPAREN, ch, i))
            i += 1
            continue

        if ch == ",":
            tokens.append(Token(TokenType.COMMA, ch, i))
            i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            start = i
            seen_dot = False
            seen_exp = False
            while i < n:
                c = source[i]
                if c.isdigit():
                    i += 1
                elif c == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    i += 1
                elif c in "eE" and not seen_exp and i + 1 < n:
                    seen_exp = True
                    i += 1
                    if i < n and source[i] in "+-":
                        i += 1
                else:
                    break
            tokens.append(Token(TokenType.NUMBER, source[start:i], start))
            continue

        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(Token(TokenType.IDENT, source[start:i], start))
            continue

        two = source[i : i + 2]
        if two in _TWO_CHAR_OPS:
            tokens.append(Token(TokenType.OP, two, i))
            i += 2
            continue

        if ch in _ONE_CHAR_OPS:
            tokens.append(Token(TokenType.OP, ch, i))
            i += 1
            continue

        raise TokenizeError(
            f"unexpected character {ch!r} at position {i} in {source!r}"
        )

    tokens.append(Token(TokenType.EOF, "", n))
    return tokens
