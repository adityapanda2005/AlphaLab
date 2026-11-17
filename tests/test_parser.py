"""Tests for the tokenizer, parser and AST."""

import pytest
from alphalab.expr import parse, ParseError, TokenizeError, tokenize
from alphalab.expr.ast_nodes import BinOp, Call, Num, UnaryOp, Var


class TestTokenizer:
    def test_simple(self):
        tokens = tokenize("close + 1")
        values = [(t.type.name, t.value) for t in tokens]
        assert ("IDENT", "close") in values
        assert ("OP", "+") in values
        assert ("NUMBER", "1") in values

    def test_two_char_ops(self):
        tokens = tokenize("x >= y")
        ops = [t.value for t in tokens if t.type.name == "OP"]
        assert ">=" in ops

    def test_scientific_notation(self):
        tokens = tokenize("1.5e-3")
        assert tokens[0].value == "1.5e-3"

    def test_bad_char(self):
        with pytest.raises(TokenizeError):
            tokenize("close @ 1")


class TestParser:
    def test_literal(self):
        assert isinstance(parse("42"), Num)
        assert parse("42").value == 42.0

    def test_variable(self):
        assert isinstance(parse("close"), Var)
        assert parse("close").name == "close"

    def test_binary(self):
        node = parse("a + b")
        assert isinstance(node, BinOp)
        assert node.op == "+"

    def test_precedence(self):
        """Multiplication binds tighter than addition."""
        assert parse("a + b * c").key == "(a+(b*c))"

    def test_right_associative_power(self):
        assert parse("2 ^ 3 ^ 2").key == "(2^(3^2))"

    def test_unary_minus(self):
        node = parse("-close")
        assert isinstance(node, UnaryOp)
        assert node.op == "-"

    def test_unary_plus_dropped(self):
        """Unary plus is a no-op; dropping it canonicalises cache keys."""
        assert parse("+close").key == "close"

    def test_function_call(self):
        node = parse("ts_mean(close, 20)")
        assert isinstance(node, Call)
        assert node.name == "ts_mean"
        assert len(node.args) == 2

    def test_nested(self):
        node = parse("rank(ts_delta(close, 5))")
        assert node.key == "rank(ts_delta(close,5))"

    def test_ternary(self):
        """The ternary ``? :`` desugars into ``if_else(cond, a, b)``."""
        node = parse("close > open ? volume : 0")
        assert isinstance(node, Call)
        assert node.name == "if_else"
        assert len(node.args) == 3

    def test_nested_ternary_right_assoc(self):
        node = parse("a ? b : c ? d : e")
        assert node.key == "if_else(a,b,if_else(c,d,e))"

    def test_parenthesized(self):
        assert parse("(a + b) * c").key == "((a+b)*c)"

    def test_canonical_key_deduplication(self):
        """Structurally identical trees produce identical keys."""
        a = parse("rank(ts_delta(close, 5))")
        b = parse("rank(  ts_delta(  close ,  5  ) )")
        assert a.key == b.key

    def test_trailing_tokens_rejected(self):
        with pytest.raises(ParseError):
            parse("close close")

    def test_missing_rparen(self):
        with pytest.raises(ParseError):
            parse("ts_mean(close, 20")

    def test_empty_raises(self):
        with pytest.raises(ParseError):
            parse("")


class TestAST:
    def test_walk(self):
        node = parse("rank(close)")
        nodes = list(node.walk())
        assert len(nodes) == 2  # Call + Var

    def test_depth(self):
        assert parse("close").depth() == 1
        assert parse("rank(close)").depth() == 2

    def test_size(self):
        assert parse("a + b").size() == 3
