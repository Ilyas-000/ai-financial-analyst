"""Unit tests for app.tools.calculator."""

from decimal import Decimal

import pytest

from app.tools.calculator import CalculatorError, calculate


def test_arithmetic_returns_decimal():
    result = calculate("(100 + 200) / 3")
    assert isinstance(result, Decimal)
    assert result == Decimal("100")


def test_integer_division_and_modulo():
    assert calculate("17 // 5") == Decimal("3")
    assert calculate("17 % 5") == Decimal("2")


def test_power_operator():
    assert calculate("2 ** 10") == Decimal("1024")


def test_unary_minus():
    assert calculate("-(5 + 3)") == Decimal("-8")


def test_min_max_round_abs():
    assert calculate("min(1, 2, 3)") == Decimal("1")
    assert calculate("max(1, 2, 3)") == Decimal("3")
    assert calculate("round(3.7)") == Decimal("4")
    assert calculate("abs(-42)") == Decimal("42")


def test_sum_over_list():
    assert calculate("sum([1, 2, 3, 4])") == Decimal("10")


def test_avg_helper():
    assert calculate("avg([2, 4, 6])") == Decimal("4")


def test_nested_calls():
    assert calculate("round(avg([1.0, 2.0, 3.0]), 2)") == Decimal("2.0")


def test_dunder_rejected():
    with pytest.raises(CalculatorError, match="dunder"):
        calculate("__import__('os')")


def test_import_statement_rejected():
    with pytest.raises(CalculatorError):
        calculate("import os")


def test_lambda_rejected():
    with pytest.raises(CalculatorError, match="forbidden syntax|unknown name"):
        calculate("lambda x: x")


def test_assignment_rejected():
    with pytest.raises(CalculatorError, match="syntax error"):
        calculate("x = 5")


def test_multi_statement_rejected():
    with pytest.raises(CalculatorError, match="syntax error"):
        calculate("1 + 1; 2 + 2")


def test_unknown_name_rejected():
    with pytest.raises(CalculatorError, match="unknown name"):
        calculate("open('foo')")


def test_attribute_access_rejected():
    with pytest.raises(CalculatorError, match="forbidden syntax"):
        calculate("(1).numerator")


def test_subscript_rejected():
    with pytest.raises(CalculatorError, match="forbidden syntax"):
        calculate("[1, 2, 3][0]")


def test_keyword_args_rejected():
    with pytest.raises(CalculatorError, match="keyword arguments"):
        calculate("round(3.7, ndigits=1)")


def test_comparison_rejected():
    with pytest.raises(CalculatorError, match="forbidden syntax"):
        calculate("1 < 2")


def test_list_comprehension_rejected():
    with pytest.raises(CalculatorError, match="forbidden syntax"):
        calculate("sum([x for x in [1, 2, 3]])")


def test_empty_expression_rejected():
    with pytest.raises(CalculatorError, match="empty"):
        calculate("   ")


def test_too_long_expression_rejected():
    with pytest.raises(CalculatorError, match="exceeds"):
        calculate("1+" * 200 + "1")


def test_non_string_rejected():
    with pytest.raises(CalculatorError, match="must be a string"):
        calculate(123)  # type: ignore[arg-type]


def test_division_by_zero_surfaces_error():
    with pytest.raises(CalculatorError, match="ZeroDivisionError"):
        calculate("1 / 0")


def test_avg_on_empty_list_rejected():
    with pytest.raises(CalculatorError, match="ValueError"):
        calculate("avg([])")


def test_float_result_converted_via_repr():
    # repr(0.1 + 0.2) = '0.30000000000000004'; verify Decimal sees the same.
    assert calculate("0.1 + 0.2") == Decimal("0.30000000000000004")
