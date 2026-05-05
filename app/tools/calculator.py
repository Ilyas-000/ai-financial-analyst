"""Sandboxed calculator for the SQL Analyst / Finalize nodes.

The agent calls ``calculate("expr")`` on summary numbers it already has from a
SELECT. We do not want the LLM to reach attributes, dunders, comprehensions,
or any builtin beyond a small whitelist, so the eval path has three guards:

1. ``ast.parse(..., mode="eval")`` — rejects multi-statement input, assignments,
   imports, function/class defs (these are not valid in expression mode).
2. AST walker — only a fixed set of expression node kinds is allowed; anything
   else (``Attribute``, ``Subscript``, ``Lambda``, ``IfExp``, comprehensions,
   ``Compare``, ``BoolOp``, ``keyword``-args, ``Starred``, f-strings) → reject.
3. ``asteval.Interpreter(minimal=True)`` with a stripped ``symtable`` containing
   only ``min/max/sum/round/abs/avg``.

The result is converted to ``Decimal`` via ``Decimal(repr(value))`` so the
caller (which deals with money) does not see ``float`` further down the stack.
Note: arithmetic itself runs as ``float`` inside asteval, so binary-fraction
artifacts like ``0.1 + 0.2 == 0.30000000000000004`` can leak into the Decimal.
This is acceptable for ratio/share/aggregation tweaks done on already-rounded
SQL outputs; high-precision money math stays in SQL or in the FX tool.
"""

import ast
from decimal import Decimal
from typing import Final

from asteval import Interpreter

# Hard limit on input length — defense against pathological eval payloads.
_MAX_EXPRESSION_LEN: Final[int] = 200

_ALLOWED_NODES: Final[frozenset[type[ast.AST]]] = frozenset(
    {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.List,
        ast.Tuple,
        # Numeric ops only.
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    }
)

_ALLOWED_NAMES: Final[frozenset[str]] = frozenset({"min", "max", "sum", "round", "abs", "avg"})


class CalculatorError(ValueError):
    """Raised when the expression is rejected or evaluation fails."""


def calculate(expression: str) -> Decimal:
    """Evaluate a numeric expression in a hardened sandbox and return Decimal."""
    if not isinstance(expression, str):
        raise CalculatorError("expression must be a string")
    if not expression.strip():
        raise CalculatorError("expression is empty")
    if len(expression) > _MAX_EXPRESSION_LEN:
        raise CalculatorError(f"expression exceeds {_MAX_EXPRESSION_LEN} chars")
    if "__" in expression:
        raise CalculatorError("dunder access is not allowed")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"syntax error: {exc.msg}") from exc

    _assert_allowed(tree)

    interp = _build_interpreter()
    result = interp(expression, show_errors=False, raise_errors=False)

    if interp.error:
        first = interp.error[0]
        exc_name = first.exc.__name__ if first.exc else "EvalError"
        raise CalculatorError(f"eval failed: {exc_name}: {first.msg}")

    if result is None:
        raise CalculatorError("expression produced no value")
    if isinstance(result, bool):  # bool is a subclass of int — reject explicitly
        raise CalculatorError("boolean result is not allowed")
    if not isinstance(result, int | float | Decimal):
        raise CalculatorError(f"unsupported result type: {type(result).__name__}")

    return Decimal(repr(result)) if isinstance(result, float) else Decimal(result)


def _assert_allowed(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_NODES:
            raise CalculatorError(f"forbidden syntax: {node_type.__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise CalculatorError(f"unknown name: {node.id}")
        if isinstance(node, ast.Call) and node.keywords:
            raise CalculatorError("keyword arguments are not allowed")


def _build_interpreter() -> Interpreter:
    """Asteval interpreter with empty builtins + whitelisted callables."""
    interp = Interpreter(minimal=True, use_numpy=False)
    for key in list(interp.symtable.keys()):
        del interp.symtable[key]
    interp.symtable["min"] = min
    interp.symtable["max"] = max
    interp.symtable["sum"] = sum
    interp.symtable["round"] = round
    interp.symtable["abs"] = abs
    interp.symtable["avg"] = _avg
    return interp


def _avg(values: list[float] | tuple[float, ...]) -> float:
    if not values:
        raise ValueError("avg() requires at least one value")
    return sum(values) / len(values)
