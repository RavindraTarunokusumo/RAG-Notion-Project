"""NRAG-031: Math Solver Tool Agent."""

import logging
from typing import Any

from src.tools.base import AgentCard, ToolAgent, ToolResult

logger = logging.getLogger(__name__)

_TRIGGER_KEYWORDS = {
    "solve", "equation", "calculate", "integral", "derivative",
    "simplify", "factor", "expand", "limit", "sum", "matrix",
    "algebra", "calculus", "math", "formula", "expression",
}


class MathSolverAgent(ToolAgent):
    """Tool agent that solves mathematical expressions using SymPy."""

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="math_solver",
            description="Solves mathematical expressions and equations using symbolic computation",
            version="1.0.0",
            capabilities=["math_solving", "equation_evaluation"],
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression or equation to solve"},
                    "operation": {
                        "type": "string",
                        "enum": ["simplify", "solve", "diff", "integrate", "evaluate"],
                        "default": "simplify",
                    },
                    "variable": {"type": "string", "default": "x"},
                },
                "required": ["expression"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "latex": {"type": "string"},
                    "operation": {"type": "string"},
                },
            },
            endpoint="tool://math_solver",
        )

    def can_handle(self, task_description: str) -> float:
        lower = task_description.lower()
        matches = sum(1 for kw in _TRIGGER_KEYWORDS if kw in lower)
        if matches:
            return min(0.3 + 0.15 * matches, 1.0)
        return 0.0

    def health_check(self) -> bool:
        try:
            import sympy  # noqa: F401
            return True
        except ImportError:
            return False

    async def execute(self, task: dict[str, Any]) -> ToolResult:
        expression = task.get("expression", "")
        operation = task.get("operation", "simplify")
        variable = task.get("variable", "x")

        if not expression:
            return ToolResult(success=False, data=None, error="No expression provided")

        try:
            import sympy
            from sympy.parsing.sympy_parser import (
                convert_xor,
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )

            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )

            var = sympy.Symbol(variable)
            expr = parse_expr(expression, local_dict={variable: var}, transformations=transformations)

            if operation == "simplify":
                result = sympy.simplify(expr)
            elif operation == "solve":
                result = sympy.solve(expr, var)
            elif operation == "diff":
                result = sympy.diff(expr, var)
            elif operation == "integrate":
                result = sympy.integrate(expr, var)
            elif operation == "evaluate":
                result = expr.evalf()
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown operation: {operation}",
                )

            result_str = str(result)
            latex_str = sympy.latex(result)

            return ToolResult(
                success=True,
                data={
                    "result": result_str,
                    "latex": latex_str,
                    "operation": operation,
                },
                metadata={"expression": expression, "variable": variable},
            )
        except ImportError:
            return ToolResult(
                success=False,
                data=None,
                error="sympy is not installed. Install it with: pip install sympy",
            )
        except Exception as e:
            logger.exception("Math solving failed")
            return ToolResult(success=False, data=None, error=str(e))
