"""NRAG-029: Code Executor Tool Agent (sandboxed)."""

import logging
import subprocess
import sys
import textwrap
from typing import Any

from src.tools.base import AgentCard, ToolAgent, ToolResult

logger = logging.getLogger(__name__)

_ALLOWED_MODULES = {"numpy", "math", "statistics", "collections", "itertools", "functools", "decimal", "fractions"}
_MAX_OUTPUT_LENGTH = 5000


class CodeExecutorAgent(ToolAgent):
    """Tool agent that executes Python code in a sandboxed subprocess."""

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="code_executor",
            description="Executes Python code in a sandboxed environment with restricted imports",
            version="1.0.0",
            capabilities=["code_execution", "python_eval"],
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "number", "default": 10},
                },
                "required": ["code"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "return_code": {"type": "integer"},
                },
            },
            endpoint="tool://code_executor",
        )

    def can_handle(self, task_description: str) -> float:
        keywords = {"calculate", "compute", "evaluate", "execute", "run code", "python", "script"}
        lower = task_description.lower()
        matches = sum(1 for kw in keywords if kw in lower)
        if matches:
            return min(0.3 + 0.15 * matches, 1.0)
        return 0.0

    async def execute(self, task: dict[str, Any]) -> ToolResult:
        code = task.get("code", "")
        timeout = min(task.get("timeout", 10), 30)  # Cap at 30s

        if not code:
            return ToolResult(success=False, data=None, error="No code provided")

        # Validate imports
        validation_error = self._validate_imports(code)
        if validation_error:
            return ToolResult(success=False, data=None, error=validation_error)

        # Wrap code with import restrictions
        wrapper = textwrap.dedent(f"""\
            import sys
            # Restrict available modules
            _allowed = {_ALLOWED_MODULES!r}
            _original_import = __builtins__.__import__
            def _restricted_import(name, *args, **kwargs):
                top_level = name.split('.')[0]
                if top_level not in _allowed and top_level not in sys.stdlib_module_names:
                    raise ImportError(f"Import of '{{name}}' is not allowed")
                return _original_import(name, *args, **kwargs)
            __builtins__.__import__ = _restricted_import

            # User code
            {textwrap.indent(code, '            ').strip()}
        """)

        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": ""},  # Restrict PATH
            )

            stdout = result.stdout[:_MAX_OUTPUT_LENGTH]
            stderr = result.stderr[:_MAX_OUTPUT_LENGTH]

            return ToolResult(
                success=result.returncode == 0,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": result.returncode,
                },
                error=stderr if result.returncode != 0 else None,
                metadata={"timeout": timeout},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                data=None,
                error=f"Code execution timed out after {timeout}s",
            )
        except Exception as e:
            logger.exception("Code execution failed")
            return ToolResult(success=False, data=None, error=str(e))

    @staticmethod
    def _validate_imports(code: str) -> str | None:
        """Pre-validate that code doesn't import disallowed modules."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error in code: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in _ALLOWED_MODULES and top not in sys.stdlib_module_names:
                        return f"Import of '{alias.name}' is not allowed. Allowed: {_ALLOWED_MODULES}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_MODULES and top not in sys.stdlib_module_names:
                    return f"Import from '{node.module}' is not allowed. Allowed: {_ALLOWED_MODULES}"
        return None
