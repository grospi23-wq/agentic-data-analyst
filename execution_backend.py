"""
execution_backend.py
--------------------
Defines the protocol for code execution backends and provides a local implementation.
This abstraction allows swapping the insecure local exec() with a sandboxed environment
(like Docker or E2B) for production deployments without changing the agent logic.
"""

import ast
import asyncio
import contextlib
import io
import re
from typing import Protocol, Dict, Any, Tuple

class CodeExecutionBackend(Protocol):
    """Protocol defining the contract for any code execution engine."""
    async def run(self, code: str, namespace: Dict[str, Any]) -> str:
        ...

class LocalExecBackend:
    """
    Local execution backend using Python's built-in exec().
    WARNING: Suitable for portfolio demonstrations and trusted environments only.
    """
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    async def run(self, code: str, namespace: Dict[str, Any]) -> str:
        stdout_buf = io.StringIO()
        nan_warnings: list[str] = []

        # 1. Security Layer: Block forbidden imports
        blocked_pattern = re.compile(
            r"^\s*(?:import|from)\s+(seaborn|scipy|sklearn|plotly|statsmodels|matplotlib)\b",
            re.MULTILINE,
        )
        blocked_match = blocked_pattern.search(code)
        if blocked_match:
            raise ModuleNotFoundError(f"'{blocked_match.group(1)}'")

        # 2. Convenience Layer: Auto-print injection (Jupyter style)
        runnable = self._inject_auto_print(code)

        def _run_code() -> str:
            import warnings
            def _warn_handler(message, category, _filename, _lineno, _file=None, _line=None):
                nan_warnings.append(f"{category.__name__}: {message}")

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.showwarning = _warn_handler
                with contextlib.redirect_stdout(stdout_buf):
                    exec(runnable, namespace)  # noqa: S102
            return stdout_buf.getvalue()

        # 3. Execution Layer: Run with timeout
        raw_output = await asyncio.wait_for(
            asyncio.to_thread(_run_code),
            timeout=self.timeout_seconds,
        )

        return self._format_output(raw_output, nan_warnings)

    def _inject_auto_print(self, source: str) -> str:
        """Wraps the last bare expression in a print() statement."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body[-1]
            is_already_print = (
                isinstance(last.value, ast.Call) and 
                isinstance(last.value.func, ast.Name) and 
                last.value.func.id == "print"
            )
            
            if not is_already_print:
                print_node = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print", ctx=ast.Load()),
                        args=[last.value],
                        keywords=[],
                    )
                )
                ast.copy_location(print_node, last)
                ast.fix_missing_locations(print_node)
                tree.body[-1] = print_node
                return ast.unparse(tree)
        return source

    def _format_output(self, raw_output: str, warnings_list: list[str]) -> str:
        """Formats the stdout and warnings into a single string for the LLM."""
        nan_tokens = {"nan", "inf", "-inf", "none"}
        output_lower = raw_output.lower()
        nan_hits = [tok for tok in nan_tokens if tok in output_lower]

        result_parts = [raw_output.strip()] if raw_output.strip() else []

        if warnings_list:
            result_parts.append(
                "MATH WARNINGS (self-correct before reporting):\n"
                + "\n".join(f"  • {w}" for w in warnings_list)
            )
        if nan_hits:
            result_parts.append(
                f"NaN/Inf DETECTED in output ({', '.join(nan_hits)}). "
                "Do NOT cite these values as findings. "
                "Filter nulls first (e.g. .dropna()) or use a subset with sufficient data."
            )

        return "\n\n".join(result_parts) or (
            "(No output detected. If you assigned a result to a variable, "
            "you MUST add a `print(variable_name)` statement to see the data.)"
        )

class DockerBackendStub:
    """
    Stub implementation demonstrating how a secure backend would conform to the protocol.
    To be implemented for production deployment.
    """
    async def run(self, code: str, namespace: Dict[str, Any]) -> str:
        return "ERROR: Secure Docker backend not yet implemented."