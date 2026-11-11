import ast
import asyncio
import contextlib
import io
import re
import os
import sys
import pandas as pd
from typing import Protocol, Dict, Any
from e2b_code_interpreter import AsyncSandbox

class CodeExecutionBackend(Protocol):
    """Protocol defining the contract for any code execution engine."""
    async def run(self, code: str, namespace: Dict[str, Any]) -> str:
        ...

class LocalExecBackend:
    """
    Local execution backend with auto-print and safety filters.
    """
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    async def run(self, code: str, namespace: Dict[str, Any]) -> str:
        stdout_buf = io.StringIO()
        nan_warnings: list[str] = []

        # 1. Security: Block forbidden imports
        self._check_security(code)

        # 2. Convenience: Auto-print injection
        runnable = _inject_auto_print(code)

        def _run_code() -> str:
            import warnings
            import sys
            def _warn_handler(message, category, _filename, _lineno, _file=None, _line=None):
                nan_warnings.append(f"{category.__name__}: {message}")

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.showwarning = _warn_handler
                with contextlib.redirect_stdout(stdout_buf):
                    exec(runnable, namespace)  # noqa: S102
                    sys.stdout.flush() # Ensure real-time terminal sync
            return stdout_buf.getvalue()

        raw_output = await asyncio.wait_for(
            asyncio.to_thread(_run_code),
            timeout=self.timeout_seconds,
        )

        return _format_output(raw_output, nan_warnings)

    @staticmethod
    def _inject_auto_print(source: str) -> str:
        """Compatibility wrapper for the module-level auto-print helper."""
        return _inject_auto_print(source)

    @staticmethod
    def _format_output(raw_output: str, warnings_list: list[str]) -> str:
        """Compatibility wrapper for the module-level output formatter."""
        return _format_output(raw_output, warnings_list)

    def _check_security(self, code: str):
        blocked_pattern = re.compile(
            r"^\s*(?:import|from)\s+(seaborn|scipy|sklearn|plotly|statsmodels|matplotlib)\b",
            re.MULTILINE,
        )
        blocked_match = blocked_pattern.search(code)
        if blocked_match:
            raise ModuleNotFoundError(f"'{blocked_match.group(1)}'")

class E2BBackend:
    """
    Production-safe execution via E2B isolated cloud sandbox.
    Uses the official .create() factory and .kill() for cleanup.
    """
    def __init__(self, api_key: str, timeout_seconds: int = 30):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def run(self, code: str, namespace: Dict[str, Any]) -> str:
        # Prepare data and code logic
        setup_code = _serialize_namespace_to_code(namespace)
        runnable_code = _inject_auto_print(code)

        # Use factory method to bypass Pydantic constructor issues in Pyright
        sandbox = await AsyncSandbox.create(api_key=self.api_key)

        try:
            # Execute the combined payload in the sandbox
            result = await sandbox.run_code(
                setup_code + "\n\n" + runnable_code,
                timeout=self.timeout_seconds
            )
            
            # Convert results to string to satisfy type checker constraints
            output_content = str(result.text) if result.text else ""
            if result.error:
                # result.error is an ExecutionError object, casting to str for feedback
                output_content += f"\nExecution Error: {str(result.error)}"

            return _format_output(output_content or "(no output)", [])
        finally:
            # The correct method to terminate an E2B sandbox is kill()
            await sandbox.kill()

# --- Global Helpers ---

def _inject_auto_print(source: str) -> str:
    """Wraps the last bare expression in a print() statement for Jupyter-like feel."""
    try:
        tree = ast.parse(source)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body[-1]
            if not (isinstance(last.value, ast.Call) and
                    isinstance(last.value.func, ast.Name) and
                    last.value.func.id == "print"):
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
    except SyntaxError:
        pass
    return source

def _format_output(raw_output: str, warnings_list: list[str]) -> str:
    """Formats stdout and warnings for the LLM's consumption."""
    nan_tokens = {"nan", "inf", "-inf", "none"}
    output_lower = raw_output.lower()
    nan_hits = [tok for tok in nan_tokens if tok in output_lower]

    result_parts = [raw_output.strip()] if raw_output.strip() else []
    if warnings_list:
        result_parts.append("MATH WARNINGS:\n" + "\n".join(f"  • {w}" for w in warnings_list))
    if nan_hits:
        result_parts.append(f"NaN/Inf detected in output ({', '.join(nan_hits)}). Filter nulls first.")

    return "\n\n".join(result_parts) or "(No output detected. Use print() to see data.)"

def _serialize_namespace_to_code(namespace: dict) -> str:
    """Converts local DataFrames into code that creates them in the sandbox."""
    lines = ["import pandas as pd", "import io", "dfs = {}"]
    for key, val in namespace.items():
        if isinstance(val, pd.DataFrame):
            csv_data = val.to_csv(index=False).replace("'", "\\'")
            lines.append(
                f"csv_{key} = '''{csv_data}'''\n"
                f"dfs['{key}'] = pd.read_csv(io.StringIO(csv_{key}))"
            )
    return "\n".join(lines)
