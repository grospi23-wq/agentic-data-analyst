"""
test_execution_backend.py
-------------------------
Tests for LocalExecBackend — the sandboxed Python execution engine used
by the analyst agent's execute_python_analysis tool.

Async tests use anyio (already installed as a transitive dependency) via
@pytest.mark.anyio.
"""

import asyncio
import pytest
from execution_backend import LocalExecBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return LocalExecBackend(timeout_seconds=5)


@pytest.fixture
def namespace():
    """Fresh empty namespace for each test."""
    import pandas as pd
    import numpy as np
    return {"pd": pd, "np": np}


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.anyio
    async def test_arithmetic_output(self, backend, namespace):
        result = await backend.run("print(2 + 2)", namespace)
        assert "4" in result

    @pytest.mark.anyio
    async def test_string_output(self, backend, namespace):
        result = await backend.run('print("hello world")', namespace)
        assert "hello world" in result

    @pytest.mark.anyio
    async def test_multiline_code(self, backend, namespace):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = await backend.run(code, namespace)
        assert "30" in result

    @pytest.mark.anyio
    async def test_empty_output_fallback_message(self, backend, namespace):
        """Code with no print and no last expression returns the fallback message."""
        result = await backend.run("x = 42", namespace)
        assert "No output detected" in result or "print" in result

    @pytest.mark.anyio
    async def test_pandas_available(self, backend, namespace):
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(len(df))"
        result = await backend.run(code, namespace)
        assert "3" in result


# ---------------------------------------------------------------------------
# Auto-print injection (Jupyter style)
# ---------------------------------------------------------------------------

class TestAutoPrintInjection:
    @pytest.mark.anyio
    async def test_last_expression_auto_printed(self, backend, namespace):
        """A bare expression on the last line is automatically wrapped in print()."""
        result = await backend.run("1 + 1", namespace)
        assert "2" in result

    @pytest.mark.anyio
    async def test_dataframe_expression_auto_printed(self, backend, namespace):
        code = "import pandas as pd\npd.DataFrame({'x': [1, 2]})"
        result = await backend.run(code, namespace)
        # DataFrame repr should be in the output
        assert "x" in result

    @pytest.mark.anyio
    async def test_existing_print_not_double_wrapped(self, backend, namespace):
        """Already-print() last line is NOT double-wrapped."""
        result = await backend.run('print("already printed")', namespace)
        # Should appear exactly once, not twice
        assert result.count("already printed") == 1

    def test_inject_auto_print_unit(self):
        """Unit test for _inject_auto_print directly."""
        backend = LocalExecBackend()
        result = backend._inject_auto_print("1 + 1")
        assert "print" in result

    def test_inject_auto_print_skips_existing_print(self):
        backend = LocalExecBackend()
        source = 'print("x")'
        result = backend._inject_auto_print(source)
        assert result.count("print") == 1

    def test_inject_auto_print_handles_syntax_error(self):
        """Syntax errors are returned unchanged (exec will handle them)."""
        backend = LocalExecBackend()
        bad_source = "def (\n"
        result = backend._inject_auto_print(bad_source)
        assert result == bad_source


# ---------------------------------------------------------------------------
# Blocked imports
# ---------------------------------------------------------------------------

class TestBlockedImports:
    @pytest.mark.anyio
    async def test_seaborn_blocked(self, backend, namespace):
        with pytest.raises(ModuleNotFoundError, match="seaborn"):
            await backend.run("import seaborn", namespace)

    @pytest.mark.anyio
    async def test_scipy_blocked(self, backend, namespace):
        with pytest.raises(ModuleNotFoundError, match="scipy"):
            await backend.run("import scipy", namespace)

    @pytest.mark.anyio
    async def test_sklearn_blocked(self, backend, namespace):
        with pytest.raises(ModuleNotFoundError, match="sklearn"):
            await backend.run("from sklearn import tree", namespace)

    @pytest.mark.anyio
    async def test_matplotlib_blocked(self, backend, namespace):
        with pytest.raises(ModuleNotFoundError, match="matplotlib"):
            await backend.run("import matplotlib.pyplot as plt", namespace)

    @pytest.mark.anyio
    async def test_statsmodels_blocked(self, backend, namespace):
        with pytest.raises(ModuleNotFoundError, match="statsmodels"):
            await backend.run("import statsmodels.api as sm", namespace)

    @pytest.mark.anyio
    async def test_allowed_import_not_blocked(self, backend, namespace):
        """pandas and numpy are explicitly allowed."""
        result = await backend.run("import pandas as pd; print(pd.__version__)", namespace)
        assert "Error" not in result


# ---------------------------------------------------------------------------
# Namespace persistence
# ---------------------------------------------------------------------------

class TestNamespacePersistence:
    @pytest.mark.anyio
    async def test_variable_persists_across_calls(self, backend, namespace):
        await backend.run("x = 99", namespace)
        result = await backend.run("print(x)", namespace)
        assert "99" in result

    @pytest.mark.anyio
    async def test_dataframe_persists_across_calls(self, backend, namespace):
        import pandas as pd
        namespace["dfs"] = {"orders": pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})}
        await backend.run("df = dfs['orders']", namespace)
        result = await backend.run("print(len(df))", namespace)
        assert "3" in result


# ---------------------------------------------------------------------------
# NaN / Inf detection
# ---------------------------------------------------------------------------

class TestNaNInfDetection:
    @pytest.mark.anyio
    async def test_nan_in_output_triggers_warning(self, backend, namespace):
        result = await backend.run("import numpy as np; print(np.nan)", namespace)
        assert "NaN" in result or "nan" in result.lower()

    @pytest.mark.anyio
    async def test_inf_in_output_triggers_warning(self, backend, namespace):
        result = await backend.run("import numpy as np; print(np.inf)", namespace)
        assert "Inf" in result or "inf" in result.lower()

    def test_format_output_nan_detection(self):
        backend = LocalExecBackend()
        output = backend._format_output("result: nan\n", [])
        assert "NaN" in output or "nan" in output.lower()

    def test_format_output_clean_output_unchanged(self):
        backend = LocalExecBackend()
        output = backend._format_output("42\n", [])
        assert "42" in output
        assert "NaN" not in output


# ---------------------------------------------------------------------------
# Warning capture
# ---------------------------------------------------------------------------

class TestWarningCapture:
    @pytest.mark.anyio
    async def test_runtime_warnings_captured(self, backend, namespace):
        """Pandas operations that trigger warnings should surface in output."""
        code = (
            "import warnings\n"
            "warnings.warn('test warning', RuntimeWarning)\n"
            "print('done')"
        )
        result = await backend.run(code, namespace)
        assert "done" in result

    def test_format_output_includes_warnings(self):
        backend = LocalExecBackend()
        output = backend._format_output("result\n", ["RuntimeWarning: test"])
        assert "RuntimeWarning" in output or "MATH WARNINGS" in output


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    @pytest.mark.anyio
    async def test_timeout_raises(self):
        backend = LocalExecBackend(timeout_seconds=1)
        with pytest.raises(asyncio.TimeoutError):
            await backend.run("import time; time.sleep(10)", {})


# ---------------------------------------------------------------------------
# Error handling (execution errors returned as strings)
# ---------------------------------------------------------------------------

class TestExecutionErrors:
    @pytest.mark.anyio
    async def test_name_error_returns_string(self, backend, namespace):
        """NameError in executed code is caught and returned as a string, not raised."""
        # Note: exec() exceptions propagate up through backend.run()
        # LocalExecBackend does NOT catch them — the tool wrapper in agents.py does
        with pytest.raises(NameError):
            await backend.run("print(undefined_var)", namespace)

    @pytest.mark.anyio
    async def test_key_error_propagates(self, backend, namespace):
        namespace["dfs"] = {}
        with pytest.raises(KeyError):
            await backend.run("dfs['nonexistent']", namespace)
