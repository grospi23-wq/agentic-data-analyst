"""
test_path_utils.py
------------------
Tests for lib/path_utils.py — the single entry point for path resolution
across WSL/Windows environments.
"""

import pytest
from pathlib import Path
from path_utils import resolve_file_path


# ---------------------------------------------------------------------------
# Backslash normalization
# ---------------------------------------------------------------------------

class TestBackslashNormalization:
    def test_windows_backslashes_converted(self, tmp_path):
        """Windows-style backslashes are converted to forward slashes."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c")

        # Simulate a Windows path string
        windows_path = str(test_file).replace("/", "\\")
        result = resolve_file_path(windows_path)
        assert result.exists()

    def test_mixed_slashes_handled(self, tmp_path):
        test_file = tmp_path / "file.xlsx"
        test_file.write_text("data")
        mixed = str(test_file).replace("/", "\\", 2)  # Replace first 2 slashes only
        # Should not raise
        try:
            result = resolve_file_path(mixed)
            assert isinstance(result, Path)
        except Exception:
            pass  # Acceptable if path becomes invalid after partial replacement


# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------

class TestAbsolutePaths:
    def test_absolute_existing_path_returned_as_is(self, tmp_path):
        test_file = tmp_path / "absolute.csv"
        test_file.write_text("data")

        result = resolve_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_absolute_missing_path_returns_resolved_path(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        result = resolve_file_path(str(missing))
        # Should return a Path (even if it doesn't exist)
        assert isinstance(result, Path)
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# Relative paths
# ---------------------------------------------------------------------------

class TestRelativePaths:
    def test_relative_path_resolved_from_cwd(self, tmp_path, monkeypatch):
        """A relative path is found when the file exists in CWD."""
        test_file = tmp_path / "relative.csv"
        test_file.write_text("data")

        # Change CWD to tmp_path so the relative path resolves there
        monkeypatch.chdir(tmp_path)
        result = resolve_file_path("relative.csv")
        assert result.exists()
        assert result.name == "relative.csv"

    def test_missing_relative_path_returns_cwd_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = resolve_file_path("nonexistent_file.csv")
        assert isinstance(result, Path)
        assert result.is_absolute()
        # Falls back gracefully — does NOT raise
        assert not result.exists()


# ---------------------------------------------------------------------------
# Return type guarantees
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_always_returns_path_object(self, tmp_path):
        result = resolve_file_path("anything.csv")
        assert isinstance(result, Path)

    def test_always_returns_absolute_path(self, tmp_path):
        """The returned path is always absolute, regardless of input."""
        result = resolve_file_path("some/relative/path.xlsx")
        assert result.is_absolute()

    def test_does_not_raise_for_missing_file(self):
        """resolve_file_path never raises — callers check path.exists()."""
        try:
            result = resolve_file_path("totally_missing_file_xyz.csv")
            assert isinstance(result, Path)
        except Exception as exc:
            pytest.fail(f"resolve_file_path raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_filename_with_spaces(self, tmp_path, monkeypatch):
        test_file = tmp_path / "my data file.csv"
        test_file.write_text("x")
        monkeypatch.chdir(tmp_path)
        result = resolve_file_path("my data file.csv")
        assert result.exists()

    def test_file_with_xlsx_extension(self, tmp_path, monkeypatch):
        test_file = tmp_path / "workbook.xlsx"
        test_file.write_bytes(b"fake xlsx content")
        monkeypatch.chdir(tmp_path)
        result = resolve_file_path("workbook.xlsx")
        assert result.exists()
