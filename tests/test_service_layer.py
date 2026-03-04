"""
test_service_layer.py
---------------------
Tests for service_layer.py — _enforce_size_guard and the routing logic
in run_analysis_pipeline.

All agent calls are mocked so no real LLM requests are made.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from service_layer import _enforce_size_guard, run_analysis_pipeline


# ---------------------------------------------------------------------------
# _enforce_size_guard
# ---------------------------------------------------------------------------

class TestEnforceSizeGuard:
    def test_small_file_does_not_raise(self, tmp_path):
        small_file = tmp_path / "small.csv"
        small_file.write_text("id,value\n1,100\n2,200\n")
        _enforce_size_guard(small_file)  # Should not raise

    def test_file_at_exactly_50mb_does_not_raise(self, tmp_path):
        test_file = tmp_path / "exact.csv"
        test_file.write_text("x")

        mock_stat = MagicMock()
        mock_stat.st_size = int(50.0 * 1024 * 1024)  # exactly 50.0 MB

        with patch.object(Path, "stat", return_value=mock_stat):
            _enforce_size_guard(test_file)  # Should not raise

    def test_file_just_over_50mb_raises(self, tmp_path):
        test_file = tmp_path / "big.csv"
        test_file.write_text("x")

        mock_stat = MagicMock()
        mock_stat.st_size = int(50.1 * 1024 * 1024)  # 50.1 MB

        with patch.object(Path, "stat", return_value=mock_stat):
            with pytest.raises(ValueError, match="File too large"):
                _enforce_size_guard(test_file)

    def test_error_message_includes_size_info(self, tmp_path):
        test_file = tmp_path / "huge.csv"
        test_file.write_text("x")

        mock_stat = MagicMock()
        mock_stat.st_size = 75 * 1024 * 1024  # 75 MB

        with patch.object(Path, "stat", return_value=mock_stat):
            with pytest.raises(ValueError) as exc_info:
                _enforce_size_guard(test_file)
            assert "75" in str(exc_info.value) or "50" in str(exc_info.value)

    def test_error_message_mentions_50mb_cap(self, tmp_path):
        test_file = tmp_path / "x.csv"
        test_file.write_text("x")

        mock_stat = MagicMock()
        mock_stat.st_size = 100 * 1024 * 1024

        with patch.object(Path, "stat", return_value=mock_stat):
            with pytest.raises(ValueError, match="50"):
                _enforce_size_guard(test_file)


# ---------------------------------------------------------------------------
# run_analysis_pipeline — routing
# ---------------------------------------------------------------------------

MOCK_RESULT = {"narrative": "Mocked analysis result", "critic_score": 0.8}


class TestRunAnalysisPipelineRouting:
    @pytest.mark.anyio
    async def test_csv_routes_to_single_sheet(self, tmp_path):
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("id,amount\n1,100\n2,200\n")

        with patch(
            "service_layer.execute_analysis_mission",
            new=AsyncMock(return_value=MOCK_RESULT),
        ) as mock_single:
            result = await run_analysis_pipeline(str(csv_file))

        mock_single.assert_called_once()
        assert result == MOCK_RESULT

    @pytest.mark.anyio
    async def test_csv_uses_stem_as_sheet_name(self, tmp_path):
        csv_file = tmp_path / "my_dataset.csv"
        csv_file.write_text("a,b\n1,2\n")

        with patch(
            "service_layer.execute_analysis_mission",
            new=AsyncMock(return_value=MOCK_RESULT),
        ) as mock_single:
            await run_analysis_pipeline(str(csv_file))

        call_kwargs = mock_single.call_args
        # target_sheet should be the file stem
        assert "my_dataset" in str(call_kwargs)

    @pytest.mark.anyio
    async def test_xlsx_with_target_sheet_routes_to_single_sheet(self, tmp_path):
        xlsx_file = tmp_path / "workbook.xlsx"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sales", index=False)
            df.to_excel(writer, sheet_name="Inventory", index=False)

        with patch(
            "service_layer.execute_analysis_mission",
            new=AsyncMock(return_value=MOCK_RESULT),
        ) as mock_single:
            with patch("service_layer.execute_multi_sheet_mission", new=AsyncMock()) as mock_multi:
                result = await run_analysis_pipeline(str(xlsx_file), target_sheet="Sales")

        mock_single.assert_called_once()
        mock_multi.assert_not_called()
        assert result == MOCK_RESULT

    @pytest.mark.anyio
    async def test_xlsx_single_sheet_auto_routes_to_single(self, tmp_path):
        xlsx_file = tmp_path / "single.xlsx"
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="OnlySheet", index=False)

        with patch(
            "service_layer.execute_analysis_mission",
            new=AsyncMock(return_value=MOCK_RESULT),
        ) as mock_single:
            with patch("service_layer.execute_multi_sheet_mission", new=AsyncMock()) as mock_multi:
                result = await run_analysis_pipeline(str(xlsx_file))

        mock_single.assert_called_once()
        mock_multi.assert_not_called()

    @pytest.mark.anyio
    async def test_xlsx_multi_sheet_routes_to_multi_sheet(self, tmp_path):
        xlsx_file = tmp_path / "multi.xlsx"
        df = pd.DataFrame({"x": [1, 2]})
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            df.to_excel(writer, sheet_name="Sheet2", index=False)
            df.to_excel(writer, sheet_name="Sheet3", index=False)

        with patch("service_layer.execute_analysis_mission", new=AsyncMock()) as mock_single:
            with patch(
                "service_layer.execute_multi_sheet_mission",
                new=AsyncMock(return_value=MOCK_RESULT),
            ) as mock_multi:
                result = await run_analysis_pipeline(str(xlsx_file))

        mock_multi.assert_called_once()
        mock_single.assert_not_called()
        assert result == MOCK_RESULT

    @pytest.mark.anyio
    async def test_size_guard_enforced_before_routing(self, tmp_path):
        """A file over 50MB is rejected before any routing logic runs."""
        csv_file = tmp_path / "huge.csv"
        csv_file.write_text("x")

        mock_stat = MagicMock()
        mock_stat.st_size = 100 * 1024 * 1024  # 100 MB

        with patch.object(Path, "stat", return_value=mock_stat):
            with pytest.raises(ValueError, match="File too large"):
                await run_analysis_pipeline(str(csv_file))
