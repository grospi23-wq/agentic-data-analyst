"""
Unit tests for the data discovery library.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from data_discovery_lib import (
    run_dataset_discovery,
    get_semantic_sample,
    GlobalDiscoveryMap,
    SheetMetadata,
    ColumnMetadata,
)


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': pd.date_range('2023-01-01', periods=3)
        })
        df.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def sample_excel_file():
    """Create a temporary Excel file with multiple sheets for testing."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        with pd.ExcelWriter(f.name) as writer:
            df1 = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
            df2 = pd.DataFrame({'C': [3, 4], 'D': ['z', 'w']})
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
        yield f.name
    Path(f.name).unlink()


def test_run_dataset_discovery_csv(sample_csv_file):
    """Test run_dataset_discovery with CSV file."""
    result = run_dataset_discovery(sample_csv_file)

    assert isinstance(result, GlobalDiscoveryMap)
    assert result.source_type == "csv"
    assert len(result.sheets) == 1
    assert result.total_rows == 3

    sheet = list(result.sheets.values())[0]
    assert len(sheet.columns) == 3
    assert sheet.row_count == 3


def test_run_dataset_discovery_excel(sample_excel_file):
    """Test run_dataset_discovery with Excel file."""
    result = run_dataset_discovery(sample_excel_file)

    assert isinstance(result, GlobalDiscoveryMap)
    assert result.source_type == "excel"
    assert len(result.sheets) == 2
    assert result.total_rows == 4

    assert "Sheet1" in result.sheets
    assert "Sheet2" in result.sheets
    assert result.sheets["Sheet1"].row_count == 2
    assert result.sheets["Sheet2"].row_count == 2


def test_run_dataset_discovery_invalid_file():
    """Test run_dataset_discovery with invalid file."""
    with pytest.raises(FileNotFoundError):
        run_dataset_discovery("nonexistent.xlsx")


def test_get_semantic_sample():
    """Test get_semantic_sample returns a markdown string."""
    df = pd.DataFrame({
        "int_col": [1, 2, 3, 4, 5, 6],
        "float_col": [1.23456, 2.34567, 3.45678, 4.56789, 5.67890, 6.78901],
        "str_col": ["a", "b", "c", "d", "e", "f"],
    })

    result = get_semantic_sample(df, n_rows=3)

    assert isinstance(result, str)
    assert "|" in result
    assert "int_col" in result
    assert "float_col" in result
    assert "str_col" in result


def test_get_semantic_sample_invalid_input():
    """Test get_semantic_sample with non-DataFrame input."""
    with pytest.raises(AttributeError):
        get_semantic_sample([1, 2, 3])  # Not a DataFrame