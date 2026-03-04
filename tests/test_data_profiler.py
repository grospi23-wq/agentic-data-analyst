"""Unit tests for the data_profiler module."""

import pytest
import pandas as pd
import numpy as np

from data_profiler import (
    detect_outliers,
    get_full_profile
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 100, 4, 5, 2, 3, 4, 2],
        'numeric2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'categorical': ['A', 'B', 'C', 'D', None, 'F', 'G', 'H', 'I', 'J'],
        'mostly_missing': [np.nan] * 8 + [1, 2]
    })

def test_detect_outliers(sample_df):
    result = detect_outliers(sample_df, 'numeric1')
    
    assert hasattr(result, 'column')
    assert hasattr(result, 'outlier_count')
    assert hasattr(result, 'severity')
    
    assert result.column == 'numeric1'
    assert result.outlier_count > 0
    assert result.severity in ['low', 'medium', 'high']

    with pytest.raises(KeyError):
        detect_outliers(sample_df, 'nonexistent')

def test_get_full_profile(sample_df):
    # skip_discrete_outliers=False ensures small numeric columns are profiled
    result = get_full_profile(sample_df, skip_discrete_outliers=False)

    assert hasattr(result, 'outliers')
    assert hasattr(result, 'missing_data')
    assert hasattr(result, 'correlations')

    assert result.missing_data.total_missing > 0
    assert 'mostly_missing' in result.missing_data.critical_columns
    assert len(result.outliers) > 0

def test_type_safety():
    invalid_input = "not a dataframe"
    with pytest.raises((TypeError, ValueError, AttributeError)):
        detect_outliers(invalid_input, 'numeric1')
    with pytest.raises((TypeError, ValueError, AttributeError)):
        get_full_profile(invalid_input)