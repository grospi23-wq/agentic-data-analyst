"""
Data Profiling & Statistical Library for Pandas DataFrames.
Provides rich, agent-readable Pydantic models for statistical insights.
"""

from typing import List, Dict, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field


class OutlierReport(BaseModel):
    """Detailed report for outliers in a specific column."""
    column: str
    outlier_count: int
    outlier_pct: float
    q1: float
    q3: float
    sample_values: List[float]
    severity: Literal["low", "medium", "high"]


class MissingDataReport(BaseModel):
    """Report for missing data across the dataframe."""
    total_missing: int
    column_stats: Dict[str, Dict[str, float]] # {col: {"count": X, "pct": Y}}
    critical_columns: List[str] # Columns with > 50% missing


class ProfilerReport(BaseModel):
    """Consolidated statistical report for an analyst agent."""
    outliers: List[OutlierReport]
    missing_data: MissingDataReport
    correlations: Dict[str, Dict[str, float]]


from typing import List, Dict, Literal, Optional, cast, Any

def detect_outliers(df: pd.DataFrame, column: str) -> OutlierReport:
    """Detect outliers using IQR and return a structured report."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")
    
    col_data = df[column].dropna()
    
    # Cast to Any to bypass Pandas return type ambiguity (Series | float)
    q1_raw = col_data.quantile(0.25)
    q3_raw = col_data.quantile(0.75)
    
    q1 = float(cast(Any, q1_raw))
    q3 = float(cast(Any, q3_raw))
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filtering outliers
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    count = int(len(outliers))
    pct = (count / len(df)) * 100
    
    severity: Literal["low", "medium", "high"] = "low"
    if pct > 10:
        severity = "high"
    elif pct > 5:
        severity = "medium"
        
    # Direct slicing [:5] is compatible with both ndarray and Series
    # This avoids the 'iloc is unknown' error while maintaining type safety for Pydantic
    sample_values = [float(x) for x in outliers[:5].tolist()]

    return OutlierReport(
        column=column,
        outlier_count=count,
        outlier_pct=round(pct, 2),
        q1=round(q1, 2),
        q3=round(q3, 2),
        sample_values=sample_values,
        severity=severity
    )


def get_full_profile(df: pd.DataFrame, skip_discrete_outliers: bool = True) -> ProfilerReport:
    """Run a comprehensive profiling suite and return a structured report."""
    
    # 1. Analyze Outliers (Bug #3 Fix: Filtering low-cardinality numeric columns)
    outlier_reports = []
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Skip columns like status codes where IQR is meaningless
        if skip_discrete_outliers and int(cast(Any, df[col].nunique())) < 20:
            continue
        outlier_reports.append(detect_outliers(df, col))
        
    # 2. Analyze Missing Data (Hardened for strict type checkers)
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    
    # Using cast(Any, ...) to resolve ambiguity between Series and Scalar values
    col_stats = {
        str(col): {
            "count": float(int(cast(Any, missing_counts[col]))),
            "pct": float(round(cast(Any, missing_pcts[col]), 2))
        }
        for col in df.columns
    }
    
    # Type-safe list comprehension for critical columns
    critical = [
        str(col) for col, pct in missing_pcts.items() 
        if float(cast(Any, pct)) > 50
    ]
    
    missing_report = MissingDataReport(
        total_missing=int(cast(Any, missing_counts.sum())),
        column_stats=col_stats,
        critical_columns=critical
    )
    
    # 3. Correlations (Directly converted to dict)
    corr_matrix = df.corr(numeric_only=True).to_dict()
    
    return ProfilerReport(
        outliers=outlier_reports,
        missing_data=missing_report,
        correlations=corr_matrix
    )


__all__ = [
    "ProfilerReport",
    "OutlierReport",
    "MissingDataReport",
    "get_full_profile",
    "detect_outliers",
]