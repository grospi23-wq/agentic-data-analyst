"""
Discovery layer utilities for Phase 3.

This module produces a lightweight, structured metadata map over one dataset source
(CSV file or Excel workbook). It is intentionally limited to **Discovery** concerns:

- Build per-sheet / per-table column metadata (types, semantic categories, samples).
- Detect likely cross-sheet relationships for downstream join planning.

Downstream layers (Strategy and Mission) consume the resulting `GlobalDiscoveryMap`.
"""
from typing import Dict, List, Literal, cast
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
import numpy as np
from sqlalchemy import inspect, text

ColumnCategory = Literal[
    "numeric_continuous", 
    "numeric_discrete", 
    "categorical_low_cardinality", 
    "categorical_high_cardinality", 
    "datetime", 
    "free_text", 
    "id_like"
]

class ColumnMetadata(BaseModel):
    """Metadata for a single column in a dataset."""
    name: str
    dtype: str
    category: ColumnCategory
    sample_values: List[str] = Field(default_factory=list)

class SheetMetadata(BaseModel):
    """Metadata for a single sheet or CSV file."""
    name: str
    columns: List[ColumnMetadata]
    row_count: int
    sample_rows: str = Field(default="", description="Markdown representation of a data sample")


class ColumnRelationship(BaseModel):
    """Represents a potential link between two different sheets."""
    sheet_a: str
    column_a: str
    sheet_b: str
    column_b: str
    confidence: float = Field(..., ge=0, le=1)  # Probability-like score in [0.0, 1.0]
    detection_method: Literal["name_match", "value_overlap", "combined"]
    cardinality: Literal["one_to_many", "many_to_many", "one_to_one", "unknown"] = "unknown"
    join_type_hint: Literal["inner", "left", "unknown"] = "unknown"

class GlobalDiscoveryMap(BaseModel):
    """
    Consolidated Discovery output for a single dataset source.

    For Excel, this includes all sheets; for CSV, this includes one virtual sheet keyed
    by the filename stem. Relationship fields are best-effort heuristics to support the
    Strategy layer (e.g., join planning) and should be treated as suggestions.
    """
    source_path: str
    source_type: str  # "excel" or "csv"
    sheets: Dict[str, SheetMetadata]
    total_rows: int
    
    # Phase 3 additions: cross-sheet relationship hints for Strategy.
    relationships: List[ColumnRelationship] = Field(default_factory=list)
    suggested_joins: List[str] = Field(default_factory=list)  # Human-readable for the CDO

def _classify_column(series: pd.Series) -> ColumnCategory:
    """
    Classify a column into coarse semantic categories for Strategy planning.
    """
    name_lower = str(series.name).lower() if series.name else ""
    unique_count = series.nunique()
    total_count = len(series)

    # 1. Datetime check is safe to do first
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
        
    # 2. Check 1: Explicit ID name indicator (MUST be before general numeric checks)
    id_suffixes = ("_id", "_key", "_code", "_num", "_no")
    if name_lower.endswith(id_suffixes):
        return "id_like"
        
    # 3. Numeric handling (including Check 2 for implicit high-cardinality IDs)
    if pd.api.types.is_numeric_dtype(series):
        cardinality_ratio = unique_count / total_count if total_count > 0 else 0
        
        # High cardinality threshold for unlabeled IDs
        if cardinality_ratio > 0.6 and unique_count > 50:
            return "id_like"
            
        if unique_count < 20:
            return "numeric_discrete"
            
        return "numeric_continuous"
        
    # 4. Categorical and text handling fallback
    if unique_count < 15:
        return "categorical_low_cardinality"
        
    if unique_count > 100:
        return "free_text"
        
    return "categorical_high_cardinality"

def _create_sheet_metadata(df: pd.DataFrame, sheet_name: str) -> SheetMetadata:
    """Create metadata for a single sheet or CSV file."""
    columns = []
    for col in df.columns:
        series = cast(pd.Series, df[col])
        sample_values = series.dropna().head(3).apply(str).tolist()
        
        columns.append(ColumnMetadata(
            name=str(col),
            dtype=str(series.dtype),
            category=_classify_column(series),
            sample_values=sample_values
        ))

    return SheetMetadata(
        name=sheet_name,
        columns=columns,
        row_count=len(df),
        sample_rows=get_semantic_sample(df, n_rows=5)
    )
def _detect_by_name(sheets_meta: Dict[str, SheetMetadata]) -> List[ColumnRelationship]:
    """
    Find relationship candidates by column-name matching (fast, low-cost heuristic).
    """
    relationships = []
    sheet_names = list(sheets_meta.keys())

    # Nested loop to compare every sheet with every other sheet
    for i, name_a in enumerate(sheet_names):
        for name_b in sheet_names[i + 1:]:
            # Get column lists for both sheets
            cols_a = {c.name.lower(): c for c in sheets_meta[name_a].columns}
            cols_b = {c.name.lower(): c for c in sheets_meta[name_b].columns}
            
            # Find shared column names (case-insensitive match)
            shared_names = set(cols_a.keys()) & set(cols_b.keys())

            for col_name in shared_names:
                meta_a = cols_a[col_name]
                meta_b = cols_b[col_name]

                # We only care about joinable types (IDs, discrete numbers, etc.)
                # This prevents joining on random 'Notes' or 'Amount' columns
                joinable_categories = ["id_like", "numeric_discrete", "categorical_low_cardinality"]
                if meta_a.category not in joinable_categories:
                    continue

                # Heuristic: Boost confidence for specific suffixes
                base_confidence = 0.7
                if col_name.endswith(("_id", "_key", "_code", "pk", "fk")):
                    base_confidence = 0.9

                relationships.append(ColumnRelationship(
                    sheet_a=name_a,
                    column_a=meta_a.name,
                    sheet_b=name_b,
                    column_b=meta_b.name,
                    confidence=base_confidence,
                    detection_method="name_match"
                ))

    return relationships

def _verify_by_value_overlap(
    dfs: Dict[str, pd.DataFrame],
    candidates: List[ColumnRelationship],
    sample_size: int = 500  # Increased sample for better statistical significance
) -> List[ColumnRelationship]:
    """
    Verify relationship candidates using sampled value overlap and uniqueness checks.

    This step intentionally trades completeness for speed; it is a heuristic designed
    to inform join planning, not to guarantee referential integrity.
    """
    verified = []

    for rel in candidates:
        if rel.sheet_a not in dfs or rel.sheet_b not in dfs:
            verified.append(rel)
            continue

        df_a = dfs[rel.sheet_a]
        df_b = dfs[rel.sheet_b]

        if rel.column_a not in df_a.columns or rel.column_b not in df_b.columns:
            continue

        # 1. Prepare data (dropping NaNs is crucial for ID matching)
        col_a = df_a[rel.column_a].dropna()
        col_b = df_b[rel.column_b].dropna()

        if col_a.empty or col_b.empty:
            continue

        # 2. Compute Value Overlap using samples
        sample_a = set(col_a.sample(min(sample_size, len(col_a)), random_state=42).astype(str))
        sample_b = set(col_b.sample(min(sample_size, len(col_b)), random_state=42).astype(str))
        
        common_values = sample_a.intersection(sample_b)
        overlap_ratio = len(common_values) / min(len(sample_a), len(sample_b))

        # Threshold: If less than 15% overlap, it's likely a false positive
        if overlap_ratio < 0.15:
            continue

        # 3. Optimal Cardinality Detection (The Uniqueness Method)
        # We check if the columns act as keys in their respective tables
        is_unique_a = col_a.is_unique
        is_unique_b = col_b.is_unique

        if is_unique_a and not is_unique_b:
            cardinality = "one_to_many"
        elif not is_unique_a and not is_unique_b:
            cardinality = "many_to_many"
        elif is_unique_a and is_unique_b:
            cardinality = "one_to_one"
        else:
            cardinality = "unknown"

        # 4. Join Type Hinting (Inclusion Dependency)
        # If nearly all values in B are present in A, it's a Foreign Key relationship
        b_in_a_ratio = len(sample_b.intersection(sample_a)) / len(sample_b)
        join_hint = "inner" if b_in_a_ratio > 0.9 else "left"

        # 5. Final Confidence Scoring
        # Start with name match confidence and boost based on actual data overlap
        final_confidence = min(1.0, rel.confidence + (overlap_ratio * 0.15))

        verified.append(rel.model_copy(update={
            "confidence": final_confidence,
            "detection_method": "combined",
            "cardinality": cardinality,
            "join_type_hint": join_hint,
        }))

    return sorted(verified, key=lambda r: r.confidence, reverse=True)
    
def run_dataset_discovery(file_path: str | Path) -> GlobalDiscoveryMap:
    """
    Run dataset discovery for a single source (CSV or Excel workbook).

    Returns:
        A `GlobalDiscoveryMap` containing per-sheet metadata and (when applicable)
        cross-sheet relationship hints for the Strategy layer.
    """
    file_path = Path(file_path)
    dfs: Dict[str, pd.DataFrame] = {}
    sheets_meta: Dict[str, SheetMetadata] = {}
    
    # 1. Load data and extract per-sheet metadata.
    if file_path.suffix.lower() == ".xlsx":
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            dfs[sheet_name] = df
            # Reuse the same per-sheet metadata extraction logic across formats.
            sheets_meta[sheet_name] = _create_sheet_metadata(df, sheet_name)
        source_type = "excel"
    else:
        # CSV handling (single sheet)
        df = pd.read_csv(file_path)
        sheet_name = file_path.stem
        dfs[sheet_name] = df
        sheets_meta[sheet_name] = _create_sheet_metadata(df, sheet_name)
        source_type = "csv"

    # 2. Relationship detection (Phase 3): fast candidate generation → slower verification.
    # First pass: name matching (fast).
    candidates = _detect_by_name(sheets_meta)
    
    # Second pass: value overlap verification (more accurate, still heuristic).
    verified_relationships = _verify_by_value_overlap(dfs, candidates)

    # 3. Build human-readable suggestions for the CDO prompt.
    join_suggestions = [
        f"{r.sheet_a} {r.join_type_hint.upper()} JOIN {r.sheet_b} ON {r.column_a} (confidence: {r.confidence:.0%})"
        for r in verified_relationships if r.confidence > 0.6
    ]

    # 4. Return the consolidated GlobalDiscoveryMap.
    return GlobalDiscoveryMap(
        source_path=str(file_path),
        source_type=source_type,
        sheets=sheets_meta,
        total_rows=sum(m.row_count for m in sheets_meta.values()),
        relationships=verified_relationships,
        suggested_joins=join_suggestions
    )

def get_semantic_sample(df: pd.DataFrame, n_rows: int = 5) -> str:
    """
    Generates a token-efficient Markdown sample of the dataframe.
    Optimized for LLM semantic understanding of cross-sheet relationships.
    """
    if df.empty:
        return "Empty DataFrame"

    # 1. Representative sampling (better than .head() for Discovery prompts).
    # random_state ensures consistent results across agent calls.
    sample_size = min(n_rows, len(df))
    sample_df = df.sample(n=sample_size, random_state=42).copy()

    # 2. Token optimization: round floats to reduce numeric noise while preserving intent.
    for col in sample_df.select_dtypes(include=['float']).columns:
        sample_df[col] = sample_df[col].round(4)

    # 3. Markdown formatting keeps the sample compact and LLM-friendly.
    # index=False avoids confusing index values with source data.
    return sample_df.to_markdown(index=False) # type: ignore

def run_sql_discovery(engine, source_path: str) -> GlobalDiscoveryMap:
    """
    Run dataset discovery for a SQL database using SQLAlchemy.
    Maps tables to 'sheets' to maintain compatibility with the multi-sheet pipeline.
    """
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    dfs: Dict[str, pd.DataFrame] = {}
    sheets_meta: Dict[str, SheetMetadata] = {}
    total_rows = 0

    for table in table_names:
        # Quote table name to handle reserved keywords and spaces safely
        quoted_table = engine.dialect.identifier_preparer.quote(table)
        
        # 1. Load a bounded sample for metadata and value overlap verification
        sample_query = text(f"SELECT * FROM {quoted_table} LIMIT 500")
        sample_df = pd.read_sql(sample_query, engine)
        dfs[table] = sample_df
        
        # 2. Generate column metadata using existing logic
        meta = _create_sheet_metadata(sample_df, table)
        
        # 3. Get exact row count for the metadata
        count_query = text(f"SELECT COUNT(*) FROM {quoted_table}")
        count_df = pd.read_sql(count_query, engine)
        meta.row_count = int(count_df.iloc[0, 0])
        
        sheets_meta[table] = meta
        total_rows += meta.row_count

    # Reuse Phase 3 relationship detection
    candidates = _detect_by_name(sheets_meta)
    verified_relationships = _verify_by_value_overlap(dfs, candidates)

    join_suggestions = [
        f"{r.sheet_a} {r.join_type_hint.upper()} JOIN {r.sheet_b} ON {r.column_a} (confidence: {r.confidence:.0%})"
        for r in verified_relationships if r.confidence > 0.6
    ]

    return GlobalDiscoveryMap(
        source_path=source_path,
        source_type="sql",
        sheets=sheets_meta,
        total_rows=total_rows,
        relationships=verified_relationships,
        suggested_joins=join_suggestions
    )

__all__ = [
    "run_dataset_discovery",
    "run_sql_discovery",
    "get_semantic_sample",
    "GlobalDiscoveryMap",
    "SheetMetadata",
    "ColumnMetadata",
    "ColumnRelationship",
    "ColumnCategory",
]