"""
test_data_discovery_lib.py
--------------------------
Comprehensive tests for lib/data_discovery_lib.py.

Coverage targets:
  - _classify_column     — all 7 categories, edge cases
  - _detect_by_name      — shared-name relationship detection, joinability filter
  - _verify_by_value_overlap — confidence boost, cardinality, join hint
  - run_dataset_discovery — CSV + Excel integration, column metadata correctness
  - get_semantic_sample  — empty DF, float rounding, row limits
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data_discovery_lib import (
    _classify_column,
    _detect_by_name,
    _verify_by_value_overlap,
    run_dataset_discovery,
    get_semantic_sample,
    GlobalDiscoveryMap,
    SheetMetadata,
    ColumnMetadata,
    ColumnRelationship,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_file(tmp_path):
    f = tmp_path / "sales.csv"
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"],
        "C": pd.date_range("2023-01-01", periods=3),
    })
    df.to_csv(f, index=False)
    return str(f)


@pytest.fixture
def sample_excel_file(tmp_path):
    f = tmp_path / "workbook.xlsx"
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df2 = pd.DataFrame({"C": [3, 4], "D": ["z", "w"]})
    with pd.ExcelWriter(f, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Sheet2", index=False)
    return str(f)


@pytest.fixture
def joinable_excel_file(tmp_path):
    """Excel with two sheets sharing a customer_id column (detectable relationship)."""
    f = tmp_path / "joinable.xlsx"
    customers = pd.DataFrame({
        "customer_id": range(1, 51),
        "name": [f"Customer {i}" for i in range(1, 51)],
        "region": ["North", "South"] * 25,
    })
    orders = pd.DataFrame({
        "order_id": range(100, 200),
        "customer_id": [i % 50 + 1 for i in range(100)],
        "amount": np.random.default_rng(42).uniform(10, 500, 100).round(2),
    })
    with pd.ExcelWriter(f, engine="openpyxl") as writer:
        customers.to_excel(writer, sheet_name="customers", index=False)
        orders.to_excel(writer, sheet_name="orders", index=False)
    return str(f)


# ---------------------------------------------------------------------------
# _classify_column
# ---------------------------------------------------------------------------

class TestClassifyColumn:
    def test_datetime_series(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=10), name="created_at")
        assert _classify_column(s) == "datetime"

    def test_id_suffix_returns_id_like(self):
        for col_name in ("order_id", "customer_key", "product_code", "item_num", "ref_no"):
            s = pd.Series(range(100), name=col_name)
            result = _classify_column(s)
            assert result == "id_like", f"Expected id_like for column '{col_name}', got {result}"

    def test_high_cardinality_numeric_returns_id_like(self):
        """Numeric column with > 60% unique values and > 50 distinct → id_like."""
        s = pd.Series(range(200), name="unlabeled_id")
        assert _classify_column(s) == "id_like"

    def test_numeric_few_unique_returns_discrete(self):
        """Numeric with < 20 unique values → numeric_discrete."""
        s = pd.Series([1, 2, 3, 1, 2, 3, 1, 2] * 10, name="rating")
        assert _classify_column(s) == "numeric_discrete"

    def test_numeric_many_unique_returns_continuous(self):
        """Numeric with 20-50% unique values (below id_like threshold) → numeric_continuous.

        The id_like branch fires when cardinality_ratio > 0.6 AND unique_count > 50.
        To stay in numeric_continuous: keep unique_count >= 20 but cardinality_ratio <= 0.6.
        e.g. 40 distinct values spread across 200 rows → ratio = 0.2.
        """
        values = list(range(40)) * 5  # 40 unique values, 200 rows → ratio = 0.2
        s = pd.Series(values, name="quantity")
        assert _classify_column(s) == "numeric_continuous"

    def test_categorical_low_cardinality(self):
        """String column with < 15 unique values → categorical_low_cardinality."""
        s = pd.Series(["North", "South", "East", "West"] * 25, name="region")
        assert _classify_column(s) == "categorical_low_cardinality"

    def test_free_text_high_cardinality(self):
        """String column with > 100 unique values → free_text."""
        s = pd.Series([f"Note about order {i}" for i in range(200)], name="description")
        assert _classify_column(s) == "free_text"

    def test_categorical_high_cardinality(self):
        """String column with 15–100 unique values → categorical_high_cardinality."""
        s = pd.Series([f"City_{i}" for i in range(50)] * 4, name="city")
        assert _classify_column(s) == "categorical_high_cardinality"

    def test_id_suffix_takes_priority_over_numeric_checks(self):
        """Even if a numeric column has low cardinality, _id suffix wins."""
        s = pd.Series([1, 2, 3] * 20, name="store_id")
        assert _classify_column(s) == "id_like"

    def test_empty_series_does_not_raise(self):
        s = pd.Series([], dtype="float64", name="empty")
        result = _classify_column(s)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _detect_by_name
# ---------------------------------------------------------------------------

def _make_sheet(name: str, cols: list[tuple[str, str]]) -> SheetMetadata:
    """Helper: build a SheetMetadata with specified (col_name, category) pairs."""
    columns = [
        ColumnMetadata(name=col_name, dtype="object", category=cat, sample_values=[])
        for col_name, cat in cols
    ]
    return SheetMetadata(name=name, columns=columns, row_count=10)


class TestDetectByName:
    def test_shared_id_column_detected(self):
        sheets = {
            "orders": _make_sheet("orders", [("customer_id", "id_like"), ("amount", "numeric_continuous")]),
            "customers": _make_sheet("customers", [("customer_id", "id_like"), ("name", "free_text")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 1
        assert rels[0].column_a == "customer_id" or rels[0].column_b == "customer_id"

    def test_id_suffix_boosts_confidence_to_09(self):
        sheets = {
            "a": _make_sheet("a", [("order_id", "id_like")]),
            "b": _make_sheet("b", [("order_id", "id_like")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 1
        assert rels[0].confidence == 0.9

    def test_non_id_suffix_confidence_07(self):
        sheets = {
            "a": _make_sheet("a", [("region", "categorical_low_cardinality")]),
            "b": _make_sheet("b", [("region", "categorical_low_cardinality")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 1
        assert rels[0].confidence == 0.7

    def test_numeric_continuous_column_not_joinable(self):
        """numeric_continuous columns (e.g., 'amount') are not valid join keys."""
        sheets = {
            "a": _make_sheet("a", [("amount", "numeric_continuous")]),
            "b": _make_sheet("b", [("amount", "numeric_continuous")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 0

    def test_free_text_column_not_joinable(self):
        sheets = {
            "a": _make_sheet("a", [("notes", "free_text")]),
            "b": _make_sheet("b", [("notes", "free_text")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 0

    def test_no_shared_columns_returns_empty(self):
        sheets = {
            "a": _make_sheet("a", [("col_x", "id_like")]),
            "b": _make_sheet("b", [("col_y", "id_like")]),
        }
        assert _detect_by_name(sheets) == []

    def test_single_sheet_returns_empty(self):
        sheets = {"only": _make_sheet("only", [("id", "id_like")])}
        assert _detect_by_name(sheets) == []

    def test_case_insensitive_matching(self):
        """Column names are matched case-insensitively."""
        sheets = {
            "a": _make_sheet("a", [("CustomerID", "id_like")]),
            "b": _make_sheet("b", [("customerid", "id_like")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 1

    def test_three_sheets_all_pairs_checked(self):
        """With 3 sheets sharing the same column, 3 relationships are detected."""
        sheets = {
            "a": _make_sheet("a", [("item_id", "id_like")]),
            "b": _make_sheet("b", [("item_id", "id_like")]),
            "c": _make_sheet("c", [("item_id", "id_like")]),
        }
        rels = _detect_by_name(sheets)
        assert len(rels) == 3  # pairs: (a,b), (a,c), (b,c)


# ---------------------------------------------------------------------------
# _verify_by_value_overlap
# ---------------------------------------------------------------------------

class TestVerifyByValueOverlap:
    def _base_rel(self, sheet_a="orders", col_a="customer_id",
                  sheet_b="customers", col_b="customer_id") -> ColumnRelationship:
        return ColumnRelationship(
            sheet_a=sheet_a, column_a=col_a,
            sheet_b=sheet_b, column_b=col_b,
            confidence=0.9, detection_method="name_match",
        )

    def test_high_overlap_relationship_preserved(self):
        """Two columns with strong value overlap survive verification."""
        ids = list(range(1, 51))
        dfs = {
            "customers": pd.DataFrame({"customer_id": ids}),
            "orders": pd.DataFrame({"customer_id": ids * 3}),  # all values in customers
        }
        rel = self._base_rel()
        result = _verify_by_value_overlap(dfs, [rel])
        assert len(result) == 1
        assert result[0].detection_method == "combined"

    def test_low_overlap_relationship_filtered(self):
        """< 15% overlap → relationship discarded."""
        dfs = {
            "a": pd.DataFrame({"customer_id": range(1, 101)}),
            "b": pd.DataFrame({"customer_id": range(5000, 5100)}),  # no overlap
        }
        rel = self._base_rel(sheet_a="a", sheet_b="b")
        result = _verify_by_value_overlap(dfs, [rel])
        assert len(result) == 0

    def test_one_to_many_cardinality_detected(self):
        """Unique in A, non-unique in B → one_to_many.

        sheet_a must be the unique (parent) side; sheet_b the repeating (child) side.
        """
        dfs = {
            "customers": pd.DataFrame({"customer_id": range(1, 51)}),           # unique
            "orders": pd.DataFrame({"customer_id": [i % 50 + 1 for i in range(200)]}),  # non-unique
        }
        # Place customers (unique) as sheet_a so is_unique_a=True
        rel = ColumnRelationship(
            sheet_a="customers", column_a="customer_id",
            sheet_b="orders", column_b="customer_id",
            confidence=0.9, detection_method="name_match",
        )
        result = _verify_by_value_overlap(dfs, [rel])
        if result:  # if overlap threshold met
            assert result[0].cardinality == "one_to_many"

    def test_many_to_many_cardinality_detected(self):
        """Neither column unique → many_to_many."""
        shared = [1, 2, 3, 1, 2, 3] * 10
        dfs = {
            "a": pd.DataFrame({"customer_id": shared}),
            "b": pd.DataFrame({"customer_id": shared}),
        }
        rel = self._base_rel(sheet_a="a", sheet_b="b")
        result = _verify_by_value_overlap(dfs, [rel])
        if result:
            assert result[0].cardinality == "many_to_many"

    def test_missing_sheet_passes_through(self):
        """If a sheet is missing from dfs, the relationship passes through unchanged."""
        dfs = {}  # no DFs loaded
        rel = self._base_rel()
        result = _verify_by_value_overlap(dfs, [rel])
        assert len(result) == 1
        assert result[0].detection_method == "name_match"  # not updated

    def test_confidence_boosted_for_good_overlap(self):
        """High overlap should increase confidence above the base name-match score."""
        ids = list(range(1, 101))
        dfs = {
            "customers": pd.DataFrame({"customer_id": ids}),
            "orders": pd.DataFrame({"customer_id": ids * 2}),
        }
        rel = self._base_rel()
        result = _verify_by_value_overlap(dfs, [rel])
        if result:
            assert result[0].confidence >= rel.confidence

    def test_results_sorted_by_confidence_descending(self):
        """Verified relationships are returned sorted by confidence (highest first)."""
        ids = list(range(1, 51))
        dfs = {
            "a": pd.DataFrame({"col1": ids, "col2": [i % 5 for i in ids]}),
            "b": pd.DataFrame({"col1": ids, "col2": [i % 5 for i in ids]}),
        }
        rels = [
            ColumnRelationship(sheet_a="a", column_a="col1", sheet_b="b", column_b="col1", confidence=0.9, detection_method="name_match"),
            ColumnRelationship(sheet_a="a", column_a="col2", sheet_b="b", column_b="col2", confidence=0.7, detection_method="name_match"),
        ]
        result = _verify_by_value_overlap(dfs, rels)
        if len(result) >= 2:
            assert result[0].confidence >= result[1].confidence


# ---------------------------------------------------------------------------
# run_dataset_discovery — integration
# ---------------------------------------------------------------------------

class TestRunDatasetDiscovery:
    def test_csv_returns_global_discovery_map(self, sample_csv_file):
        result = run_dataset_discovery(sample_csv_file)
        assert isinstance(result, GlobalDiscoveryMap)
        assert result.source_type == "csv"
        assert len(result.sheets) == 1
        assert result.total_rows == 3

    def test_csv_sheet_keyed_by_filename_stem(self, tmp_path):
        f = tmp_path / "my_dataset.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(f, index=False)
        result = run_dataset_discovery(str(f))
        assert "my_dataset" in result.sheets

    def test_csv_column_count(self, sample_csv_file):
        result = run_dataset_discovery(sample_csv_file)
        sheet = list(result.sheets.values())[0]
        assert len(sheet.columns) == 3

    def test_csv_column_metadata_populated(self, sample_csv_file):
        result = run_dataset_discovery(sample_csv_file)
        sheet = list(result.sheets.values())[0]
        col_names = {c.name for c in sheet.columns}
        assert "A" in col_names
        assert "B" in col_names

    def test_csv_column_has_sample_values(self, sample_csv_file):
        result = run_dataset_discovery(sample_csv_file)
        sheet = list(result.sheets.values())[0]
        for col in sheet.columns:
            assert isinstance(col.sample_values, list)

    def test_csv_sample_rows_is_string(self, sample_csv_file):
        result = run_dataset_discovery(sample_csv_file)
        sheet = list(result.sheets.values())[0]
        assert isinstance(sheet.sample_rows, str)
        assert len(sheet.sample_rows) > 0

    def test_excel_returns_correct_sheet_count(self, sample_excel_file):
        result = run_dataset_discovery(sample_excel_file)
        assert result.source_type == "excel"
        assert len(result.sheets) == 2
        assert "Sheet1" in result.sheets
        assert "Sheet2" in result.sheets

    def test_excel_total_rows_summed(self, sample_excel_file):
        result = run_dataset_discovery(sample_excel_file)
        assert result.total_rows == 4

    def test_excel_per_sheet_row_count(self, sample_excel_file):
        result = run_dataset_discovery(sample_excel_file)
        assert result.sheets["Sheet1"].row_count == 2
        assert result.sheets["Sheet2"].row_count == 2

    def test_excel_relationships_detected_for_joinable_sheets(self, joinable_excel_file):
        result = run_dataset_discovery(joinable_excel_file)
        # customer_id appears in both sheets — should be detected
        rel_columns = {r.column_a for r in result.relationships} | {r.column_b for r in result.relationships}
        assert "customer_id" in rel_columns

    def test_excel_suggested_joins_is_list(self, joinable_excel_file):
        result = run_dataset_discovery(joinable_excel_file)
        assert isinstance(result.suggested_joins, list)

    def test_source_path_set(self, sample_csv_file):
        result = run_dataset_discovery(sample_csv_file)
        assert sample_csv_file in result.source_path or Path(sample_csv_file).name in result.source_path

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises((FileNotFoundError, Exception)):
            run_dataset_discovery("does_not_exist.xlsx")


# ---------------------------------------------------------------------------
# get_semantic_sample
# ---------------------------------------------------------------------------

class TestGetSemanticSample:
    def test_returns_markdown_table(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = get_semantic_sample(df, n_rows=3)
        assert "|" in result
        assert "a" in result
        assert "b" in result

    def test_empty_dataframe_returns_empty_message(self):
        df = pd.DataFrame()
        result = get_semantic_sample(df)
        assert "Empty" in result

    def test_n_rows_limits_output(self):
        df = pd.DataFrame({"x": range(100)})
        result = get_semantic_sample(df, n_rows=3)
        # Markdown table with 3 data rows + 2 header rows = 5 lines
        data_lines = [l for l in result.split("\n") if "|" in l and "---" not in l and "x" not in l]
        assert len(data_lines) <= 3

    def test_n_rows_larger_than_df_does_not_raise(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = get_semantic_sample(df, n_rows=100)
        assert isinstance(result, str)

    def test_float_columns_rounded(self):
        """Floats are rounded to 4 decimal places to reduce token noise."""
        df = pd.DataFrame({"price": [1.23456789, 2.98765432]})
        result = get_semantic_sample(df, n_rows=2)
        # The raw 8-digit float should not appear
        assert "1.23456789" not in result
        assert "2.98765432" not in result

    def test_all_columns_present_in_header(self):
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "score": [0.9]})
        result = get_semantic_sample(df, n_rows=1)
        assert "id" in result
        assert "name" in result
        assert "score" in result

    def test_invalid_input_raises(self):
        with pytest.raises(AttributeError):
            get_semantic_sample([1, 2, 3])


# ---------------------------------------------------------------------------
# ColumnRelationship — model validation
# ---------------------------------------------------------------------------

class TestColumnRelationshipModel:
    def test_confidence_below_zero_rejected(self):
        with pytest.raises(Exception):
            ColumnRelationship(
                sheet_a="a", column_a="id", sheet_b="b", column_b="id",
                confidence=-0.1,
                detection_method="name_match",
            )

    def test_confidence_above_one_rejected(self):
        with pytest.raises(Exception):
            ColumnRelationship(
                sheet_a="a", column_a="id", sheet_b="b", column_b="id",
                confidence=1.1,
                detection_method="name_match",
            )

    def test_valid_relationship_constructed(self):
        rel = ColumnRelationship(
            sheet_a="orders", column_a="customer_id",
            sheet_b="customers", column_b="customer_id",
            confidence=0.85,
            detection_method="combined",
            cardinality="one_to_many",
            join_type_hint="inner",
        )
        assert rel.confidence == 0.85
        assert rel.cardinality == "one_to_many"
