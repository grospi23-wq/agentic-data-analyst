"""
test_memory_schema.py
---------------------
Tests for memory_schema.py — generate_schema_hash determinism and edge cases,
plus AnalystMemoryStore.to_cdo_context_block rendering.
"""

from datetime import datetime, timezone

import pytest
from memory_schema import (
    AnalystMemoryStore,
    JoinMemoryRecord,
    QualityWarning,
    generate_schema_hash,
)


# ---------------------------------------------------------------------------
# generate_schema_hash — determinism
# ---------------------------------------------------------------------------

class TestGenerateSchemaHashDeterminism:
    BASE_MAP = {
        "sheets": {
            "orders": {"columns": ["id", "amount", "customer_id"]},
            "customers": {"columns": ["id", "name", "region"]},
        }
    }

    def test_same_input_produces_same_hash(self):
        h1 = generate_schema_hash(self.BASE_MAP)
        h2 = generate_schema_hash(self.BASE_MAP)
        assert h1 == h2

    def test_hash_is_16_chars(self):
        h = generate_schema_hash(self.BASE_MAP)
        assert len(h) == 16

    def test_hash_is_hex_string(self):
        h = generate_schema_hash(self.BASE_MAP)
        int(h, 16)  # should not raise

    def test_different_column_names_produce_different_hash(self):
        map_a = {"sheets": {"orders": {"columns": ["id", "amount"]}}}
        map_b = {"sheets": {"orders": {"columns": ["id", "revenue"]}}}
        assert generate_schema_hash(map_a) != generate_schema_hash(map_b)

    def test_different_sheet_names_produce_different_hash(self):
        map_a = {"sheets": {"orders": {"columns": ["id"]}}}
        map_b = {"sheets": {"transactions": {"columns": ["id"]}}}
        assert generate_schema_hash(map_a) != generate_schema_hash(map_b)

    def test_column_order_does_not_affect_hash(self):
        """Columns are sorted internally, so order doesn't matter."""
        map_a = {"sheets": {"orders": {"columns": ["amount", "id"]}}}
        map_b = {"sheets": {"orders": {"columns": ["id", "amount"]}}}
        assert generate_schema_hash(map_a) == generate_schema_hash(map_b)

    def test_sheet_order_does_not_affect_hash(self):
        """Sheets are sorted by key, so order doesn't matter."""
        map_a = {
            "sheets": {
                "orders": {"columns": ["id"]},
                "customers": {"columns": ["id"]},
            }
        }
        map_b = {
            "sheets": {
                "customers": {"columns": ["id"]},
                "orders": {"columns": ["id"]},
            }
        }
        assert generate_schema_hash(map_a) == generate_schema_hash(map_b)


# ---------------------------------------------------------------------------
# generate_schema_hash — column format flexibility
# ---------------------------------------------------------------------------

class TestGenerateSchemaHashColumnFormats:
    def test_columns_as_plain_strings(self):
        discovery = {"sheets": {"sales": {"columns": ["id", "amount", "date"]}}}
        h = generate_schema_hash(discovery)
        assert len(h) == 16

    def test_columns_as_list_of_dicts_with_name(self):
        """Columns stored as metadata dicts — common in discovery output."""
        discovery = {
            "sheets": {
                "sales": {
                    "columns": [
                        {"name": "id", "type": "int"},
                        {"name": "amount", "type": "float"},
                    ]
                }
            }
        }
        h = generate_schema_hash(discovery)
        assert len(h) == 16

    def test_columns_as_dict_keys(self):
        """Columns stored as {col_name: metadata} dict."""
        discovery = {
            "sheets": {
                "sales": {
                    "columns": {
                        "id": {"type": "int"},
                        "amount": {"type": "float"},
                    }
                }
            }
        }
        h = generate_schema_hash(discovery)
        assert len(h) == 16

    def test_string_and_dict_formats_produce_same_hash(self):
        """Both formats that describe the same columns must produce the same hash."""
        map_strings = {"sheets": {"orders": {"columns": ["amount", "id"]}}}
        map_dicts = {
            "sheets": {
                "orders": {
                    "columns": [
                        {"name": "id", "type": "int64"},
                        {"name": "amount", "type": "float64"},
                    ]
                }
            }
        }
        assert generate_schema_hash(map_strings) == generate_schema_hash(map_dicts)

    def test_empty_sheets_dict(self):
        h = generate_schema_hash({"sheets": {}})
        assert len(h) == 16

    def test_missing_sheets_key(self):
        """Falls back gracefully to empty sheets."""
        h = generate_schema_hash({})
        assert len(h) == 16

    def test_sheet_with_empty_columns(self):
        h = generate_schema_hash({"sheets": {"empty": {"columns": []}}})
        assert len(h) == 16


# ---------------------------------------------------------------------------
# AnalystMemoryStore.to_cdo_context_block
# ---------------------------------------------------------------------------

class TestToCdoContextBlock:
    def test_no_runs_returns_no_prior_runs_message(self):
        store = AnalystMemoryStore(schema_hash="abc123", total_runs=0)
        result = store.to_cdo_context_block()
        assert "No prior runs" in result

    def test_with_runs_includes_schema_hash(self):
        store = AnalystMemoryStore(schema_hash="abc123", total_runs=5)
        result = store.to_cdo_context_block()
        assert "abc123" in result

    def test_with_runs_includes_run_count(self):
        store = AnalystMemoryStore(schema_hash="abc", total_runs=7)
        result = store.to_cdo_context_block()
        assert "7" in result

    def test_with_successful_joins_listed(self):
        join = JoinMemoryRecord(
            schema_hash="abc",
            left_sheet="orders",
            right_sheet="customers",
            on_column="customer_id",
            join_type="inner",
            result_name="orders_customers",
            critic_score=0.82,
        )
        store = AnalystMemoryStore(
            schema_hash="abc",
            total_runs=2,
            successful_joins=[join],
        )
        result = store.to_cdo_context_block()
        assert "orders" in result
        assert "customers" in result
        assert "customer_id" in result
        assert "0.82" in result

    def test_with_quality_warnings_listed(self):
        warning = QualityWarning(
            schema_hash="abc",
            finding_type="missing_data_inflation",
            affected_column_or_insight="loyalty_points",
            reason="Column is 90% null — expected for sparse events",
            severity="warn",
            times_seen=3,
        )
        store = AnalystMemoryStore(
            schema_hash="abc",
            total_runs=3,
            quality_warnings=[warning],
        )
        result = store.to_cdo_context_block()
        assert "loyalty_points" in result
        assert "3" in result  # times_seen

    def test_join_reuse_hint_present(self):
        join = JoinMemoryRecord(
            schema_hash="abc",
            left_sheet="A",
            right_sheet="B",
            on_column="id",
            join_type="left",
            result_name="ab",
        )
        store = AnalystMemoryStore(
            schema_hash="abc",
            total_runs=1,
            successful_joins=[join],
        )
        result = store.to_cdo_context_block()
        assert "Reuse" in result

    def test_runs_with_no_joins_or_warnings_still_works(self):
        store = AnalystMemoryStore(schema_hash="abc", total_runs=10)
        result = store.to_cdo_context_block()
        assert "abc" in result
        assert isinstance(result, str)
