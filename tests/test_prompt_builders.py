"""
test_prompt_builders.py
-----------------------
Tests for prompt_builders.py — pure factory functions with zero external deps.
All tests are synchronous; no mocking required.
"""

import json
import pytest
from prompt_builders import (
    build_analyst_mission_prompt,
    build_retry_prompt,
    build_self_validation_prompt,
    build_cdo_multisheet_prompt,
    build_formatter_prompt,
)


# ---------------------------------------------------------------------------
# build_analyst_mission_prompt
# ---------------------------------------------------------------------------

class TestBuildAnalystMissionPrompt:
    TEMPLATE = (
        "Hypothesis: {hypothesis}\n"
        "Questions:\n{questions}\n"
        "Data:\n{data}\n"
        "Mode: {analysis_mode}\n"
        "Cross-sheet: {cross_sheet_hypothesis}"
    )

    def test_schema_header_prepended(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="Sales are declining",
            questions=["Q1", "Q2"],
            data_json='{"key": "value"}',
            schema_context="orders: [id, amount]",
        )
        assert result.startswith("CRITICAL — VERIFIED SCHEMA")

    def test_schema_context_in_output(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=["Q"],
            data_json="{}",
            schema_context="Table 'sales' — columns: [id, revenue]",
        )
        assert "sales" in result
        assert "revenue" in result

    def test_hypothesis_injected(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="Revenue is correlated with region",
            questions=[],
            data_json="{}",
            schema_context="ctx",
        )
        assert "Revenue is correlated with region" in result

    def test_questions_formatted_as_bullets(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=["First question", "Second question"],
            data_json="{}",
            schema_context="ctx",
        )
        assert "- First question" in result
        assert "- Second question" in result

    def test_data_json_injected(self):
        data = {"orders": {"row_count": 100}}
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=[],
            data_json=json.dumps(data),
            schema_context="ctx",
        )
        assert "row_count" in result

    def test_analysis_mode_default_separate(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=[],
            data_json="{}",
            schema_context="ctx",
        )
        assert "separate" in result

    def test_analysis_mode_custom(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=[],
            data_json="{}",
            schema_context="ctx",
            analysis_mode="joined",
        )
        assert "joined" in result

    def test_cross_sheet_hypothesis_injected(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=[],
            data_json="{}",
            schema_context="ctx",
            cross_sheet_hypothesis="High orders correlate with high revenue",
        )
        assert "High orders correlate with high revenue" in result

    def test_empty_questions_list(self):
        result = build_analyst_mission_prompt(
            template=self.TEMPLATE,
            hypothesis="H",
            questions=[],
            data_json="{}",
            schema_context="ctx",
        )
        # Should not raise; questions block is just empty
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_retry_prompt
# ---------------------------------------------------------------------------

class TestBuildRetryPrompt:
    TEMPLATE = "Fix this:\n{revision_instructions}\n\nPrevious:\n{previous_narrative}"

    def test_no_schema_header(self):
        """Retry prompt intentionally omits the schema header (lean resend)."""
        result = build_retry_prompt(
            template=self.TEMPLATE,
            revision_instructions="Fix the math error in insight_1.",
            previous_narrative="Old report content.",
        )
        assert "CRITICAL — VERIFIED SCHEMA" not in result

    def test_revision_instructions_injected(self):
        result = build_retry_prompt(
            template=self.TEMPLATE,
            revision_instructions="Recalculate the percentage.",
            previous_narrative="Draft narrative.",
        )
        assert "Recalculate the percentage." in result

    def test_previous_narrative_injected(self):
        result = build_retry_prompt(
            template=self.TEMPLATE,
            revision_instructions="Fix it.",
            previous_narrative="The previous analysis found that...",
        )
        assert "The previous analysis found that..." in result

    def test_multiline_instructions_preserved(self):
        instructions = "- Fix insight_1\n- Fix insight_2\n- Fix insight_3"
        result = build_retry_prompt(
            template=self.TEMPLATE,
            revision_instructions=instructions,
            previous_narrative="draft",
        )
        assert "insight_1" in result
        assert "insight_3" in result


# ---------------------------------------------------------------------------
# build_self_validation_prompt
# ---------------------------------------------------------------------------

class TestBuildSelfValidationPrompt:
    TEMPLATE = "CoT:\n{chain_of_thought}\n\nDraft:\n{draft_narrative}"

    def test_schema_header_prepended(self):
        result = build_self_validation_prompt(
            template=self.TEMPLATE,
            schema_context="Table 'orders'",
            chain_of_thought=["step 1", "step 2"],
            draft_narrative="Draft report here.",
        )
        assert result.startswith("CRITICAL — VERIFIED SCHEMA")

    def test_chain_of_thought_joined_with_newlines(self):
        result = build_self_validation_prompt(
            template=self.TEMPLATE,
            schema_context="ctx",
            chain_of_thought=["First reasoning step.", "Second reasoning step."],
            draft_narrative="draft",
        )
        assert "First reasoning step." in result
        assert "Second reasoning step." in result

    def test_draft_narrative_injected(self):
        result = build_self_validation_prompt(
            template=self.TEMPLATE,
            schema_context="ctx",
            chain_of_thought=[],
            draft_narrative="Final narrative text.",
        )
        assert "Final narrative text." in result

    def test_empty_chain_of_thought(self):
        result = build_self_validation_prompt(
            template=self.TEMPLATE,
            schema_context="ctx",
            chain_of_thought=[],
            draft_narrative="draft",
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_cdo_multisheet_prompt
# ---------------------------------------------------------------------------

class TestBuildCdoMultisheetPrompt:
    PRUNED_MAP = {
        "sheets": {
            "orders": {"columns": ["id", "amount"]},
            "customers": {"columns": ["id", "name"]},
        }
    }

    def test_dataset_structure_header_present(self):
        result = build_cdo_multisheet_prompt(
            pruned_map=self.PRUNED_MAP,
            memory_context="MEMORY: No prior runs.",
            lean_mode=False,
        )
        assert "DATASET STRUCTURE" in result

    def test_full_mode_label(self):
        result = build_cdo_multisheet_prompt(
            pruned_map=self.PRUNED_MAP,
            memory_context="",
            lean_mode=False,
        )
        assert "Full" in result

    def test_lean_mode_label(self):
        result = build_cdo_multisheet_prompt(
            pruned_map=self.PRUNED_MAP,
            memory_context="",
            lean_mode=True,
        )
        assert "Lean" in result

    def test_pruned_map_serialized_as_json(self):
        result = build_cdo_multisheet_prompt(
            pruned_map=self.PRUNED_MAP,
            memory_context="",
            lean_mode=False,
        )
        assert '"orders"' in result
        assert '"customers"' in result

    def test_memory_context_appended(self):
        memory = "MEMORY CONTEXT (schema_hash=abc123, runs=3)"
        result = build_cdo_multisheet_prompt(
            pruned_map=self.PRUNED_MAP,
            memory_context=memory,
            lean_mode=False,
        )
        assert memory in result

    def test_empty_pruned_map(self):
        result = build_cdo_multisheet_prompt(
            pruned_map={},
            memory_context="no memory",
            lean_mode=False,
        )
        assert isinstance(result, str)
        assert "DATASET STRUCTURE" in result


# ---------------------------------------------------------------------------
# build_formatter_prompt
# ---------------------------------------------------------------------------

class TestBuildFormatterPrompt:
    def test_report_json_included(self):
        result = build_formatter_prompt(
            report_json='{"hypothesis_validation": "CONFIRMED"}',
            deck_title="Q3 Sales Analysis",
            critic_score=0.85,
            analysis_mode="separate",
        )
        assert "CONFIRMED" in result

    def test_deck_title_included(self):
        result = build_formatter_prompt(
            report_json="{}",
            deck_title="My Custom Deck",
            critic_score=0.7,
            analysis_mode="joined",
        )
        assert "My Custom Deck" in result

    def test_critic_score_included(self):
        result = build_formatter_prompt(
            report_json="{}",
            deck_title="Analysis",
            critic_score=0.92,
            analysis_mode="separate",
        )
        assert "0.92" in result

    def test_analysis_mode_included(self):
        result = build_formatter_prompt(
            report_json="{}",
            deck_title="Analysis",
            critic_score=0.5,
            analysis_mode="both",
        )
        assert "both" in result

    def test_output_is_string(self):
        result = build_formatter_prompt(
            report_json="{}",
            deck_title="T",
            critic_score=0.0,
            analysis_mode="separate",
        )
        assert isinstance(result, str)
