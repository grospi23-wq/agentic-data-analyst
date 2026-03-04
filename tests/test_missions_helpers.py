"""
test_missions_helpers.py
------------------------
Tests for the pure helper functions in missions.py:
  - _format_rewrites     — renders CriticReport revision payload as text
  - _print_rejection_detail — prints structured rejection summary to stdout
  - _load_specialist_prompt — loads domain-specific or GENERAL fallback prompt
"""

import pytest
from unittest.mock import MagicMock, patch
from schema import CriticReport, CriticalFinding, RevisionRewrite
from missions import _format_rewrites, _print_rejection_detail, _load_specialist_prompt
from schema import SpecialistType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_critic_report(
    rewrites=None,
    revision_instructions=None,
    approved=False,
    score=0.4,
) -> CriticReport:
    return CriticReport(
        approved=approved,
        score=score,
        findings=[],
        required_rewrites=rewrites or [],
        revision_instructions=revision_instructions,
    )


def _make_rewrite(insight_id="insight_1", issue_type="math_error", fix="Recalculate.") -> RevisionRewrite:
    return RevisionRewrite(
        insight_id=insight_id,
        issue_type=issue_type,
        fix_instruction=fix,
    )


# ---------------------------------------------------------------------------
# _format_rewrites
# ---------------------------------------------------------------------------

class TestFormatRewrites:
    def test_required_rewrites_formatted_as_bullets(self):
        rw1 = _make_rewrite("insight_1", "math_error", "Fix the calculation.")
        rw2 = _make_rewrite("insight_2", "hallucination", "Remove unsupported claim.")
        report = _make_critic_report(rewrites=[rw1, rw2])

        result = _format_rewrites(report)

        assert "[insight_1]" in result
        assert "(math_error)" in result
        assert "Fix the calculation." in result
        assert "[insight_2]" in result

    def test_each_rewrite_on_separate_line(self):
        rw1 = _make_rewrite("i1", "math_error", "Fix A.")
        rw2 = _make_rewrite("i2", "hallucination", "Fix B.")
        report = _make_critic_report(rewrites=[rw1, rw2])

        result = _format_rewrites(report)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_fallback_to_revision_instructions_when_no_rewrites(self):
        report = _make_critic_report(
            rewrites=[],
            revision_instructions="Please revise insight 3.",
        )
        result = _format_rewrites(report)
        assert "Please revise insight 3." in result

    def test_fallback_default_message_when_both_empty(self):
        report = _make_critic_report(rewrites=[], revision_instructions=None)
        result = _format_rewrites(report)
        assert "Revise" in result or len(result) > 0

    def test_required_rewrites_take_precedence_over_revision_instructions(self):
        rw = _make_rewrite("i1", "math_error", "Specific fix.")
        report = _make_critic_report(
            rewrites=[rw],
            revision_instructions="Generic fallback instruction.",
        )
        result = _format_rewrites(report)
        assert "Specific fix." in result
        assert "Generic fallback instruction." not in result

    def test_single_rewrite(self):
        rw = _make_rewrite("insight_5", "direction_error", "Reverse the trend statement.")
        report = _make_critic_report(rewrites=[rw])
        result = _format_rewrites(report)
        assert "insight_5" in result
        assert "direction_error" in result


# ---------------------------------------------------------------------------
# _print_rejection_detail
# ---------------------------------------------------------------------------

class TestPrintRejectionDetail:
    def _make_report_with_rewrites(self, n_rewrites: int, structural_failures=None):
        rewrites = [
            _make_rewrite(f"insight_{i}", "math_error", f"Fix {i}.")
            for i in range(1, n_rewrites + 1)
        ]
        report = CriticReport(
            approved=False,
            score=0.35,
            overall_value_score=0.4,
            findings=[],
            required_rewrites=rewrites,
            structural_failures=structural_failures or [],
        )
        return report

    def test_prints_score_and_value(self, capsys):
        report = self._make_report_with_rewrites(2)
        report.score = 0.35
        report.overall_value_score = 0.42
        _print_rejection_detail(report, attempt=1, max_attempts=3)

        captured = capsys.readouterr()
        assert "0.35" in captured.out
        assert "0.42" in captured.out

    def test_prints_attempt_info(self, capsys):
        report = self._make_report_with_rewrites(1)
        _print_rejection_detail(report, attempt=2, max_attempts=4)

        captured = capsys.readouterr()
        assert "2/4" in captured.out

    def test_prints_rewrite_count(self, capsys):
        report = self._make_report_with_rewrites(3)
        _print_rejection_detail(report, attempt=1, max_attempts=3)

        captured = capsys.readouterr()
        assert "3" in captured.out

    def test_prints_structural_failures(self, capsys):
        report = self._make_report_with_rewrites(
            0,
            structural_failures=["Math error in insight_1", "ID column used as metric"],
        )
        _print_rejection_detail(report, attempt=1, max_attempts=3)

        captured = capsys.readouterr()
        assert "Math error" in captured.out
        assert "ID column" in captured.out

    def test_caps_rewrites_at_three(self, capsys):
        report = self._make_report_with_rewrites(6)  # 6 rewrites
        _print_rejection_detail(report, attempt=1, max_attempts=3)

        captured = capsys.readouterr()
        # Should show "… and 3 more"
        assert "3 more" in captured.out

    def test_no_truncation_message_for_three_or_fewer(self, capsys):
        report = self._make_report_with_rewrites(3)
        _print_rejection_detail(report, attempt=1, max_attempts=3)

        captured = capsys.readouterr()
        assert "more" not in captured.out

    def test_no_structural_failures_no_section_printed(self, capsys):
        report = self._make_report_with_rewrites(1, structural_failures=[])
        _print_rejection_detail(report, attempt=1, max_attempts=3)

        captured = capsys.readouterr()
        assert "Structural failures" not in captured.out


# ---------------------------------------------------------------------------
# _load_specialist_prompt
# ---------------------------------------------------------------------------

class TestLoadSpecialistPrompt:
    def test_retail_specialist_loaded(self):
        result = _load_specialist_prompt(SpecialistType.RETAIL)
        assert isinstance(result, str)
        assert len(result) > 50  # Should be a real prompt, not empty

    def test_finance_specialist_loaded(self):
        result = _load_specialist_prompt(SpecialistType.FINANCE)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_sports_specialist_loaded(self):
        result = _load_specialist_prompt(SpecialistType.SPORTS)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_healthcare_specialist_loaded(self):
        result = _load_specialist_prompt(SpecialistType.HEALTHCARE)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_logistics_specialist_loaded(self):
        result = _load_specialist_prompt(SpecialistType.LOGISTICS)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_general_specialist_loaded(self):
        result = _load_specialist_prompt(SpecialistType.GENERAL)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_missing_specialist_falls_back_to_general(self, monkeypatch, tmp_path):
        """A specialist type without a prompt file falls back to GENERAL.txt."""
        # Monkeypatch the Path class to intercept the specialist path check
        mock_specialist = MagicMock(spec=SpecialistType)
        mock_specialist.value = "NONEXISTENT_DOMAIN_XYZ"

        with patch("missions.logfire") as mock_logfire:
            result = _load_specialist_prompt(mock_specialist)

        # Should have fallen back — logfire.warn called
        mock_logfire.warn.assert_called_once()
        assert isinstance(result, str)
        assert len(result) > 50  # GENERAL.txt content

    def test_all_specialist_types_are_loadable(self):
        """Smoke test: every SpecialistType enum value resolves without error."""
        for st in SpecialistType:
            result = _load_specialist_prompt(st)
            assert isinstance(result, str)
            assert len(result) > 0, f"Empty prompt for specialist type {st.value}"

    def test_returned_prompt_is_stripped(self):
        """load_specialist_prompt returns stripped text (no leading/trailing whitespace)."""
        result = _load_specialist_prompt(SpecialistType.GENERAL)
        assert result == result.strip()
