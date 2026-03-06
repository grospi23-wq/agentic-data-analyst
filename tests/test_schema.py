"""
test_schema.py
--------------
Unit tests for schema.py — focused on CriticReport.enforce_score_consistency,
which encodes the business rule that scores must match approval/block state.
"""

import pytest
from schema import CriticReport, CriticalFinding, InsightScore, RevisionRewrite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_finding() -> CriticalFinding:
    return CriticalFinding(
        finding_type="unsupported_claim",
        affected_insight="Revenue claim",
        reason="No supporting data",
        severity="block",
    )


def _warn_finding() -> CriticalFinding:
    return CriticalFinding(
        finding_type="redundancy",
        affected_insight="Sales trend",
        reason="Duplicate insight",
        severity="warn",
    )

from typing import Any
def _make_report(**kwargs: Any) -> CriticReport:
    # We ensure 'approved' is treated as a bool to satisfy the schema
    approved = bool(kwargs.get("approved", False))
    score = float(kwargs.get("score", 0.0))
    findings = kwargs.get("findings", [])
    
    return CriticReport(
        approved=approved,
        score=score,
        findings=findings,
        # pass the rest if they exist
        **{k: v for k, v in kwargs.items() if k not in ["approved", "score", "findings"]}
    )


# ---------------------------------------------------------------------------
# Block finding → score clamped to 0.0
# ---------------------------------------------------------------------------

class TestBlockFindingClamping:
    def test_block_finding_high_score_clamped_to_zero(self):
        report = _make_report(
            approved=False,
            score=0.9,
            findings=[_block_finding()],
        )
        assert report.score == 0.0

    def test_block_finding_moderate_score_clamped_to_zero(self):
        report = _make_report(
            approved=False,
            score=0.4,
            findings=[_block_finding()],
        )
        assert report.score == 0.0

    def test_block_finding_below_threshold_score_not_clamped(self):
        """Score <= 0.3 with block finding is NOT clamped (intentional design)."""
        report = _make_report(
            approved=False,
            score=0.3,
            findings=[_block_finding()],
        )
        assert report.score == 0.3

    def test_block_finding_zero_score_unchanged(self):
        report = _make_report(
            approved=False,
            score=0.0,
            findings=[_block_finding()],
        )
        assert report.score == 0.0


# ---------------------------------------------------------------------------
# Not approved + score > 0.5 → clamped to 0.4
# ---------------------------------------------------------------------------

class TestNotApprovedClamping:
    def test_not_approved_high_score_clamped(self):
        report = _make_report(
            approved=False,
            score=0.8,
            findings=[_warn_finding()],
        )
        assert report.score == 0.4

    def test_not_approved_exactly_at_limit(self):
        report = _make_report(
            approved=False,
            score=0.51,
            findings=[],
        )
        assert report.score == 0.4

    def test_not_approved_at_threshold_not_clamped(self):
        """Score == 0.5 is NOT > 0.5, so no clamping."""
        report = _make_report(
            approved=False,
            score=0.5,
            findings=[],
        )
        assert report.score == 0.5

    def test_not_approved_low_score_unchanged(self):
        report = _make_report(
            approved=False,
            score=0.35,
            findings=[],
        )
        assert report.score == 0.35


# ---------------------------------------------------------------------------
# Approved + findings + score > 0.85 → clamped to 0.85
# ---------------------------------------------------------------------------

class TestApprovedWithFindingsClamping:
    def test_approved_with_findings_very_high_score_clamped(self):
        report = _make_report(
            approved=True,
            score=0.95,
            findings=[_warn_finding()],
        )
        assert report.score == 0.85

    def test_approved_with_findings_at_threshold_not_clamped(self):
        report = _make_report(
            approved=True,
            score=0.85,
            findings=[_warn_finding()],
        )
        assert report.score == 0.85

    def test_approved_no_findings_high_score_unchanged(self):
        """Approved with NO findings: high score is allowed."""
        report = _make_report(
            approved=True,
            score=0.95,
            findings=[],
        )
        assert report.score == 0.95

    def test_approved_with_findings_low_score_unchanged(self):
        report = _make_report(
            approved=True,
            score=0.80,
            findings=[_warn_finding()],
        )
        assert report.score == 0.80


# ---------------------------------------------------------------------------
# Happy paths — no clamping
# ---------------------------------------------------------------------------

class TestHappyPaths:
    def test_approved_no_findings_any_score(self):
        for score in [0.0, 0.5, 0.75, 1.0]:
            report = _make_report(approved=True, score=score, findings=[])
            assert report.score == score

    def test_not_approved_score_at_or_below_05_no_clamping(self):
        report = _make_report(approved=False, score=0.3, findings=[])
        assert report.score == 0.3

    def test_multiple_warn_findings_no_clamping(self):
        report = _make_report(
            approved=True,
            score=0.75,
            findings=[_warn_finding(), _warn_finding()],
        )
        assert report.score == 0.75


# ---------------------------------------------------------------------------
# InsightScore field bounds (Pydantic validation)
# ---------------------------------------------------------------------------

class TestInsightScoreBounds:
    def test_valid_insight_score(self):
        score = InsightScore(
            insight_id="insight_1",
            insight_summary="Sales declined in Q3",
            statistical_strength=0.8,
            novelty_score=0.6,
            decision_leverage=0.7,
            value_score=0.72,
            value_tier="high_value",
        )
        assert score.value_score == 0.72

    def test_statistical_strength_out_of_bounds(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            InsightScore(
                insight_id="i1",
                insight_summary="test",
                statistical_strength=1.5,  # > 1.0
                novelty_score=0.5,
                decision_leverage=0.5,
                value_score=0.5,
                value_tier="moderate_value",
            )

    def test_value_score_negative(self):
        with pytest.raises(Exception):
            InsightScore(
                insight_id="i1",
                insight_summary="test",
                statistical_strength=0.5,
                novelty_score=0.5,
                decision_leverage=0.5,
                value_score=-0.1,  # < 0.0
                value_tier="low_value",
            )


# ---------------------------------------------------------------------------
# RevisionRewrite structure
# ---------------------------------------------------------------------------

class TestRevisionRewrite:
    def test_basic_construction(self):
        rw = RevisionRewrite(
            insight_id="insight_2",
            issue_type="math_error",
            fix_instruction="Recalculate using correct denominator.",
        )
        assert rw.insight_id == "insight_2"
        assert rw.issue_type == "math_error"

    def test_used_in_critic_report(self):
        rw = RevisionRewrite(
            insight_id="i1",
            issue_type="hallucination",
            fix_instruction="Remove unsupported claim.",
        )
        report = _make_report(
            approved=False,
            score=0.3,
            findings=[_warn_finding()],
            required_rewrites=[rw],
        )
        assert len(report.required_rewrites) == 1
        assert report.required_rewrites[0].insight_id == "i1"
