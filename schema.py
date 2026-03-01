from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional


class AnalysisStrategy(BaseModel):
    run_outlier_detection: bool
    target_columns_for_outliers: List[str]
    run_correlation_matrix: bool
    business_hypothesis: str
    primary_questions: List[str]
    reasoning: str
    id_like_columns: List[str]
    excluded_from_profiling: List[str]
    correlation_columns: List[str]


class CriticalFinding(BaseModel):
    finding_type: Literal[
        "id_column_misuse",
        "correlation_causation_error",
        "unsupported_claim",
        "missing_data_inflation",
        "outlier_misinterpretation",
        "redundancy",
        "weak_signal_overstatement",
        "missing_limitations",
    ]
    affected_insight: str
    reason: str
    severity: Literal["block", "warn"]


class CriticReport(BaseModel):
    approved: bool
    score: float
    findings: List[CriticalFinding]
    revision_instructions: Optional[str] = None

    @model_validator(mode="after")
    def enforce_score_consistency(self) -> "CriticReport":
        has_block = any(f.severity == "block" for f in self.findings)
        if has_block and self.score > 0.3:
            self.score = 0.0
        if not self.approved and self.score > 0.5:
            self.score = min(self.score, 0.4)
        if self.approved and self.findings and self.score > 0.85:
            self.score = 0.85
        return self


class AnalystOutput(BaseModel):
    internal_thought_process: List[str] = Field(
        ..., description="Step-by-step reasoning and mathematical checks."
    )
    hypothesis_validation: str = Field(
        ..., description="Confirmation or rejection of the CDO's hypothesis."
    )
    final_report_markdown: str = Field(
        ..., description="The final business-centric report."
    )


class SheetAnalysisPlan(BaseModel):
    sheet_name: str
    should_profile: bool
    id_like_columns: List[str]
    excluded_from_profiling: List[str]
    correlation_columns: List[str]
    primary_focus: str


class JoinInstruction(BaseModel):
    left_sheet: str
    right_sheet: str
    on_column: str
    join_type: Literal["inner", "left", "outer"]
    result_name: str


class MultiSheetStrategy(BaseModel):
    business_hypothesis: str
    primary_questions: List[str]
    analysis_mode: Literal["separate", "joined", "both"]
    per_sheet_plans: List[SheetAnalysisPlan]
    joins_to_execute: List[JoinInstruction]
    cross_sheet_hypothesis: str
    reasoning: str


class SlideContent(BaseModel):
    slide_type: Literal["title", "hypothesis", "finding", "limitation", "conclusion"]
    title: str
    body_bullets: List[str] = Field(default_factory=list)
    metric_callout: Optional[str] = None
    speaker_notes: str = ""


class PresentationSpec(BaseModel):
    deck_title: str
    subtitle: str
    hypothesis_verdict: Literal["CONFIRMED", "PARTIALLY CONFIRMED", "REJECTED"]
    slides: List[SlideContent]
    critic_score: float
    analysis_mode: str
