from enum import Enum

from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional


class SpecialistType(str, Enum):
    """Domain classification used to select the analyst specialist prompt."""
    RETAIL = "RETAIL"
    FINANCE = "FINANCE"
    SPORTS = "SPORTS"
    HEALTHCARE = "HEALTHCARE"
    LOGISTICS = "LOGISTICS"
    GENERAL = "GENERAL"


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
    specialist_type: SpecialistType = SpecialistType.GENERAL


class InsightScore(BaseModel):
    """Per-insight quality score produced by the Critic's Track 2 evaluation."""
    insight_id: str = Field(..., description="Short slug identifier (e.g. 'insight_1').")
    insight_summary: str = Field(..., description="One-sentence summary of the insight being scored.")
    statistical_strength: float = Field(..., ge=0.0, le=1.0, description="How statistically robust (effect size, sample size)?")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="How surprising relative to domain baseline expectations?")
    decision_leverage: float = Field(..., ge=0.0, le=1.0, description="How directly actionable for business decisions?")
    value_score: float = Field(..., ge=0.0, le=1.0, description="0.4*statistical_strength + 0.3*novelty_score + 0.3*decision_leverage.")
    value_tier: Literal["low_value", "moderate_value", "high_value"]


class RevisionRewrite(BaseModel):
    """Structured fix instruction emitted by the Critic for a single failing insight."""
    insight_id: str = Field(..., description="Which insight needs to be revised.")
    issue_type: str = Field(..., description="Category: 'math_error', 'id_misuse', 'hallucination', 'direction_error'.")
    fix_instruction: str = Field(..., description="Specific, actionable instruction for the Analyst.")


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
    # Track 2: insight value scoring
    insight_scores: List[InsightScore] = Field(default_factory=list)
    overall_value_score: float = Field(default=0.0, description="Mean value_score across all scored insights.")
    low_value_insights: List[str] = Field(default_factory=list, description="insight_ids with value_tier == 'low_value'.")
    structural_failures: List[str] = Field(default_factory=list, description="Descriptions of hard structural errors (math, ID misuse, hallucination).")
    required_rewrites: List[RevisionRewrite] = Field(default_factory=list, description="Structured revision instructions for retry attempts.")
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
    specialist_type: SpecialistType = SpecialistType.GENERAL


class ChartDataPoint(BaseModel):
    label: str = Field(..., description="The category or x-axis label (e.g., 'Bikes', '2023').")
    value: float = Field(..., description="The numerical value for this point.")

class ChartSpec(BaseModel):
    """Specification for a deterministic, non-hallucinated chart."""
    chart_type: Literal["bar", "line", "pie"]
    title: str = Field(..., description="A short, descriptive title for the chart.")
    x_label: Optional[str] = Field(default=None, description="Label for the X axis (if applicable).")
    y_label: Optional[str] = Field(default=None, description="Label for the Y axis (if applicable).")
    data_points: List[ChartDataPoint] = Field(
        ..., 
        max_length=7, 
        description="Limit to max 7 data points for visual clarity on the slide."
    )

class SlideContent(BaseModel):
    slide_type: Literal["title", "hypothesis", "finding", "limitation", "action_plan", "conclusion"]
    title: str
    body_bullets: List[str] = Field(default_factory=list)
    metric_callout: Optional[str] = None
    speaker_notes: str = ""
    chart: Optional[ChartSpec] = Field(
        default=None, 
        description="Optional chart to visualize the finding. Use only when numeric trends or comparisons are central to the insight."
    )    


class PresentationSpec(BaseModel):
    deck_title: str
    subtitle: str
    hypothesis_verdict: Literal["CONFIRMED", "PARTIALLY CONFIRMED", "REJECTED"]
    slides: List[SlideContent]
    critic_score: float
    analysis_mode: str


