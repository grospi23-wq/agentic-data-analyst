"""
memory_schema.py
----------------
Pydantic models for the persistent memory store, keyed by schema_hash.
Also exposes generate_schema_hash() — the single source of truth for
turning a discovery_map into a deterministic string key.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hash Generation
# ---------------------------------------------------------------------------

def generate_schema_hash(discovery_map_dict: Dict[str, Any]) -> str:
    """
    Produce a stable SHA-256 fingerprint from the structural signature of
    a dataset. Safely extracts column names whether they are strings or dicts.
    """
    sheets: Dict[str, Any] = discovery_map_dict.get("sheets", {})
    canonical: Dict[str, List[str]] = {}

    for sheet_name, sheet_data in sorted(sheets.items()):
        raw_columns = sheet_data.get("columns", [])
        col_names = []
        
        # Extract column names safely depending on the data structure
        if isinstance(raw_columns, list):
            for col in raw_columns:
                if isinstance(col, dict):
                    # If columns are stored as metadata dicts
                    col_names.append(col.get("name", str(col)))
                else:
                    # If columns are stored as plain strings
                    col_names.append(str(col))
        elif isinstance(raw_columns, dict):
            # If columns are stored as a dict mapping {col_name: metadata}
            col_names = list(raw_columns.keys())

        # Store the sorted list of column names for this sheet
        canonical[sheet_name] = sorted(col_names)

    # Serialize with sorted keys for extra determinism, then hash
    canonical_json = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return digest[:16]


# ---------------------------------------------------------------------------
# Memory Record Models
# ---------------------------------------------------------------------------

class JoinMemoryRecord(BaseModel):
    """
    A previously validated join configuration that produced a meaningful
    result for a given schema family.
    """
    schema_hash: str = Field(..., description="Schema fingerprint this record belongs to.")
    left_sheet: str
    right_sheet: str
    on_column: str
    join_type: str  # inner | left | outer
    result_name: str

    # Quality signals collected at the time of the successful run
    row_count_after_join: int = Field(
        default=0,
        description="Row count of the merged DataFrame — used to detect degenerate joins.",
    )
    critic_score: float = Field(
        default=0.0,
        description="Critic score from the run that approved this join.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this record was first persisted.",
    )
    last_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the most recent successful run with this join.",
    )


class QualityWarning(BaseModel):
    """
    A Critic finding that was either approved despite a warning, or blocked
    and later overridden by a human operator (severity == 'warn' only).

    Persisting these allows the Critic to calibrate expectations:
    repeated warnings on the same schema family are acknowledged rather
    than re-escalated on every quarterly refresh.
    """
    schema_hash: str
    finding_type: str  # mirrors CriticalFinding.finding_type
    affected_column_or_insight: str
    reason: str
    severity: str  # 'warn' | 'block'
    times_seen: int = Field(default=1, description="How many runs have raised this finding.")

    first_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AnalystMemoryStore(BaseModel):
    """
    The complete in-memory snapshot for a single schema_hash family,
    loaded from SQLite at the start of a mission and written back at the end.
    """
    schema_hash: str
    successful_joins: List[JoinMemoryRecord] = Field(default_factory=list)
    quality_warnings: List[QualityWarning] = Field(default_factory=list)

    # Aggregate counters for the CDO prompt context
    total_runs: int = Field(
        default=0,
        description="Total number of missions executed against this schema family.",
    )
    last_run_at: Optional[datetime] = None

    def to_cdo_context_block(self) -> str:
        """
        Render a compact, prompt-friendly summary to inject into the
        Multi-Sheet CDO system prompt.
        """
        if self.total_runs == 0:
            return "MEMORY: No prior runs recorded for this schema family."

        lines = [
            f"MEMORY CONTEXT (schema_hash={self.schema_hash}, runs={self.total_runs})",
            "",
        ]

        if self.successful_joins:
            lines.append("## Previously Validated Joins")
            for j in self.successful_joins:
                lines.append(
                    f"  - {j.left_sheet} ⋈ {j.right_sheet} ON {j.on_column} "
                    f"[{j.join_type}] → {j.result_name} "
                    f"(critic_score={j.critic_score:.2f}, last_seen={j.last_seen_at.date()})"
                )
            lines.append(
                "  ↳ Reuse these join configs unless the current data contradicts them."
            )
            lines.append("")

        if self.quality_warnings:
            lines.append("## Known Data Quality Patterns (do NOT re-escalate as new findings)")
            for w in self.quality_warnings:
                lines.append(
                    f"  - [{w.finding_type}] on '{w.affected_column_or_insight}': "
                    f"{w.reason} (seen {w.times_seen}x, severity={w.severity})"
                )
            lines.append("")

        return "\n".join(lines)
