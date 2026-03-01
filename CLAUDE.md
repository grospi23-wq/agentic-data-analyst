# CLAUDE.md — Agentic Data Analyst: Project Memory & Source of Truth

## Architecture

3-agent pipeline running through OpenRouter. Minimalist agent count is a hard constraint — no new agents without a strong reason.

| Agent | Role | Model | Output type |
|-------|------|-------|-------------|
| `orchestrator_agent` | Chief Data Officer (CDO) — defines strategy, excludes ID columns | `anthropic/claude-3.5-sonnet` | `AnalysisStrategy` |
| `analyst_agent` | Senior Analyst — CoT reasoning + report | `openai/gpt-4o` | `AnalystOutput` |
| `critic_agent` | Ruthless Auditor — validates narrative vs. raw data | `anthropic/claude-3.5-sonnet` | `CriticReport` |

**Phase 2 decision (locked):** Option A — single analyst with structured Pydantic output (`AnalystOutput`) for chain-of-thought. No separate reasoning agent.

---

## Terminology (Phase 3)

- **Discovery layer:** Metadata/schema mapping only. Entry point: `run_dataset_discovery(path)` → `GlobalDiscoveryMap`. No business logic.
- **Strategy:** CDO outputs — `AnalysisStrategy` (single-sheet) or `MultiSheetStrategy` (multi-sheet). Defines hypothesis, excluded columns, correlation targets.
- **Mission:** Top-level execution. Two entry points with a unified naming pattern:
  - `execute_analysis_mission(file_path, target_sheet)` — single-sheet flow (Discovery → Strategy → Profiling → Synthesis).
  - `execute_multi_sheet_mission(file_path)` — multi-sheet flow (Discovery → MultiSheetStrategy → Joins → Parallel Profiling → Synthesis).
- **Profiling:** `get_full_profile(df)` → `ProfilerReport`. Used after Strategy defines which columns to drop.

**Multi-sheet Strategy (`MultiSheetStrategy`) conventions**
- **analysis_mode**: `"separate" | "joined" | "both"` to control whether profiling runs per-sheet, on joined data, or both.
- **per_sheet_plans**: `SheetAnalysisPlan` entries indicate which sheets to profile and which columns are excluded or used for correlations.
- **joins_to_execute**: `JoinInstruction` list to materialize additional joined DataFrames prior to profiling.
- **cross_sheet_hypothesis**: A concrete, testable hypothesis that depends on relationships between sheets.

---

## Key Files

```
research.ipynb              # Main notebook (cell map below)
prompts/
  cdo_strategy.txt          # CDO system prompt (single-sheet)
  cdo_multi_sheet.txt       # CDO system prompt (multi-sheet) → MultiSheetStrategy
  analyst_synthesis.txt    # Analyst system prompt (structured CoT output)
  analyst_mission.txt      # Analyst user/mission prompt ({hypothesis}, {questions}, {data})
  critic_auditor.txt       # Critic system prompt
lib/
  path_utils.py             # resolve_file_path() — single entry point for path resolution
  data_discovery_lib.py     # run_dataset_discovery() → GlobalDiscoveryMap (.xlsx and .csv)
generated_modules/
  data_profiler.py          # get_full_profile() → ProfilerReport
```

---

## Notebook Cell Map

| Cell ID | Purpose |
|---------|---------|
| `d61d2243` | Basic imports (`os`, `pandas`, `logfire`, `dotenv`) |
| `d64283e7` | Logfire init — `logfire.instrument_pydantic(record='off')` |
| `5b95509f` | `validate_environment()` — checks all API keys |
| `47c821ca` | Pydantic models + agent definitions + `load_prompt()` |
| `730214d6` | `resolve_file_path()` + `execute_analysis_mission()` + `execute_multi_sheet_mission()` |
| `dd3f2ac7` | Launch/Display cell — calls `execute_analysis_mission()`, renders CoT, hypothesis validation, final report |

---

## Pydantic Models (cell `47c821ca`)

```python
class AnalystOutput(BaseModel):
    internal_thought_process: List[str]   # Step-by-step CoT, min 4 items
    hypothesis_validation: str             # CONFIRMED / PARTIALLY / REJECTED + evidence
    final_report_markdown: str             # Business report: Finding → Implication → Action

class AnalysisStrategy(BaseModel):
    id_like_columns: List[str]            # Always excluded from stats
    excluded_from_profiling: List[str]
    correlation_columns: List[str]        # CDO-approved cols for correlation
    business_hypothesis: str
    primary_questions: List[str]
    # ...

class CriticReport(BaseModel):
    approved: bool
    score: float                          # Forced to 0.0 on any "block" finding
    findings: List[CriticalFinding]
    revision_instructions: Optional[str]

# Phase 3 — Multi-sheet models (cell 47c821ca)

class SheetAnalysisPlan(BaseModel):
    sheet_name: str
    should_profile: bool
    id_like_columns: List[str]
    excluded_from_profiling: List[str]
    correlation_columns: List[str]
    primary_focus: str                    # CDO's one-liner focus for this sheet

class JoinInstruction(BaseModel):
    left_sheet: str
    right_sheet: str
    on_column: str                        # Must exist in both sheets; validated before merge
    join_type: Literal["inner", "left", "outer"]
    result_name: str                      # Key used to store merged DataFrame in dfs dict

class MultiSheetStrategy(BaseModel):
    business_hypothesis: str
    primary_questions: List[str]
    analysis_mode: Literal["separate", "joined", "both"]
    per_sheet_plans: List[SheetAnalysisPlan]
    joins_to_execute: List[JoinInstruction]
    cross_sheet_hypothesis: str           # Testable hypothesis spanning multiple sheets
    reasoning: str

class ColumnRelationship(BaseModel):
    parent_sheet: str
    parent_column: str
    child_sheet: str
    child_column: str
    confidence: float
    # Updated cardinality to include one_to_one
    cardinality: Literal["one_to_many", "many_to_many", "one_to_one", "unknown"] = "unknown"
    join_type_hint: str    
```

`CriticReport` has a `model_validator` that clamps `score` to `0.0` if any `CriticalFinding` has `severity="block"`.

---

## File Extension Handling

| Extension | Discovery (Phase 1) | Load (Phase 3) |
|-----------|---------------------|----------------|
| `.xlsx` | `discovery_map.sheets.get(target_sheet)` (`run_dataset_discovery` → `GlobalDiscoveryMap`) | `pd.read_excel(file_path, sheet_name=target_sheet)` |
| `.csv` | `discovery_map.sheets.get(path.stem)` or first sheet | `pd.read_csv(file_path)` — `target_sheet` ignored |

---

## Path Resolution

**Standard:** `resolve_file_path(path_input: str) -> Path` (in `lib/path_utils.py`) is the single entry point for path handling.

- Converts Windows-style backslashes to forward slashes.
- Attempts to resolve the path by checking a small set of pragmatic candidates (in order):
  - Relative to the current working directory (CWD)
  - Relative to the project root (`Path(__file__).parent.parent`)
  - Common Windows mount locations for the configured user (`/mnt/c/Users/grosp/Downloads/`, `/mnt/c/Users/grosp/Desktop/`)
- Returns an absolute `pathlib.Path`. If no candidates exist, it falls back to the CWD-resolved absolute path (even if missing) so Missions can fail fast with a clear error.
- Missions call `resolve_file_path` at the very beginning and use the resolved path for all subsequent file operations (Discovery, loading, logging).

---

## Hardening

- **Path Management:** All mission flows use `lib/path_utils.resolve_file_path()` as the single entry point. The resolved path is passed to `run_dataset_discovery()`, `pd.read_excel()`/`pd.read_csv()`, and logging.
- **WSL/Windows interop:** `resolve_file_path` includes convenience fallbacks for common Windows locations; if the Windows username changes, update the mount paths in `lib/path_utils.py`.
- **Join validation (Phase 3):** Before each `pd.DataFrame.merge`, two pre-flight checks run: (1) both sheets must be present in `dfs`; (2) `on_column` must exist in both DataFrames. Failures emit a `logfire.warn` + console message and `continue` to the next join — the mission is never aborted. An outer `except` catches unexpected runtime errors (e.g. dtype incompatibility) with the same skip-and-log behaviour.

---

## Display Cell Layout (cell `dd3f2ac7`)

Three rendered sections in order:
1. **🔬 Hypothesis Validation** — `mission_data["hypothesis_validation"]`
2. **🧠 Chain-of-Thought Reasoning** — `mission_data["reasoning"]` as a bulleted list
3. **📜 Final Validated Report** — `mission_data["narrative"]`

---

## Coding Standards

- All Python comments in **English only**
- All agent outputs typed with **Pydantic models** — no raw dicts from agents
- All prompt strings live in `prompts/*.txt` — loaded via `load_prompt(name)`
- No inline prompt strings in notebook cells

---

## Environment

- Provider: OpenRouter (`https://openrouter.ai/api/v1`) — key: `OPENROUTER_API_KEY`
- Logfire project: `grospi/agenticdataanalyst` (EU region)
- Runtime: WSL2 + Cursor, Jupyter kernel via `uv`
