"""
missions.py
-----------
Orchestration layer for all analysis mission types.

Phase 4B changes (execute_multi_sheet_mission):
  - generate_schema_hash() called after discovery to derive the family key.
  - MemoryManager loads the prior AnalystMemoryStore before Phase 2.
  - AgentDeps carries the store into multi_orchestrator_agent.run().
  - Successful joins and Critic findings are persisted at the end of Phase 5.

Phase 5 (Specialist Agency):
  - Multi-Sheet CDO now classifies the dataset domain via Domain Discovery
    and populates MultiSheetStrategy.specialist_type.
  - Phase 5 synthesis loads a domain-tuned mission prompt from
    prompts/specialists/{SPECIALIST_TYPE}.txt, falling back to GENERAL.txt
    when no specialist file exists. Every specialist prompt enforces the
    "Gold Standard" trio: Pareto Concentration, Bimodal Detection, Paradox Flagging.

Phase 6 (Evaluation Architecture):
  - Self-Reflection pass added between analyst and critic to catch trivial/tautological
    insights before they reach the auditor (reduces critic rejections).
  - Retry loop uses lean prompts on attempt 2+: only previous narrative + structured
    required_rewrites are sent, not the full profiler JSON.
  - CriticReport.required_rewrites drives revision instructions (structured over prose).

Innovative Leap — Transparent Domain Routing:
  - specialist_type is logged to Logfire and printed to console so operators
    can see which domain engine is active for each run.
  - The fallback path is also logged (warn level) to surface missing specialists.
"""

import asyncio
import json
import logfire
import pandas as pd
from pathlib import Path
from typing import Dict, List, cast


from generated_modules.data_profiler import ProfilerReport, get_full_profile
from schema import SheetAnalysisPlan, SpecialistType, AnalystOutput, MultiSheetStrategy
from agents import (
    AgentDeps,
    AnalystDeps,
    AnalysisStrategy,
    orchestrator_agent,
    analyst_agent,
    critic_agent,
    multi_orchestrator_agent,
    formatter_agent,
    load_prompt,
)
from memory_schema import (
    AnalystMemoryStore,
    JoinMemoryRecord,
    QualityWarning,
    generate_schema_hash,
)
from memory_manager import MemoryManager
import prompt_builders as pb
from lib.path_utils import resolve_file_path, resolve_sql_connection
from lib.data_discovery_lib import run_dataset_discovery, run_sql_discovery
from sqlalchemy import text

OUTPUT_DIR = Path("outputs")



# ---------------------------------------------------------------------------
# Shared helper: specialist prompt loader
# ---------------------------------------------------------------------------

def _load_specialist_prompt(specialist_type: SpecialistType) -> str:
    """
    Load the analyst mission prompt for a given specialist domain.

    Resolution order:
      1. prompts/specialists/{SPECIALIST_TYPE}.txt  (domain-specific)
      2. prompts/specialists/GENERAL.txt             (universal fallback)

    Logs a warning when falling back so operators can track missing specialists.
    """
    specialist_path = Path(f"prompts/specialists/{specialist_type.value}.txt")
    if specialist_path.exists():
        return specialist_path.read_text(encoding="utf-8").strip()

    logfire.warn(
        "No specialist prompt for '{t}' — falling back to GENERAL",
        t=specialist_type.value,
    )
    return Path("prompts/specialists/GENERAL.txt").read_text(encoding="utf-8").strip()


# Load evaluation-architecture prompt templates once at import time.
_SELF_VALIDATION_TEMPLATE = load_prompt("self_validation")
_ANALYST_RETRY_TEMPLATE = load_prompt("analyst_retry")


# ---------------------------------------------------------------------------
# Shared helper: analyst self-reflection pass (Step 2)
# ---------------------------------------------------------------------------

async def _self_validate_report(report: AnalystOutput, analyst_deps: AnalystDeps, schema_context: str) -> AnalystOutput:
    """
    Run a self-reflection pass on the analyst's draft before it reaches the Critic.

    Reuses analyst_agent — no new agent is introduced.  The self-validation
    prompt instructs the analyst to check for noise-level correlations,
    small samples, tautologies, missing business impact, and novelty gaps.
    The returned AnalystOutput replaces the original draft for critic evaluation.

    analyst_deps is forwarded so the tool remains callable during reflection
    (though typical self-reflection passes don't need raw data execution).
    """
    prompt = pb.build_self_validation_prompt(
        template=_SELF_VALIDATION_TEMPLATE,
        schema_context=schema_context,
        chain_of_thought=report.internal_thought_process,
        draft_narrative=report.final_report_markdown
    )
    result = await analyst_agent.run(prompt, deps=analyst_deps)
    return result.output


# ---------------------------------------------------------------------------
# Shared helper: format structured revision payload (Step 4)
# ---------------------------------------------------------------------------

def _print_rejection_detail(critic_report, attempt: int, max_attempts: int) -> None:
    """
    Print a scannable rejection summary to the terminal during the retry loop.
    Surfaces structural failures and the first few rewrite instructions so the
    developer can see exactly why the Critic rejected without opening Logfire.
    """
    n = len(critic_report.required_rewrites)
    print(
        f"  ⚠️  Revision Required (Attempt {attempt}/{max_attempts})"
        f" — Score {critic_report.score:.2f}"
        f", Value {critic_report.overall_value_score:.2f}"
        f" — {n} rewrite(s)"
    )

    if critic_report.structural_failures:
        print("  🔴  Structural failures:")
        for failure in critic_report.structural_failures:
            print(f"       • {failure}")

    if critic_report.required_rewrites:
        print("  🔧  Required rewrites:")
        for rw in critic_report.required_rewrites[:3]:   # cap at 3 to stay readable
            print(f"       [{rw.insight_id}] ({rw.issue_type}): {rw.fix_instruction}")
        if n > 3:
            print(f"       … and {n - 3} more (see Logfire for full list)")


def _format_rewrites(critic_report) -> str:
    """
    Render CriticReport.required_rewrites as a compact bulleted list.

    Falls back to revision_instructions (free-form string) if required_rewrites
    is empty — maintains backward compatibility with any cached CriticReport
    that pre-dates the structured revision schema.
    """
    if critic_report.required_rewrites:
        lines = [
            f"- [{rw.insight_id}] ({rw.issue_type}): {rw.fix_instruction}"
            for rw in critic_report.required_rewrites
        ]
        return "\n".join(lines)
    return critic_report.revision_instructions or "Revise the report to address the Critic's findings."

# ---------------------------------------------------------------------------
# Shared helper: mission summary display
# ---------------------------------------------------------------------------

def _display_mission_summary(
    mission_name: str,
    source_label: str,
    source_value: str,
    items_label: str,
    items_value: str,
    specialist: str,
    critic_report,
    pptx_path: str | None,
    narrative: str
):
    """Prints a standardized, professional ASCII summary of the mission."""
    _W = 64
    approved = bool(critic_report and critic_report.approved)
    score_str = f"Critic {critic_report.score:.2f}" if critic_report else "N/A"
    
    print(f"\n{'━' * _W}")
    print(f"  {'✅' if approved else '🚨'}  MISSION COMPLETE ({mission_name})")
    print(f"  {source_label:<12} : {source_value}")
    print(f"  {items_label:<12} : {items_value}")
    print(f"  🔬  Specialist : {specialist}")
    print(f"  📊  Scores     : {score_str}")
    print(f"  💾  PPTX       : {pptx_path or 'not generated'}")
    print(f"{'━' * _W}\n")
    print(f"📜  FINAL VALIDATED {mission_name.upper()} REPORT\n\n{narrative}\n\n{'━' * _W}\n")



# ---------------------------------------------------------------------------
# Shared helper: async sheet profiler
# ---------------------------------------------------------------------------

async def _profile_sheet_async(
    df: pd.DataFrame, plan: SheetAnalysisPlan
) -> tuple[str, ProfilerReport]:
    """Profile a single sheet asynchronously, offloading CPU-bound work to a thread."""
    cols_to_drop = list(set(plan.id_like_columns + plan.excluded_from_profiling))
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    report = await asyncio.to_thread(get_full_profile, df_clean, skip_discrete_outliers=True)

    if plan.correlation_columns:
        valid = [c for c in plan.correlation_columns if c in df_clean.columns]
        if len(valid) >= 2:
            target_df = cast(pd.DataFrame, df_clean[valid])
            report.correlations = await asyncio.to_thread(
                lambda: target_df.corr(numeric_only=True).to_dict()
            )

    return plan.sheet_name, report


# ---------------------------------------------------------------------------
# Mission A: Single-Sheet Analysis (Refactored)
# ---------------------------------------------------------------------------

def _single_phase1_discovery(path: Path, target_sheet: str, extension: str):
    """Phase 1: Load and discover dataset metadata with automatic fallback."""
    with logfire.span("Phase 1: Discovery"):
        discovery_map = run_dataset_discovery(path)
        
        # logic for CSV: take the only existing sheet
        if extension == ".csv":
            sheet_key = path.stem
            sheet_meta = discovery_map.sheets.get(sheet_key) or next(
                iter(discovery_map.sheets.values()), None
            )
        else:
            # Logic for Excel:
            # 1. Try to find the specific sheet requested
            sheet_meta = discovery_map.sheets.get(target_sheet)
            
            # 2. Automatic Fallback: If not found and there's only ONE sheet, use it!
            if not sheet_meta and len(discovery_map.sheets) == 1:
                sheet_meta = next(iter(discovery_map.sheets.values()))
                logfire.info("Target sheet not found, defaulting to the only available sheet: {s}", s=sheet_meta.name)
            elif not sheet_meta:
                # Still not found and multiple sheets exist - we can't guess
                available = ", ".join(discovery_map.sheets.keys())
                logfire.error("Sheet '{sheet}' not found. Available sheets: {available}", sheet=target_sheet, available=available)
                print(f"❌ Error: Sheet '{target_sheet}' not found. Available: {available}")
                return None

        return sheet_meta


async def _single_phase2_strategy(sheet_meta) -> AnalysisStrategy | None:
    """Phase 2: CDO defines the strategic analysis plan."""
    with logfire.span("Phase 2: Strategic Planning") as span:
        try:
            strategy_res = await orchestrator_agent.run(
                f"Define strategy for {sheet_meta.name}: {sheet_meta.model_dump_json()}"
            )
            strategy = strategy_res.output
            span.set_attribute("hypothesis", strategy.business_hypothesis)
            print(f"🎯 CDO Hypothesis: {strategy.business_hypothesis}")
            return strategy
        except Exception as e:
            logfire.error("Phase 2 strategy failed: {error}", error=str(e))
            print(f"❌ CDO strategy agent failed: {e}")
            return None


def _single_phase3_profiling(path: Path, target_sheet: str, extension: str, strategy: AnalysisStrategy, sheet_meta):
    """Phase 3: Load real data and execute statistical profiling."""
    with logfire.span("Phase 3: Tactical Execution"):
        try:
            if extension == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, sheet_name=sheet_meta.name)
        except Exception as e:
            logfire.error("Phase 3 data load failed: {error}", error=str(e))
            print(f"❌ Failed to load data from {path}: {e}")
            return None

        try:
            cols_to_drop = list(set(strategy.id_like_columns + strategy.excluded_from_profiling))
            df_clean = df.drop(
                columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
            )

            raw_results = get_full_profile(df_clean)

            if strategy.correlation_columns:
                valid_corr_cols = [c for c in strategy.correlation_columns if c in df_clean.columns]
                if len(valid_corr_cols) >= 2:
                    corr_only_df = cast(pd.DataFrame, df_clean[valid_corr_cols])
                    raw_results.correlations = corr_only_df.corr(numeric_only=True).to_dict()

            combined_data = {sheet_meta.name: raw_results.model_dump()}
            return df, raw_results, combined_data
        except Exception as e:
            logfire.error("Phase 3 profiling failed: {error}", error=str(e))
            print(f"❌ Statistical profiling failed: {e}")
            return None


async def _single_phase4_synthesis(
    df: pd.DataFrame, combined_data: dict, raw_results, strategy: AnalysisStrategy, sheet_meta
):
    """Phase 4: Analyst writes the report and Critic evaluates it in a retry loop."""
    with logfire.span("Phase 4: Synthesis & Validation"):
        final_narrative = ""
        final_report_obj = None
        critic_report = None

        specialist_type = strategy.specialist_type
        specialist_prompt_template = _load_specialist_prompt(specialist_type)
        schema_context = f"Table '{sheet_meta.name}' — columns: {list(df.columns)}"

        logfire.info("Domain specialist activated: {domain}", domain=specialist_type.value)
        print(f"🔬 Domain Specialist: {specialist_type.value}")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"\n🚀 --- STARTING ATTEMPT {attempt}/{max_attempts} ---")
            logfire.info("Starting attempt", attempt=attempt, max_attempts=max_attempts)

            if attempt == 1:
                mission_prompt = pb.build_analyst_mission_prompt(
                    template=specialist_prompt_template,
                    hypothesis=strategy.business_hypothesis,
                    questions=strategy.primary_questions,
                    data_json=json.dumps(combined_data, indent=2),
                    schema_context=schema_context,
                    analysis_mode="separate"
                )
            else:
                rewrites_text = _format_rewrites(critic_report)
                safe_narrative = final_report_obj.final_report_markdown if final_report_obj else ""
                mission_prompt = pb.build_retry_prompt(
                    template=_ANALYST_RETRY_TEMPLATE,
                    revision_instructions=rewrites_text,
                    previous_narrative=safe_narrative
                )

            analyst_deps = AnalystDeps(dfs={sheet_meta.name: df})

            with logfire.span(f"Attempt {attempt}"):
                try:
                    report_res = await analyst_agent.run(mission_prompt, deps=analyst_deps)
                    final_report_obj = report_res.output
                except Exception as e:
                    logfire.error("Analyst agent failed on attempt {a}: {err}", a=attempt, err=str(e))
                    print(f"❌ Analyst failed on attempt {attempt}: {e}")
                    continue

                try:
                    with logfire.span("Self-Reflection"):
                        final_report_obj = await _self_validate_report(final_report_obj, analyst_deps, schema_context)
                        logfire.info("Self-validation complete")
                except Exception as e:
                    logfire.warn("Self-validation failed, using raw draft: {err}", err=str(e))

                final_narrative = final_report_obj.final_report_markdown

                critic_input = (
                    f"PROFILER_DATA:\n{raw_results.model_dump_json()}\n\n"
                    f"ANALYST_NARRATIVE:\n{final_narrative}"
                )
                try:
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output
                except Exception as e:
                    logfire.error("Critic agent failed on attempt {a}: {err}", a=attempt, err=str(e))
                    print(f"❌ Critic failed on attempt {attempt}: {e}")
                    continue

                if critic_report.approved:
                    logfire.info("Report approved by Critic", score=critic_report.score)
                else:
                    logfire.warn(f"Attempt {attempt} rejected", structural_failures=critic_report.structural_failures)

            if critic_report and critic_report.approved:
                print(f"  ✅  Approved (Attempt {attempt}/{max_attempts}) — Score {critic_report.score:.2f}")
                break

            _print_rejection_detail(critic_report, attempt, max_attempts)
        else:
            logfire.warn("Published after max retries", final_score=critic_report.score if critic_report else 0)

        return final_report_obj, critic_report, final_narrative


async def _single_phase5_formatting(path: Path, sheet_meta, final_report_obj, critic_report) -> str | None:
    """Phase 5: Convert the validated JSON report into a native PowerPoint deck."""
    pptx_path_str = None
    with logfire.span("Phase 5: Formatting"):
        if critic_report and critic_report.approved and final_report_obj:
            from pptx_renderer import render_pptx

            format_prompt = pb.build_formatter_prompt(
                report_json=final_report_obj.model_dump_json(),
                deck_title=f"{sheet_meta.name} Analysis",
                critic_score=critic_report.score,
                analysis_mode="separate"
            )
            try:
                format_res = await formatter_agent.run(format_prompt)
                OUTPUT_DIR.mkdir(exist_ok=True)
                file_name = f"output_{path.stem}_{sheet_meta.name}.pptx"
                out_path = str(OUTPUT_DIR / file_name)
                pptx_path = render_pptx(format_res.output, out_path)
                pptx_path_str = str(pptx_path)
                logfire.info("PPTX generated", path=pptx_path_str)
            except Exception as e:
                logfire.error("Failed to generate PPTX", error=str(e))
                print(f"❌ Failed to generate presentation: {e}")
    return pptx_path_str


async def execute_analysis_mission(file_path: str, target_sheet: str = "order_items") -> dict | None:
    """
    Main orchestration function for Single-Sheet analysis.
    Refactored to use modular phase execution for clean architecture.
    """
    path = resolve_file_path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        logfire.error("File not found: {path}", path=str(path))
        print(f"❌ Error: File not found at {path}")
        return None

    with logfire.span("Analysis Mission: {file}", file=path.name):
        
        # 1. Discovery
        sheet_meta = _single_phase1_discovery(path, target_sheet, extension)
        if not sheet_meta:
            return None
            
        # 2. Strategy
        strategy = await _single_phase2_strategy(sheet_meta)
        if not strategy:
            return None

        # 3. Execution (Profiling)
        phase3_result = _single_phase3_profiling(path, target_sheet, extension, strategy, sheet_meta)
        if not phase3_result:
            return None
        df, raw_results, combined_data = phase3_result
        
        # 4. Synthesis & Validation Loop
        final_report_obj, critic_report, final_narrative = await _single_phase4_synthesis(
            df, combined_data, raw_results, strategy, sheet_meta
        )
        
        # 5. Presentation Rendering
        pptx_path_str = await _single_phase5_formatting(path, sheet_meta, final_report_obj, critic_report)

        # ── Mission Summary ──
        _display_mission_summary(
            mission_name="Single-Sheet",
            source_label="File",
            source_value=f"{path.name}",
            items_label="Sheet",
            items_value=target_sheet,
            specialist=strategy.specialist_type.value,
            critic_report=critic_report,
            pptx_path=pptx_path_str,
            narrative=final_narrative
        )

        return {
            "strategy": strategy,
            "profile_results": raw_results,
            "narrative": final_narrative,
            "hypothesis_validation": final_report_obj.hypothesis_validation if final_report_obj else "",
            "critic_score": critic_report.score if critic_report else 0.0,
            "specialist_type": strategy.specialist_type.value,
            "reasoning": final_report_obj.internal_thought_process if final_report_obj else [],
            "pptx_path": pptx_path_str,
        }


# ---------------------------------------------------------------------------
# Mission B: Multi-Sheet Analysis (Refactored)
# ---------------------------------------------------------------------------

async def _multi_phase1_discovery(path: Path):
    """Phase 1: Discover schema and generate unique fingerprint."""
    with logfire.span("Phase 1: Discovery") as span:
        try:
            global_map = run_dataset_discovery(path)
            span.set_attribute("discovery_result", global_map.model_dump())
            logfire.info("Discovered {n} relationships", n=len(global_map.relationships))

            schema_hash = generate_schema_hash(global_map.model_dump())
            span.set_attribute("schema_hash", schema_hash)
            print(f"🔑 Schema hash: {schema_hash}")
            return global_map, schema_hash
        except Exception as e:
            logfire.error("Phase 1 discovery failed: {error}", error=str(e))
            print(f"❌ Dataset discovery failed: {e}")
            return None, None


async def _multi_phase2_strategy(global_map, memory_store, agent_deps) -> MultiSheetStrategy | None:
    """Phase 2: Adaptive CDO defines the join and analysis strategy."""
    with logfire.span("Phase 2: Multi-Sheet Strategy") as span:
        num_sheets = len(global_map.sheets)
        total_cols = sum(len(s.columns) for s in global_map.sheets.values())
        
        if num_sheets <= 5:
            sample_limit = 10
            lean_mode = False
        elif num_sheets <= 10:
            sample_limit = 5
            lean_mode = False
        else:
            sample_limit = 2
            lean_mode = True

        logfire.info(f"Adaptive Pruning: {num_sheets} sheets, {total_cols} cols. Mode: {'Lean' if lean_mode else 'Full'}")

        pruned_map = global_map.model_dump()
        for sheet_name, sheet_data in pruned_map.get("sheets", {}).items():
            if "sample_data" in sheet_data and isinstance(sheet_data["sample_data"], list):
                sheet_data["sample_data"] = sheet_data["sample_data"][:sample_limit]
            
            if lean_mode:
                for col in sheet_data.get("columns", []):
                    keys_to_keep = {"name", "type"}
                    for key in list(col.keys()):
                        if key not in keys_to_keep:
                            del col[key]

        input_text = pb.build_cdo_multisheet_prompt(
            pruned_map=pruned_map,
            memory_context=memory_store.to_cdo_context_block(),
            lean_mode=lean_mode
        )

        logfire.info(f"Requesting strategy from CDO (Tier: {sample_limit} samples)...")
        try:
            strategy_res = await multi_orchestrator_agent.run(input_text, deps=agent_deps)
            strategy = strategy_res.output
        except Exception as e:
            logfire.error("Multi-sheet CDO strategy failed: {error}", error=str(e))
            print(f"❌ Multi-sheet CDO strategy agent failed: {e}")
            return None
        span.set_attribute("analysis_mode", strategy.analysis_mode)
        print(f"🎯 Multi-Sheet Hypothesis: {strategy.business_hypothesis}")
        return strategy


def _multi_phase3_joins(path: Path, global_map, strategy: MultiSheetStrategy, schema_hash: str):
    """Phase 3: Load DataFrames using discovered names and execute AI-planned joins."""
    executed_join_records: List[JoinMemoryRecord] = []
    dfs: Dict[str, pd.DataFrame] = {}

    with logfire.span("Phase 3: Data Loading & Joins"):
        if path.suffix.lower() == ".xlsx":
            for sheet_name in global_map.sheets:
                try:
                    dfs[sheet_name] = pd.read_excel(path, sheet_name=sheet_name)
                except Exception as e:
                    logfire.error("Failed to load sheet '{s}': {err}", s=sheet_name, err=str(e))
                    print(f"⚠️  Sheet '{sheet_name}' skipped — load failed: {e}")
        else:
            try:
                dfs[path.stem] = pd.read_csv(path)
            except Exception as e:
                logfire.error("Failed to load CSV '{path}': {err}", path=str(path), err=str(e))
                print(f"❌ CSV load failed: {e}")

        for join_instr in strategy.joins_to_execute:
            try:
                missing_sheets = [s for s in (join_instr.left_sheet, join_instr.right_sheet) if s not in dfs]
                if missing_sheets:
                    logfire.warn("Join skipped — sheets missing: {sheets}", sheets=", ".join(missing_sheets))
                    print(f"⚠️ Join '{join_instr.result_name}' skipped: sheets not loaded — {', '.join(missing_sheets)}")
                    continue

                left_df = dfs[join_instr.left_sheet]
                right_df = dfs[join_instr.right_sheet]

                schema_issues = [
                    f"'{join_instr.on_column}' missing in '{sheet}'"
                    for sheet, df in ((join_instr.left_sheet, left_df), (join_instr.right_sheet, right_df))
                    if join_instr.on_column not in df.columns
                ]
                if schema_issues:
                    logfire.warn("Join skipped — schema mismatch: {issues}", issues="; ".join(schema_issues))
                    print(f"⚠️ Join '{join_instr.result_name}' skipped — schema mismatch: {'; '.join(schema_issues)}")
                    continue

                merged = left_df.merge(right_df, on=join_instr.on_column, how=join_instr.join_type)
                dfs[join_instr.result_name] = merged

                executed_join_records.append(
                    JoinMemoryRecord(
                        schema_hash=schema_hash,
                        left_sheet=join_instr.left_sheet,
                        right_sheet=join_instr.right_sheet,
                        on_column=join_instr.on_column,
                        join_type=join_instr.join_type,
                        result_name=join_instr.result_name,
                        row_count_after_join=len(merged),
                        critic_score=0.0,
                    )
                )
                logfire.info("Join created: {name} ({rows} rows)", name=join_instr.result_name, rows=len(merged))

            except Exception as e:
                logfire.error("Join '{name}' failed: {err}", name=join_instr.result_name, err=str(e))
                print(f"❌ Unexpected join error for '{join_instr.result_name}': {e}")

    return dfs, executed_join_records


async def _multi_phase4_profiling(dfs: Dict[str, pd.DataFrame], strategy: MultiSheetStrategy):
    """Phase 4: Run statistical profiling on loaded and joined sheets."""
    with logfire.span("Phase 4: Parallel Profiling"):
        active_plans = [p for p in strategy.per_sheet_plans if p.should_profile]
        tasks = [_profile_sheet_async(dfs[p.sheet_name], p) for p in active_plans if p.sheet_name in dfs]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        profile_results: Dict[str, ProfilerReport] = {}
        for item in raw_results:
            if isinstance(item, BaseException):
                logfire.error("Sheet profiling task failed: {error}", error=str(item))
                print(f"⚠️  A profiling task failed and was skipped: {item}")
            else:
                sheet_name, report = cast(tuple, item)
                profile_results[sheet_name] = report
        combined_data = {k: v.model_dump() for k, v in profile_results.items()}
        return profile_results, combined_data


async def _multi_phase5_synthesis(dfs: Dict[str, pd.DataFrame], combined_data: dict, strategy: MultiSheetStrategy):
    """Phase 5: Multi-attempt synthesis loop with Analyst and Critic."""
    with logfire.span("Phase 5: Synthesis & Validation"):
        final_report_obj = None
        critic_report = None

        specialist_type = strategy.specialist_type
        specialist_prompt_template = _load_specialist_prompt(specialist_type)
        logfire.info("Domain specialist activated: {domain}", domain=specialist_type.value)
        print(f"🔬 Domain Specialist: {specialist_type.value}")

        schema_lines = [f"• '{name}': {list(df.columns)}" for name, df in dfs.items()]
        schema_context = "\n".join(schema_lines)
        profiler_json_str = json.dumps(combined_data)

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            print(f"\n🚀 --- STARTING ATTEMPT {attempt}/{max_attempts} ---")
            analyst_deps = AnalystDeps(dfs=dfs)

            if attempt == 1:
                mission_prompt = pb.build_analyst_mission_prompt(
                    template=specialist_prompt_template,
                    hypothesis=strategy.business_hypothesis,
                    questions=strategy.primary_questions,
                    data_json=profiler_json_str,
                    schema_context=schema_context,
                    analysis_mode=strategy.analysis_mode,
                    cross_sheet_hypothesis=strategy.cross_sheet_hypothesis
                )
            else:
                rewrites_text = _format_rewrites(critic_report)
                safe_narrative = final_report_obj.final_report_markdown if final_report_obj else ""
                mission_prompt = pb.build_retry_prompt(
                    template=_ANALYST_RETRY_TEMPLATE,
                    revision_instructions=rewrites_text,
                    previous_narrative=safe_narrative
                )   

            with logfire.span(f"Attempt {attempt}"):
                try:
                    report_res = await analyst_agent.run(mission_prompt, deps=analyst_deps)
                    final_report_obj = report_res.output
                except Exception as e:
                    logfire.error("Analyst agent failed on attempt {a}: {err}", a=attempt, err=str(e))
                    print(f"❌ Analyst failed on attempt {attempt}: {e}")
                    continue

                try:
                    with logfire.span("Self-Reflection"):
                        final_report_obj = await _self_validate_report(final_report_obj, analyst_deps, schema_context)
                except Exception as e:
                    logfire.warn("Self-validation failed, using raw draft: {err}", err=str(e))

                critic_input = f"PROFILER_DATA:\n{profiler_json_str}\n\nANALYST_NARRATIVE:\n{final_report_obj.final_report_markdown}"
                try:
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output
                except Exception as e:
                    logfire.error("Critic agent failed on attempt {a}: {err}", a=attempt, err=str(e))
                    print(f"❌ Critic failed on attempt {attempt}: {e}")
                    continue

                if critic_report.approved:
                    logfire.info("Multi-sheet report approved", score=critic_report.score)
                else:
                    logfire.warn(f"Attempt {attempt} rejected")

            if critic_report and critic_report.approved:
                print(f"  ✅  Approved (Attempt {attempt}/{max_attempts}) — Score {critic_report.score:.2f}")
                break
            _print_rejection_detail(critic_report, attempt, max_attempts)
        else:
            logfire.warn("Published after max retries")

        return final_report_obj, critic_report


async def _multi_phase5_5_persist_memory(
    schema_hash: str,
    executed_join_records: List[JoinMemoryRecord],
    critic_report,
    final_report_obj,
    memory_manager,
    memory_store
):
    """Phase 5.5: Persist join successes and critic findings to local memory."""
    with logfire.span("Phase 5.5: Memory Persistence"):
        if not (critic_report and final_report_obj):
            logfire.warn("Skipping memory persistence — no critic report.")
            return
        try:
            for jr in executed_join_records:
                jr.critic_score = critic_report.score

            new_warnings: List[QualityWarning] = [
                QualityWarning(
                    schema_hash=schema_hash,
                    finding_type=finding.finding_type,
                    affected_column_or_insight=finding.affected_insight,
                    reason=finding.reason,
                    severity=finding.severity,
                )
                for finding in critic_report.findings
            ]

            await memory_manager.persist(
                store=memory_store,
                new_joins=executed_join_records,
                new_warnings=new_warnings,
                critic_approved=critic_report.approved,
            )
            logfire.info("Memory persisted")
        except Exception as e:
            logfire.error("Memory persistence failed: {error}", error=str(e))
            print(f"⚠️  Memory persistence skipped due to error: {e}")


async def _multi_phase6_formatting(path: Path, strategy: MultiSheetStrategy, final_report_obj, critic_report) -> str | None:
    """Phase 6: Render the final presentation deck."""
    pptx_path_str = None
    with logfire.span("Phase 6: Formatting"):
        if critic_report and critic_report.approved and final_report_obj:
            from pptx_renderer import render_pptx
            format_prompt = pb.build_formatter_prompt(
                report_json=final_report_obj.model_dump_json(),
                deck_title=f"{path.stem} Comprehensive Analysis",
                critic_score=critic_report.score,
                analysis_mode=strategy.analysis_mode
            )
            try:
                format_res = await formatter_agent.run(format_prompt)
                OUTPUT_DIR.mkdir(exist_ok=True)
                out_path = str(OUTPUT_DIR / f"output_{path.stem}_multisheet.pptx")
                pptx_path = render_pptx(format_res.output, out_path)
                pptx_path_str = str(pptx_path)
                logfire.info("PPTX generated", path=pptx_path_str)
            except Exception as e:
                logfire.error("Failed to generate PPTX", error=str(e))
                print(f"❌ Failed to generate presentation: {e}")
    return pptx_path_str


async def execute_multi_sheet_mission(file_path: str) -> dict | None:
    """Main Orchestrator for Multi-Sheet analysis missions."""
    path = resolve_file_path(file_path)
    if not path.exists():
        logfire.error("File not found: {path}", path=str(path))
        print(f"❌ Error: File not found at {path}")
        return None

    with logfire.span("Multi-Sheet Mission: {file}", file=path.name):

        # 1. Discovery & Hash
        global_map, schema_hash = await _multi_phase1_discovery(path)
        if not global_map or not schema_hash:
            return None

        # 1.5. Load Memory
        memory_manager = MemoryManager()
        try:
            await memory_manager.init()
            memory_store = await memory_manager.load(schema_hash)
        except Exception as e:
            logfire.error("Memory init/load failed: {error}", error=str(e))
            print(f"⚠️  Memory system unavailable, starting fresh: {e}")
            memory_store = AnalystMemoryStore(schema_hash=schema_hash)
        agent_deps = AgentDeps(memory_store=memory_store, schema_hash=schema_hash)
        if memory_store.total_runs > 0:
            print(f"🧠 Memory loaded — {memory_store.total_runs} prior run(s)")

        # 2. Strategy
        strategy = await _multi_phase2_strategy(global_map, memory_store, agent_deps)
        if not strategy:
            await memory_manager.close()
            return None

        try:
            # 3. Data Loading & Joins
            dfs, executed_join_records = _multi_phase3_joins(path, global_map, strategy, schema_hash)

            # 4. Profiling
            profile_results, combined_data = await _multi_phase4_profiling(dfs, strategy)

            # 5. Synthesis Loop
            final_report_obj, critic_report = await _multi_phase5_synthesis(dfs, combined_data, strategy)

            # 5.5. Memory Persistence
            await _multi_phase5_5_persist_memory(
                schema_hash, executed_join_records, critic_report, final_report_obj, memory_manager, memory_store
            )
        finally:
            await memory_manager.close()

        # 6. Formatting
        pptx_path_str = await _multi_phase6_formatting(path, strategy, final_report_obj, critic_report)

        # ── Mission Summary ──
        narrative = final_report_obj.final_report_markdown if final_report_obj else ""
        sheets_profiled = ", ".join(profile_results.keys()) if profile_results else "none"
        
        _display_mission_summary(
            mission_name="Multi-Sheet",
            source_label="File",
            source_value=path.name,
            items_label="Sheets",
            items_value=sheets_profiled,
            specialist=strategy.specialist_type.value,
            critic_report=critic_report,
            pptx_path=pptx_path_str,
            narrative=final_report_obj.final_report_markdown if final_report_obj else ""
        )

        return {
            "strategy": strategy,
            "profile_results": profile_results,
            "narrative": narrative,
            "hypothesis_validation": final_report_obj.hypothesis_validation if final_report_obj else "",
            "critic_score": critic_report.score if critic_report else 0.0,
            "schema_hash": schema_hash,
            "specialist_type": strategy.specialist_type.value,
            "reasoning": final_report_obj.internal_thought_process if final_report_obj else [],
            "pptx_path": pptx_path_str,
        }


# ---------------------------------------------------------------------------
# Mission C: SQL Database Analysis (Refactored)
# ---------------------------------------------------------------------------

async def _sql_phase1_discovery(connection_input: str):
    """Phase 1: Discover database schema and generate unique fingerprint."""
    engine, display_path = resolve_sql_connection(connection_input)
    with logfire.span("Phase 1: Discovery") as span:
        try:
            global_map = run_sql_discovery(engine, display_path)
        except Exception as e:
            logfire.error("SQL Discovery failed: {error}", error=str(e))
            print(f"❌ Error connecting to database: {e}")
            return None, None, None, None

        span.set_attribute("discovery_result", global_map.model_dump())
        logfire.info("Discovered {n} relationships", n=len(global_map.relationships))

        schema_hash = generate_schema_hash(global_map.model_dump())
        span.set_attribute("schema_hash", schema_hash)
        print(f"🔑 SQL Schema hash: {schema_hash}")
        return engine, display_path, global_map, schema_hash


async def _sql_phase2_strategy(global_map, memory_store, agent_deps) -> MultiSheetStrategy | None:
    """Phase 2: Adaptive CDO defines the SQL join and analysis strategy."""
    with logfire.span("Phase 2: Database Strategy") as span:
        num_tables = len(global_map.sheets)
        total_cols = sum(len(s.columns) for s in global_map.sheets.values())
        
        if num_tables <= 5:
            sample_limit = 10
            lean_mode = False
        elif num_tables <= 10:
            sample_limit = 5
            lean_mode = False
        else:
            sample_limit = 2
            lean_mode = True

        logfire.info(f"Adaptive Pruning: {num_tables} tables, {total_cols} cols. Mode: {'Lean' if lean_mode else 'Full'}")

        pruned_map = global_map.model_dump()
        for sheet_name, sheet_data in pruned_map.get("sheets", {}).items():
            if "sample_data" in sheet_data and isinstance(sheet_data["sample_data"], list):
                sheet_data["sample_data"] = sheet_data["sample_data"][:sample_limit]
            
            if lean_mode:
                for col in sheet_data.get("columns", []):
                    keys_to_keep = {"name", "type"}
                    for key in list(col.keys()):
                        if key not in keys_to_keep:
                            del col[key]

        input_text = pb.build_cdo_multisheet_prompt(
            pruned_map=pruned_map,
            memory_context=memory_store.to_cdo_context_block(),
            lean_mode=lean_mode
        )

        logfire.info(f"Requesting strategy from CDO (Tier: {sample_limit} samples)...")
        try:
            strategy_res = await multi_orchestrator_agent.run(input_text, deps=agent_deps)
            strategy = strategy_res.output
        except Exception as e:
            logfire.error("SQL CDO strategy failed: {error}", error=str(e))
            print(f"❌ SQL CDO strategy agent failed: {e}")
            return None
        span.set_attribute("analysis_mode", strategy.analysis_mode)
        print(f"🎯 SQL Strategy Hypothesis: {strategy.business_hypothesis}")
        return strategy


def _sql_phase3_joins(engine, global_map, strategy: MultiSheetStrategy, schema_hash: str):
    """Phase 3: Load tables into DataFrames and execute AI-planned joins."""
    executed_join_records: List[JoinMemoryRecord] = []
    dfs: Dict[str, pd.DataFrame] = {}

    with logfire.span("Phase 3: Data Loading & Joins"):
        for table_name in global_map.sheets:
            try:
                quoted_name = engine.dialect.identifier_preparer.quote(table_name)
                dfs[table_name] = pd.read_sql(text(f"SELECT * FROM {quoted_name}"), engine)
                logfire.info("Table loaded: {name}", name=table_name)
            except Exception as e:
                logfire.error("Failed to load table '{t}': {err}", t=table_name, err=str(e))
                print(f"⚠️  Table '{table_name}' skipped — load failed: {e}")

        for join_instr in strategy.joins_to_execute:
            try:
                missing_sheets = [s for s in (join_instr.left_sheet, join_instr.right_sheet) if s not in dfs]
                if missing_sheets:
                    continue

                left_df = dfs[join_instr.left_sheet]
                right_df = dfs[join_instr.right_sheet]

                if join_instr.on_column not in left_df.columns or join_instr.on_column not in right_df.columns:
                    continue

                merged = left_df.merge(right_df, on=join_instr.on_column, how=join_instr.join_type)
                dfs[join_instr.result_name] = merged

                executed_join_records.append(
                    JoinMemoryRecord(
                        schema_hash=schema_hash,
                        left_sheet=join_instr.left_sheet,
                        right_sheet=join_instr.right_sheet,
                        on_column=join_instr.on_column,
                        join_type=join_instr.join_type,
                        result_name=join_instr.result_name,
                        row_count_after_join=len(merged),
                        critic_score=0.0,
                    )
                )
                logfire.info("Join created: {name} ({rows} rows)", name=join_instr.result_name, rows=len(merged))

            except Exception as e:
                logfire.error("Join '{name}' failed unexpectedly: {err}", name=join_instr.result_name, err=str(e))

    return dfs, executed_join_records


async def _sql_phase4_profiling(dfs: Dict[str, pd.DataFrame], strategy: MultiSheetStrategy):
    """Phase 4: Run statistical profiling on loaded SQL tables and joined views."""
    with logfire.span("Phase 4: Parallel Profiling"):
        active_plans = [p for p in strategy.per_sheet_plans if p.should_profile]
        tasks = [_profile_sheet_async(dfs[p.sheet_name], p) for p in active_plans if p.sheet_name in dfs]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        profile_results: Dict[str, ProfilerReport] = {}
        for item in raw_results:
            if isinstance(item, BaseException):
                logfire.error("SQL table profiling task failed: {error}", error=str(item))
                print(f"⚠️  A SQL profiling task failed and was skipped: {item}")
            else:
                table_name, report = cast(tuple, item)
                profile_results[table_name] = report
        combined_data = {k: v.model_dump() for k, v in profile_results.items()}
        return profile_results, combined_data


async def _sql_phase5_synthesis(dfs: Dict[str, pd.DataFrame], combined_data: dict, strategy: MultiSheetStrategy):
    """Phase 5: Multi-attempt synthesis loop with Analyst and Critic."""
    with logfire.span("Phase 5: Synthesis & Validation"):
        final_report_obj = None
        critic_report = None

        specialist_type = strategy.specialist_type
        specialist_prompt_template = _load_specialist_prompt(specialist_type)
        logfire.info("Domain specialist activated: {domain}", domain=specialist_type.value)
        print(f"🔬 Domain Specialist: {specialist_type.value}")

        schema_lines = [f"• '{name}': {list(df.columns)}" for name, df in dfs.items()]
        schema_context = "\n".join(schema_lines)
        profiler_json_str = json.dumps(combined_data)

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            print(f"\n🚀 --- STARTING ATTEMPT {attempt}/{max_attempts} ---")
            analyst_deps = AnalystDeps(dfs=dfs)

            if attempt == 1:
                mission_prompt = pb.build_analyst_mission_prompt(
                    template=specialist_prompt_template,
                    hypothesis=strategy.business_hypothesis,
                    questions=strategy.primary_questions,
                    data_json=profiler_json_str,
                    schema_context=schema_context,
                    analysis_mode=strategy.analysis_mode,
                    cross_sheet_hypothesis=strategy.cross_sheet_hypothesis
                )
            else:
                rewrites_text = _format_rewrites(critic_report)
                safe_narrative = final_report_obj.final_report_markdown if final_report_obj else ""
                mission_prompt = pb.build_retry_prompt(
                    template=_ANALYST_RETRY_TEMPLATE,
                    revision_instructions=rewrites_text,
                    previous_narrative=safe_narrative
                )   

            with logfire.span(f"Attempt {attempt}"):
                try:
                    report_res = await analyst_agent.run(mission_prompt, deps=analyst_deps)
                    final_report_obj = report_res.output
                except Exception as e:
                    logfire.error("Analyst agent failed on attempt {a}: {err}", a=attempt, err=str(e))
                    print(f"❌ Analyst failed on attempt {attempt}: {e}")
                    continue

                try:
                    with logfire.span("Self-Reflection"):
                        final_report_obj = await _self_validate_report(final_report_obj, analyst_deps, schema_context)
                except Exception as e:
                    logfire.warn("Self-validation failed, using raw draft: {err}", err=str(e))

                critic_input = f"PROFILER_DATA:\n{profiler_json_str}\n\nANALYST_NARRATIVE:\n{final_report_obj.final_report_markdown}"
                try:
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output
                except Exception as e:
                    logfire.error("Critic agent failed on attempt {a}: {err}", a=attempt, err=str(e))
                    print(f"❌ Critic failed on attempt {attempt}: {e}")
                    continue

                if critic_report.approved:
                    logfire.info("SQL report approved", score=critic_report.score)
                else:
                    logfire.warn(f"Attempt {attempt} rejected")

            if critic_report and critic_report.approved:
                print(f"  ✅  Approved (Attempt {attempt}/{max_attempts}) — Score {critic_report.score:.2f}")
                break
            _print_rejection_detail(critic_report, attempt, max_attempts)
        else:
            logfire.warn("Published after max retries")

        return final_report_obj, critic_report


async def _sql_phase5_5_persist_memory(
    schema_hash: str,
    executed_join_records: List[JoinMemoryRecord],
    critic_report,
    final_report_obj,
    memory_manager,
    memory_store
):
    """Phase 5.5: Persist join successes and critic findings to local memory."""
    with logfire.span("Phase 5.5: Memory Persistence"):
        if not (critic_report and final_report_obj):
            logfire.warn("Skipping SQL memory persistence — no critic report.")
            return
        try:
            for jr in executed_join_records:
                jr.critic_score = critic_report.score

            new_warnings: List[QualityWarning] = []
            for finding in critic_report.findings:
                new_warnings.append(
                    QualityWarning(
                        schema_hash=schema_hash,
                        finding_type=finding.finding_type,
                        affected_column_or_insight=finding.affected_insight,
                        reason=finding.reason,
                        severity=finding.severity,
                    )
                )

            await memory_manager.persist(
                store=memory_store,
                new_joins=executed_join_records,
                new_warnings=new_warnings,
                critic_approved=critic_report.approved,
            )
            logfire.info("Memory persisted")
        except Exception as e:
            logfire.error("SQL memory persistence failed: {error}", error=str(e))
            print(f"⚠️  SQL memory persistence skipped due to error: {e}")


async def _sql_phase6_formatting(display_path: str, strategy: MultiSheetStrategy, final_report_obj, critic_report) -> str | None:
    """Phase 6: Render the final presentation deck for the SQL analysis."""
    pptx_path_str = None
    with logfire.span("Phase 6: Formatting"):
        if critic_report and critic_report.approved and final_report_obj:
            from pptx_renderer import render_pptx
            
            safe_name = "".join(c if c.isalnum() else "_" for c in display_path.split("/")[-1])
            deck_title = f"{safe_name} DB Analysis"

            format_prompt = pb.build_formatter_prompt(
                report_json=final_report_obj.model_dump_json(),
                deck_title=deck_title,
                critic_score=critic_report.score,
                analysis_mode=strategy.analysis_mode
            )
            try:
                format_res = await formatter_agent.run(format_prompt)
                OUTPUT_DIR.mkdir(exist_ok=True)
                out_path = str(OUTPUT_DIR / f"output_{safe_name}_sql.pptx")
                pptx_path = render_pptx(format_res.output, out_path)
                pptx_path_str = str(pptx_path)
                logfire.info("PPTX generated", path=pptx_path_str)
            except Exception as e:
                logfire.error("Failed to generate PPTX", error=str(e))
                print(f"❌ Failed to generate presentation: {e}")
    return pptx_path_str


async def execute_sql_mission(connection_input: str) -> dict | None:
    """Main Orchestrator for SQL Database analysis missions."""
    
    # 1. Discovery & Connection
    engine, display_path, global_map, schema_hash = await _sql_phase1_discovery(connection_input)
    
    # Strict Type Guard: Ensure display_path is validated alongside other variables
    if not engine or not display_path or not global_map or not schema_hash:
        return None

    with logfire.span("SQL Mission: {db}", db=display_path):

        # 1.5. Load Memory
        memory_manager = MemoryManager()
        try:
            await memory_manager.init()
            memory_store = await memory_manager.load(schema_hash)
        except Exception as e:
            logfire.error("SQL memory init/load failed: {error}", error=str(e))
            print(f"⚠️  Memory system unavailable, starting fresh: {e}")
            memory_store = AnalystMemoryStore(schema_hash=schema_hash)
        agent_deps = AgentDeps(memory_store=memory_store, schema_hash=schema_hash)
        if memory_store.total_runs > 0:
            print(f"🧠 Memory loaded — {memory_store.total_runs} prior run(s)")

        # 2. Strategy
        strategy = await _sql_phase2_strategy(global_map, memory_store, agent_deps)
        if not strategy:
            await memory_manager.close()
            return None

        try:
            # 3. Data Loading & Joins (SQL Engine) — engine disposed in finally
            try:
                dfs, executed_join_records = _sql_phase3_joins(engine, global_map, strategy, schema_hash)
            finally:
                # CRITICAL: Disconnect from DB immediately after loading, even on partial failure
                engine.dispose()

            # 4. Profiling
            profile_results, combined_data = await _sql_phase4_profiling(dfs, strategy)

            # 5. Synthesis Loop
            final_report_obj, critic_report = await _sql_phase5_synthesis(dfs, combined_data, strategy)

            # 5.5. Memory Persistence
            await _sql_phase5_5_persist_memory(
                schema_hash, executed_join_records, critic_report, final_report_obj, memory_manager, memory_store
            )
        finally:
            await memory_manager.close()

        # 6. Formatting
        pptx_path_str = await _sql_phase6_formatting(display_path, strategy, final_report_obj, critic_report)

        # -- Mission Summary --
        narrative = final_report_obj.final_report_markdown if final_report_obj else ""
        tables_profiled = ", ".join(profile_results.keys()) if profile_results else "none"
        
        _display_mission_summary(
            mission_name="SQL Database",
            source_label="Database",
            source_value=display_path,
            items_label="Tables",
            items_value=tables_profiled,
            specialist=strategy.specialist_type.value,
            critic_report=critic_report,
            pptx_path=pptx_path_str,
            narrative=final_report_obj.final_report_markdown if final_report_obj else ""
        )

        return {
            "strategy": strategy,
            "profile_results": profile_results,
            "narrative": narrative,
            "hypothesis_validation": final_report_obj.hypothesis_validation if final_report_obj else "",
            "critic_score": critic_report.score if critic_report else 0.0,
            "schema_hash": schema_hash,
            "specialist_type": strategy.specialist_type.value,
            "reasoning": final_report_obj.internal_thought_process if final_report_obj else [],
            "pptx_path": pptx_path_str,
        }