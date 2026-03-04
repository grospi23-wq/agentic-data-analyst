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


from lib.path_utils import resolve_file_path
from lib.data_discovery_lib import run_dataset_discovery
from generated_modules.data_profiler import ProfilerReport, get_full_profile
from schema import SheetAnalysisPlan, SpecialistType, AnalystOutput
from agents import (
    AgentDeps,
    AnalystDeps,
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
# Mission A: Single-Sheet Analysis
# ---------------------------------------------------------------------------

async def execute_analysis_mission(
    file_path: str, target_sheet: str = "order_items"
) -> dict | None:
    """
    Execute a single-sheet analysis mission (Discovery → Strategy → Profiling → Synthesis).
    Handles .xlsx (with sheet selection) and .csv (sheet ignored).
    Returns a result dict on success, or None if the file is not found.
    """
    path = resolve_file_path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        logfire.error("File not found: {path}", path=str(path))
        print(f"❌ Error: File not found at {path}")
        return None

    with logfire.span("Analysis Mission: {file}", file=path.name):

        # Phase 1: Discovery
        with logfire.span("Phase 1: Discovery"):
            discovery_map = run_dataset_discovery(path)

            if extension == ".csv":
                sheet_key = path.stem
                sheet_meta = discovery_map.sheets.get(sheet_key) or next(
                    iter(discovery_map.sheets.values()), None
                )
            else:
                sheet_meta = discovery_map.sheets.get(target_sheet)

            if not sheet_meta:
                logfire.error("Sheet/source '{sheet}' not found", sheet=target_sheet)
                return None

        # Phase 2: Strategic Planning (CDO)
        with logfire.span("Phase 2: Strategic Planning") as span:
            strategy_res = await orchestrator_agent.run(
                f"Define strategy for {sheet_meta.name}: {sheet_meta.model_dump_json()}"
            )
            strategy = strategy_res.output
            span.set_attribute("hypothesis", strategy.business_hypothesis)
            print(f"🎯 CDO Hypothesis: {strategy.business_hypothesis}")

        # Phase 3: Tactical Execution (Profiling)
        with logfire.span("Phase 3: Tactical Execution"):
            if extension == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, sheet_name=target_sheet)

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

            # Build analyst deps with the live DataFrame for code execution
            analyst_deps = AnalystDeps(dfs={sheet_meta.name: df})

        # Phase 4: Synthesis + Critic Loop (Max 3 Attempts)
        with logfire.span("Phase 4: Synthesis & Validation"):
            final_narrative = ""
            final_report_obj = None
            critic_report = None

            # Domain-aware prompt selection
            specialist_type = strategy.specialist_type
            specialist_prompt_template = _load_specialist_prompt(specialist_type)

            # Define schema_context here to ensure it is always bound
            schema_context = f"Table '{sheet_meta.name}' — columns: {list(df.columns)}"

            logfire.info("Domain specialist activated: {domain}", domain=specialist_type.value)
            print(f"🔬 Domain Specialist: {specialist_type.value}")

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                print(f"\n🚀 --- STARTING ATTEMPT {attempt}/{max_attempts} ---")
                logfire.info("Starting attempt", attempt=attempt, max_attempts=max_attempts)

                # Build prompt outside the span so it never interferes with control flow.
                if attempt == 1:
                    schema_context = f"Table '{sheet_meta.name}' — columns: {list(df.columns)}"
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

                # Each attempt gets a fresh execution namespace so variables from a
                # previous (rejected) attempt don't bleed into the retry.
                analyst_deps = AnalystDeps(dfs={sheet_meta.name: df})

                # Span wraps only the agent calls — break/continue live outside it.
                with logfire.span(f"Attempt {attempt}"):
                    report_res = await analyst_agent.run(mission_prompt, deps=analyst_deps)
                    final_report_obj = report_res.output

                    with logfire.span("Self-Reflection"):
                        final_report_obj = await _self_validate_report(final_report_obj, analyst_deps,schema_context)
                        logfire.info("Self-validation complete")

                    final_narrative = final_report_obj.final_report_markdown

                    critic_input = (
                        f"PROFILER_DATA:\n{raw_results.model_dump_json()}\n\n"
                        f"ANALYST_NARRATIVE:\n{final_narrative}"
                    )
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output

                    # Log outcome inside span for traceability.
                    if critic_report.approved:
                        logfire.info(
                            "Report approved by Critic",
                            score=critic_report.score,
                            overall_value_score=critic_report.overall_value_score,
                        )
                        logfire.info(
                            "Analyst CoT", thoughts=final_report_obj.internal_thought_process
                        )
                    else:
                        logfire.warn(
                            f"Attempt {attempt} rejected",
                            structural_failures=critic_report.structural_failures,
                            low_value_insights=critic_report.low_value_insights,
                        )

                # break and rejection detail are outside the span — no control-flow conflict.
                if critic_report.approved:
                    print(
                        f"  ✅  Approved (Attempt {attempt}/{max_attempts})"
                        f" — Score {critic_report.score:.2f}"
                        f", Value {critic_report.overall_value_score:.2f}"
                    )
                    break

                _print_rejection_detail(critic_report, attempt, max_attempts)
            else:
                # for…else fires only when all attempts were exhausted without a break.
                logfire.warn(
                    "Published after max retries",
                    final_score=critic_report.score if critic_report else 0,
                )

        # Phase 5: Formatting (PowerPoint Generation)
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
                    file_name = f"output_{path.stem}_{target_sheet}.pptx"
                    out_path = str(OUTPUT_DIR / file_name)
                    pptx_path = render_pptx(format_res.output, out_path)
                    pptx_path_str = str(pptx_path)
                    logfire.info("PPTX generated", path=pptx_path_str)
                except Exception as e:
                    logfire.error("Failed to generate PPTX", error=str(e))
                    print(f"❌ Failed to generate presentation: {e}")

        # ── Mission Summary (printed after all Logfire spans have closed) ──
        _W = 64
        approved = bool(critic_report and critic_report.approved)
        score_str = (
            f"Critic {critic_report.score:.2f}  │  Value {critic_report.overall_value_score:.2f}"
            if critic_report else "N/A"
        )
        print(f"\n{'━' * _W}")
        if approved:
            print(f"  ✅  MISSION COMPLETE")
        else:
            print(f"  🚨  MISSION COMPLETE WITH WARNINGS  (Critic did not fully approve)")
        print(f"  📁  File       : {path.name}  /  {target_sheet}")
        print(f"  🔬  Specialist : {specialist_type.value}")
        print(f"  📊  Scores     : {score_str}")
        if pptx_path_str:
            print(f"  💾  PPTX       : {pptx_path_str}")
        else:
            print(f"  💾  PPTX       : not generated")
        print(f"{'━' * _W}\n")
        print("📜  FINAL VALIDATED REPORT\n")
        print(final_narrative)
        print(f"\n{'━' * _W}\n")

        return {
            "strategy": strategy,
            "profile_results": raw_results,
            "narrative": final_narrative,
            "hypothesis_validation": (
                final_report_obj.hypothesis_validation if final_report_obj else ""
            ),
            "critic_score": critic_report.score if critic_report else 0.0,
            "specialist_type": strategy.specialist_type.value,
            "reasoning": (
                final_report_obj.internal_thought_process if final_report_obj else []
            ),
            "pptx_path": pptx_path_str,
        }


# ---------------------------------------------------------------------------
# Mission B: Multi-Sheet Analysis  (Phase 4B: memory-aware)
# ---------------------------------------------------------------------------

async def execute_multi_sheet_mission(file_path: str) -> dict | None:
    """
    Execute a multi-sheet analysis mission with persistent memory.

    Phase 4B memory flow:
      1. generate_schema_hash() fingerprints the workbook structure.
      2. MemoryManager.load() retrieves prior join configs + quality warnings.
      3. AgentDeps injects the store into the Multi-Sheet CDO prompt.
      4. After the Critic loop, successful joins and findings are persisted.
    """
    path = resolve_file_path(file_path)

    if not path.exists():
        logfire.error("File not found: {path}", path=str(path))
        print(f"❌ Error: File not found at {path}")
        return None

    with logfire.span("Multi-Sheet Mission: {file}", file=path.name):

        # ------------------------------------------------------------------
        # Phase 1: Discovery + Schema Hash
        # ------------------------------------------------------------------
        with logfire.span("Phase 1: Discovery") as span:
            global_map = run_dataset_discovery(path)
            span.set_attribute("discovery_result", global_map.model_dump())
            logfire.info("Discovered {n} relationships", n=len(global_map.relationships))

            # Derive the structural fingerprint for this schema family
            schema_hash = generate_schema_hash(global_map.model_dump())
            span.set_attribute("schema_hash", schema_hash)
            print(f"🔑 Schema hash: {schema_hash}")

        # ------------------------------------------------------------------
        # Phase 1.5: Load Memory & Prepare Context
        # ------------------------------------------------------------------
        memory_manager = MemoryManager()
        await memory_manager.init()

        # Load the store (fingerprinted by schema_hash)
        memory_store: AnalystMemoryStore = await memory_manager.load(schema_hash)

        if memory_store.total_runs > 0:
            print(
                f"🧠 Memory loaded — {memory_store.total_runs} prior run(s), "
                f"{len(memory_store.successful_joins)} known join(s), "
                f"{len(memory_store.quality_warnings)} known warning(s)"
            )
        else:
            print("🆕 No prior memory for this schema family — starting fresh.")

        # Bundle deps for memory-aware agents
        agent_deps = AgentDeps(
            memory_store=memory_store,
            schema_hash=schema_hash,
        )

        # ------------------------------------------------------------------
        # Phase 2: Adaptive CDO Multi-Sheet Strategy
        # ------------------------------------------------------------------
        with logfire.span("Phase 2: Multi-Sheet Strategy") as span:
            # 1. Determine complexity tier
            num_sheets = len(global_map.sheets)
            total_cols = sum(len(s.columns) for s in global_map.sheets.values())
            
            # Tier logic: 
            # Small (<5 sheets): 10 samples (Full)
            # Medium (5-10 sheets): 5 samples (Balanced)
            # Large (>10 sheets): 2 samples + Minimal metadata (Lean)
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

            # 2. Prune the map based on calculated tier
            pruned_map = global_map.model_dump()
            for sheet_name, sheet_data in pruned_map.get("sheets", {}).items():
                # Apply sample limit
                if "sample_data" in sheet_data and isinstance(sheet_data["sample_data"], list):
                    sheet_data["sample_data"] = sheet_data["sample_data"][:sample_limit]
                
                # If in Lean mode (Northwind style), strip heavy stats to save tokens
                if lean_mode:
                    for col in sheet_data.get("columns", []):
                        # Keep only essential schema info
                        keys_to_keep = {"name", "type"}
                        for key in list(col.keys()):
                            if key not in keys_to_keep:
                                del col[key]

            # 3. Build the prompt text
            memory_context = memory_store.to_cdo_context_block()
            input_text = pb.build_cdo_multisheet_prompt(
                pruned_map=pruned_map,
                memory_context=memory_store.to_cdo_context_block(),
                lean_mode=lean_mode
            )

            # 4. Run the Agent
            logfire.info(f"Requesting strategy from CDO (Tier: {sample_limit} samples)...")
            strategy_res = await multi_orchestrator_agent.run(
                input_text,
                deps=agent_deps,
            )
            
            strategy = strategy_res.output
            span.set_attribute("analysis_mode", strategy.analysis_mode)
            print(f"🎯 Multi-Sheet Hypothesis: {strategy.business_hypothesis}")

        # ------------------------------------------------------------------
        # Phase 3: Data Loading & Joins
        # ------------------------------------------------------------------
        # Track which joins were actually executed so we can persist them.
        executed_join_records: List[JoinMemoryRecord] = []

        with logfire.span("Phase 3: Data Loading & Joins"):
            dfs: Dict[str, pd.DataFrame] = {}
            if path.suffix.lower() == ".xlsx":
                for sheet_name in global_map.sheets:
                    dfs[sheet_name] = pd.read_excel(path, sheet_name=sheet_name)
            else:
                dfs[path.stem] = pd.read_csv(path)

            for join_instr in strategy.joins_to_execute:
                try:
                    missing_sheets = [
                        s for s in (join_instr.left_sheet, join_instr.right_sheet)
                        if s not in dfs
                    ]
                    if missing_sheets:
                        logfire.warn(
                            "Join '{name}' skipped — sheets not loaded: {sheets}",
                            name=join_instr.result_name,
                            sheets=", ".join(missing_sheets),
                        )
                        print(
                            f"⚠️ Join '{join_instr.result_name}' skipped: "
                            f"sheet(s) not loaded — {', '.join(missing_sheets)}"
                        )
                        continue

                    left_df = dfs[join_instr.left_sheet]
                    right_df = dfs[join_instr.right_sheet]

                    schema_issues = [
                        f"'{join_instr.on_column}' missing in '{sheet}'"
                        for sheet, df in (
                            (join_instr.left_sheet, left_df),
                            (join_instr.right_sheet, right_df),
                        )
                        if join_instr.on_column not in df.columns
                    ]
                    if schema_issues:
                        logfire.warn(
                            "Join '{name}' skipped — schema mismatch: {issues}",
                            name=join_instr.result_name,
                            issues="; ".join(schema_issues),
                        )
                        print(
                            f"⚠️ Join '{join_instr.result_name}' skipped — "
                            f"schema mismatch: {'; '.join(schema_issues)}"
                        )
                        continue

                    merged = left_df.merge(
                        right_df, on=join_instr.on_column, how=join_instr.join_type
                    )
                    dfs[join_instr.result_name] = merged

                    # Record this join execution for later persistence
                    executed_join_records.append(
                        JoinMemoryRecord(
                            schema_hash=schema_hash,
                            left_sheet=join_instr.left_sheet,
                            right_sheet=join_instr.right_sheet,
                            on_column=join_instr.on_column,
                            join_type=join_instr.join_type,
                            result_name=join_instr.result_name,
                            row_count_after_join=len(merged),
                            critic_score=0.0,   # filled in after critic runs
                        )
                    )

                    logfire.info(
                        "Join created: {name} ({rows} rows)",
                        name=join_instr.result_name,
                        rows=len(merged),
                    )

                except Exception as e:
                    logfire.error(
                        "Join '{name}' failed unexpectedly: {err}",
                        name=join_instr.result_name,
                        err=str(e),
                    )
                    print(f"❌ Unexpected join error for '{join_instr.result_name}': {e}")

        # ------------------------------------------------------------------
        # Phase 4: Parallel Profiling
        # ------------------------------------------------------------------
        with logfire.span("Phase 4: Parallel Profiling"):
            active_plans = [p for p in strategy.per_sheet_plans if p.should_profile]
            tasks = [
                _profile_sheet_async(dfs[p.sheet_name], p)
                for p in active_plans
                if p.sheet_name in dfs
            ]
            profile_results_list = await asyncio.gather(*tasks)
            profile_results: Dict[str, ProfilerReport] = dict(profile_results_list)

        # ------------------------------------------------------------------
        # Phase 5: Synthesis & Validation Loop (Max 2 Attempts)
        # ------------------------------------------------------------------
        with logfire.span("Phase 5: Synthesis & Validation"):
            combined_data = {k: v.model_dump() for k, v in profile_results.items()}
            final_report_obj = None
            critic_report = None

            # Domain-aware prompt selection
            specialist_type = strategy.specialist_type
            specialist_prompt_template = _load_specialist_prompt(specialist_type)
            logfire.info(
                "Domain specialist activated: {domain}",
                domain=specialist_type.value,
            )
            print(f"🔬 Domain Specialist: {specialist_type.value}")

            # Build schema context once — used in every Attempt 1 prompt.
            schema_lines = [
                f"• '{name}': {list(df.columns)}" for name, df in dfs.items()
            ]
            schema_context = "\n".join(schema_lines)

            max_attempts = 4
            profiler_json_str = json.dumps(combined_data)
            for attempt in range(1, max_attempts + 1):
                print(f"\n🚀 --- STARTING ATTEMPT {attempt}/{max_attempts} ---")
                logfire.info("Starting attempt", attempt=attempt, max_attempts=max_attempts)

                # Each attempt gets a fresh execution namespace so variables from a
                # previous (rejected) attempt don't bleed into the retry.
                analyst_deps = AnalystDeps(dfs=dfs)

                # Build prompt outside the span so it never interferes with control flow.
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

                # Span wraps only the agent calls — break/continue live outside it.
                with logfire.span(f"Attempt {attempt}"):
                    report_res = await analyst_agent.run(mission_prompt, deps=analyst_deps)
                    final_report_obj = report_res.output

                    with logfire.span("Self-Reflection"):
                        final_report_obj = await _self_validate_report(final_report_obj, analyst_deps, schema_context)
                        logfire.info("Self-validation complete")

                    critic_input = (
                        f"PROFILER_DATA:\n{profiler_json_str}\n\n"
                        f"ANALYST_NARRATIVE:\n{final_report_obj.final_report_markdown}"
                    )
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output

                    # Log outcome inside span for traceability.
                    if critic_report.approved:
                        logfire.info(
                            "Multi-sheet report approved",
                            score=critic_report.score,
                            overall_value_score=critic_report.overall_value_score,
                        )
                    else:
                        logfire.warn(
                            f"Attempt {attempt} rejected",
                            structural_failures=critic_report.structural_failures,
                            low_value_insights=critic_report.low_value_insights,
                        )

                # break and rejection detail are outside the span — no control-flow conflict.
                if critic_report.approved:
                    print(
                        f"  ✅  Approved (Attempt {attempt}/{max_attempts})"
                        f" — Score {critic_report.score:.2f}"
                        f", Value {critic_report.overall_value_score:.2f}"
                    )
                    break

                _print_rejection_detail(critic_report, attempt, max_attempts)
            else:
                # for…else fires only when all attempts were exhausted without a break.
                logfire.warn(
                    "Published after max retries",
                    final_score=critic_report.score if critic_report else 0,
                )

        # ------------------------------------------------------------------
        # Phase 5.5: Persist Memory
        # ------------------------------------------------------------------
        with logfire.span("Phase 5.5: Memory Persistence"):
            if critic_report and final_report_obj:
                # Back-fill critic_score into every executed join record
                for jr in executed_join_records:
                    jr.critic_score = critic_report.score

                # Convert Critic findings → QualityWarning records
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

                logfire.info(
                    "Memory persisted",
                    schema_hash=schema_hash,
                    joins=len(executed_join_records),
                    warnings=len(new_warnings),
                    approved=critic_report.approved,
                )
            else:
                logfire.warn("Skipping memory persistence — no critic report available.")

        await memory_manager.close()

        # ------------------------------------------------------------------
        # Phase 6: Formatting (PowerPoint Generation)
        # ------------------------------------------------------------------
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
                    file_name = f"output_{path.stem}_multisheet.pptx"
                    out_path = str(OUTPUT_DIR / file_name)
                    pptx_path = render_pptx(format_res.output, out_path)
                    pptx_path_str = str(pptx_path)
                    logfire.info("PPTX generated", path=pptx_path_str)
                except Exception as e:
                    logfire.error("Failed to generate PPTX", error=str(e))
                    print(f"❌ Failed to generate presentation: {e}")

        # ── Mission Summary (printed after all Logfire spans have closed) ──
        _W = 64
        approved = bool(critic_report and critic_report.approved)
        score_str = (
            f"Critic {critic_report.score:.2f}  │  Value {critic_report.overall_value_score:.2f}"
            if critic_report else "N/A"
        )
        sheets_profiled = ", ".join(profile_results.keys()) if profile_results else "none"
        print(f"\n{'━' * _W}")
        if approved:
            print(f"  ✅  MISSION COMPLETE  (Multi-Sheet)")
        else:
            print(f"  🚨  MISSION COMPLETE WITH WARNINGS  (Critic did not fully approve)")
        print(f"  📁  File       : {path.name}")
        print(f"  🗂️   Sheets     : {sheets_profiled}")
        print(f"  🔬  Specialist : {specialist_type.value}")
        print(f"  📊  Scores     : {score_str}")
        if pptx_path_str:
            print(f"  💾  PPTX       : {pptx_path_str}")
        else:
            print(f"  💾  PPTX       : not generated")
        print(f"{'━' * _W}\n")
        print("📜  FINAL VALIDATED MULTI-SHEET REPORT\n")
        print(final_report_obj.final_report_markdown if final_report_obj else "")
        print(f"\n{'━' * _W}\n")

        return {
            "strategy": strategy,
            "profile_results": profile_results,
            "narrative": final_report_obj.final_report_markdown if final_report_obj else "",
            "hypothesis_validation": (
                final_report_obj.hypothesis_validation if final_report_obj else ""
            ),
            "critic_score": critic_report.score if critic_report else 0.0,
            "schema_hash": schema_hash,
            "specialist_type": strategy.specialist_type.value,
            "reasoning": (
                final_report_obj.internal_thought_process if final_report_obj else []
            ),
            "pptx_path": pptx_path_str,
        }


__all__ = ["execute_analysis_mission", "execute_multi_sheet_mission"]
