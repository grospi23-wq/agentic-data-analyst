"""
missions.py
-----------
Orchestration layer for all analysis mission types.
Refactored to enforce DRY principles across Multi-Sheet and SQL pipelines.
Maintains backward compatibility for tests via function aliases.
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
# SHARED HELPERS: PROMPTING & FORMATTING
# ---------------------------------------------------------------------------

def _load_specialist_prompt(specialist_type: SpecialistType) -> str:
    """Loads the specific domain prompt or falls back to GENERAL."""
    specialist_path = Path(f"prompts/specialists/{specialist_type.value}.txt")
    
    if specialist_path.exists():
        return specialist_path.read_text(encoding="utf-8").strip()

    logfire.warn("No specialist prompt for '{t}' — falling back to GENERAL", t=specialist_type.value)
    return Path("prompts/specialists/GENERAL.txt").read_text(encoding="utf-8").strip()


_SELF_VALIDATION_TEMPLATE = load_prompt("self_validation")
_ANALYST_RETRY_TEMPLATE = load_prompt("analyst_retry")


async def _self_validate_report(report: AnalystOutput, analyst_deps: AnalystDeps, schema_context: str) -> AnalystOutput:
    """Performs a self-reflection pass to catch trivial errors before the Critic audit."""
    prompt = pb.build_self_validation_prompt(
        template=_SELF_VALIDATION_TEMPLATE,
        schema_context=schema_context,
        chain_of_thought=report.internal_thought_process,
        draft_narrative=report.final_report_markdown
    )
    result = await analyst_agent.run(prompt, deps=analyst_deps)
    return result.output


def _print_rejection_detail(critic_report, attempt: int, max_attempts: int) -> None:
    """Prints a structured summary of Critic rejections, including value scores and caps."""
    n = len(critic_report.required_rewrites)
    
    # The test expects both Score and Value in this specific format
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
        # Show only first 3 rewrites
        for rw in critic_report.required_rewrites[:3]:
            print(f"       [{rw.insight_id}] ({rw.issue_type}): {rw.fix_instruction}")
        
        # The test specifically expects this line if there are more than 3 rewrites
        if n > 3:
            print(f"       … and {n - 3} more (see Logfire for full list)")


def _format_rewrites(critic_report) -> str:
    """Converts structured Critic rewrites into a readable prompt string."""
    if critic_report.required_rewrites:
        lines = [
            f"- [{rw.insight_id}] ({rw.issue_type}): {rw.fix_instruction}" 
            for rw in critic_report.required_rewrites
        ]
        return "\n".join(lines)
    
    return critic_report.revision_instructions or "Revise the report to address the Critic's findings."


def _display_mission_summary(mission_name, source_label, source_value, items_label, items_value, specialist, critic_report, pptx_path, narrative):
    """Prints a final beautiful summary of the mission outcome."""
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


async def _profile_sheet_async(df: pd.DataFrame, plan: SheetAnalysisPlan) -> tuple[str, ProfilerReport]:
    """Runs statistical profiling on a single dataframe in a thread pool."""
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
# SHARED CORE PHASES (The DRY Engine)
# ---------------------------------------------------------------------------

async def _shared_phase2_strategy(global_map, memory_store, agent_deps, mission_label="Mission") -> MultiSheetStrategy | None:
    """Shared Phase 2: High-level orchestration and adaptive pruning."""
    with logfire.span(f"Phase 2: {mission_label} Strategy"):
        num_items = len(global_map.sheets)
        
        # Adaptive Pruning: Decide how much sample data to send to the LLM
        sample_limit = 10 if num_items <= 5 else (5 if num_items <= 10 else 2)
        lean_mode = num_items > 10

        pruned_map = global_map.model_dump()
        for _, sheet_data in pruned_map.get("sheets", {}).items():
            if "sample_data" in sheet_data:
                sheet_data["sample_data"] = sheet_data["sample_data"][:sample_limit]
            
            if lean_mode:
                for col in sheet_data.get("columns", []):
                    for k in list(col.keys()): 
                        if k not in {"name", "type"}: 
                            del col[k]

        input_text = pb.build_cdo_multisheet_prompt(
            pruned_map=pruned_map, 
            memory_context=memory_store.to_cdo_context_block(), 
            lean_mode=lean_mode
        )
        
        try:
            strategy_res = await multi_orchestrator_agent.run(input_text, deps=agent_deps)
            strategy = strategy_res.output
            print(f"🎯 {mission_label} Hypothesis: {strategy.business_hypothesis}")
            return strategy
        except Exception as e:
            logfire.error(f"{mission_label} strategy failed", error=str(e))
            return None


def _shared_phase3_joins(dfs: Dict[str, pd.DataFrame], strategy: MultiSheetStrategy, schema_hash: str) -> List[JoinMemoryRecord]:
    """Shared Phase 3: Executing tactical joins across DataFrames."""
    executed_records: List[JoinMemoryRecord] = []
    
    with logfire.span("Phase 3: Tactical Joins"):
        for join_instr in strategy.joins_to_execute:
            try:
                if join_instr.left_sheet not in dfs or join_instr.right_sheet not in dfs:
                    continue
                
                left_df, right_df = dfs[join_instr.left_sheet], dfs[join_instr.right_sheet]
                
                if join_instr.on_column not in left_df.columns or join_instr.on_column not in right_df.columns:
                    continue

                merged = left_df.merge(right_df, on=join_instr.on_column, how=join_instr.join_type)
                dfs[join_instr.result_name] = merged
                
                executed_records.append(JoinMemoryRecord(
                    schema_hash=schema_hash,
                    left_sheet=join_instr.left_sheet,
                    right_sheet=join_instr.right_sheet,
                    on_column=join_instr.on_column,
                    join_type=join_instr.join_type,
                    result_name=join_instr.result_name,
                    row_count_after_join=len(merged),
                    critic_score=0.0
                ))
            except Exception as e:
                logfire.error(f"Join {join_instr.result_name} failed: {e}")
                
    return executed_records


async def _shared_phase4_profiling(dfs, strategy, mission_label="Mission"):
    """Shared Phase 4: Parallel profiling of all active datasets."""
    with logfire.span(f"Phase 4: Parallel Profiling ({mission_label})"):
        active_plans = [p for p in strategy.per_sheet_plans if p.should_profile]
        tasks = [
            _profile_sheet_async(dfs[p.sheet_name], p) 
            for p in active_plans if p.sheet_name in dfs
        ]
        
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        profile_results = {}
        
        for item in raw_results:
            if not isinstance(item, BaseException):
                name, report = cast(tuple, item)
                profile_results[name] = report
                
        return profile_results, {k: v.model_dump() for k, v in profile_results.items()}


async def _shared_phase5_synthesis(dfs, combined_data, strategy, mission_label="Mission"):
    """Shared Phase 5: The iterative Analyst-Critic loop."""
    with logfire.span(f"Phase 5: Synthesis & Validation ({mission_label})"):
        final_report_obj, critic_report = None, None
        
        specialist_template = _load_specialist_prompt(strategy.specialist_type)
        schema_context = "\n".join([f"• '{n}': {list(d.columns)}" for n, d in dfs.items()])
        profiler_json_str = json.dumps(combined_data)

        # Implementation of the Specialist Agency and Retry Loop
        for attempt in range(1, 5):
            print(f"\n🚀 --- ATTEMPT {attempt}/4 ---")
            analyst_deps = AnalystDeps(dfs=dfs)
            
            if attempt == 1:
                prompt = pb.build_analyst_mission_prompt(
                    template=specialist_template, 
                    hypothesis=strategy.business_hypothesis, 
                    questions=strategy.primary_questions, 
                    data_json=profiler_json_str, 
                    schema_context=schema_context, 
                    analysis_mode=strategy.analysis_mode, 
                    cross_sheet_hypothesis=strategy.cross_sheet_hypothesis
                )
            else:
                prompt = pb.build_retry_prompt(
                    template=_ANALYST_RETRY_TEMPLATE, 
                    revision_instructions=_format_rewrites(critic_report), 
                    previous_narrative=final_report_obj.final_report_markdown if final_report_obj else ""
                )

            try:
                # Analyst Phase
                report_res = await analyst_agent.run(prompt, deps=analyst_deps)
                final_report_obj = report_res.output
                
                # Self-Reflection pass
                final_report_obj = await _self_validate_report(final_report_obj, analyst_deps, schema_context)
                
                # Critic Audit
                critic_res = await critic_agent.run(
                    f"PROFILER_DATA:\n{profiler_json_str}\n\nANALYST_NARRATIVE:\n{final_report_obj.final_report_markdown}"
                )
                critic_report = critic_res.output
                
                if critic_report.approved: 
                    break
                
                _print_rejection_detail(critic_report, attempt, 4)
            except Exception as e:
                logfire.error(f"Attempt {attempt} failed", error=str(e))
                
        return final_report_obj, critic_report


async def _shared_phase5_5_persist_memory(schema_hash, records, critic_report, final_report_obj, manager, store, label):
    """Shared Phase 5.5: Persisting findings to long-term memory."""
    if not (critic_report and final_report_obj): 
        return
        
    try:
        for jr in records: 
            jr.critic_score = critic_report.score
            
        warnings = [
            QualityWarning(
                schema_hash=schema_hash, 
                finding_type=f.finding_type, 
                affected_column_or_insight=f.affected_insight, 
                reason=f.reason, 
                severity=f.severity
            ) 
            for f in critic_report.findings
        ]
        
        await manager.persist(
            store=store, 
            new_joins=records, 
            new_warnings=warnings, 
            critic_approved=critic_report.approved
        )
    except Exception as e:
        logfire.error(f"Memory persistence failed for {label}: {e}")


async def _shared_phase6_formatting(title, filename, strategy, report, critic):
    """Shared Phase 6: PPTX generation architecture."""
    if not (critic and critic.approved and report): 
        return None
        
    try:
        prompt = pb.build_formatter_prompt(
            report_json=report.model_dump_json(), 
            deck_title=title, 
            critic_score=critic.score, 
            analysis_mode=strategy.analysis_mode
        )
        res = await formatter_agent.run(prompt)
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = str(OUTPUT_DIR / filename)
        
        from pptx_renderer import render_pptx
        return str(render_pptx(res.output, out_path))
    except Exception as e:
        logfire.error(f"Formatting failed for {title}: {e}")
        return None


# ---------------------------------------------------------------------------
# MISSION ENTRY POINTS
# ---------------------------------------------------------------------------

async def execute_analysis_mission(file_path: str, target_sheet: str = "order_items") -> dict | None:
    """Entry point for single-sheet file analysis."""
    path = resolve_file_path(file_path)
    if not path.exists(): 
        return None
    
    ext = path.suffix.lower()
    
    with logfire.span(f"Single-Sheet Mission: {path.name}"):
        discovery = run_dataset_discovery(path)
        meta = discovery.sheets.get(target_sheet) if ext != ".csv" else next(iter(discovery.sheets.values()), None)
        
        if not meta: 
            return None
            
        strat_res = await orchestrator_agent.run(f"Define strategy for {meta.name}: {meta.model_dump_json()}")
        strategy = strat_res.output
        
        df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path, sheet_name=meta.name)
        _, raw = await _profile_sheet_async(df, SheetAnalysisPlan(sheet_name=meta.name, **strategy.model_dump()))
        
        shared_strat = MultiSheetStrategy(
            **strategy.model_dump(), 
            per_sheet_plans=[], 
            joins_to_execute=[], 
            cross_sheet_hypothesis=""
        )
        
        report, critic = await _shared_phase5_synthesis(
            {meta.name: df}, 
            {meta.name: raw.model_dump()}, 
            shared_strat, 
            "Single-Sheet"
        )
        
        pptx = await _shared_phase6_formatting(
            f"{meta.name} Analysis", 
            f"output_{path.stem}_{meta.name}.pptx", 
            shared_strat, 
            report, 
            critic
        )
        
        narrative = report.final_report_markdown if report else ""
        _display_mission_summary("Single-Sheet", "File", path.name, "Sheet", target_sheet, strategy.specialist_type.value, critic, pptx, narrative)
        
        return {
            "narrative": narrative, 
            "critic_score": critic.score if critic else 0.0, 
            "pptx_path": pptx
        }


async def execute_multi_sheet_mission(file_path: str) -> dict | None:
    """Entry point for multi-sheet Excel/CSV file analysis."""
    path = resolve_file_path(file_path)
    if not path.exists(): 
        return None
        
    with logfire.span(f"Multi-Sheet Mission: {path.name}"):
        # Discovery & Memory Loading
        discovery = run_dataset_discovery(path)
        schema_hash = generate_schema_hash(discovery.model_dump())
        
        manager = MemoryManager()
        await manager.init()
        store = await manager.load(schema_hash)
        
        strategy = await _shared_phase2_strategy(
            discovery, 
            store, 
            AgentDeps(memory_store=store, schema_hash=schema_hash), 
            "Multi-Sheet"
        )
        
        if not strategy: 
            await manager.close()
            return None
            
        try:
            dfs = {
                n: pd.read_excel(path, sheet_name=n) if path.suffix == ".xlsx" else pd.read_csv(path) 
                for n in discovery.sheets
            }
            joins = _shared_phase3_joins(dfs, strategy, schema_hash)
            profiles, combined = await _shared_phase4_profiling(dfs, strategy, "Multi-Sheet")
            
            report, critic = await _shared_phase5_synthesis(dfs, combined, strategy, "Multi-Sheet")
            await _shared_phase5_5_persist_memory(schema_hash, joins, critic, report, manager, store, "Multi-Sheet")
        finally: 
            await manager.close()
            
        pptx = await _shared_phase6_formatting(f"{path.stem} Analysis", f"output_{path.stem}_multi.pptx", strategy, report, critic)
        
        _display_mission_summary("Multi-Sheet", "File", path.name, "Sheets", ", ".join(profiles.keys()), strategy.specialist_type.value, critic, pptx, report.final_report_markdown if report else "")
        
        return {
            "strategy": strategy, 
            "profile_results": profiles, 
            "narrative": report.final_report_markdown if report else "", 
            "pptx_path": pptx
        }


async def execute_sql_mission(connection_input: str) -> dict | None:
    """Entry point for SQL database analysis."""
    engine, display_path = resolve_sql_connection(connection_input)
    
    with logfire.span(f"SQL Mission: {display_path}"):
        discovery = run_sql_discovery(engine, display_path)
        schema_hash = generate_schema_hash(discovery.model_dump())
        
        manager = MemoryManager()
        await manager.init()
        store = await manager.load(schema_hash)
        
        strategy = await _shared_phase2_strategy(
            discovery, 
            store, 
            AgentDeps(memory_store=store, schema_hash=schema_hash), 
            "SQL"
        )
        
        if not strategy: 
            engine.dispose()
            await manager.close()
            return None
            
        try:
            # Table loading with proper quoting
            dfs = {
                n: pd.read_sql(text(f"SELECT * FROM {engine.dialect.identifier_preparer.quote(n)}"), engine) 
                for n in discovery.sheets
            }
            engine.dispose()
            
            joins = _shared_phase3_joins(dfs, strategy, schema_hash)
            profiles, combined = await _shared_phase4_profiling(dfs, strategy, "SQL")
            
            report, critic = await _shared_phase5_synthesis(dfs, combined, strategy, "SQL")
            await _shared_phase5_5_persist_memory(schema_hash, joins, critic, report, manager, store, "SQL")
        finally: 
            await manager.close()
            
        pptx = await _shared_phase6_formatting(f"{display_path.split('/')[-1]} DB", "output_sql.pptx", strategy, report, critic)
        
        _display_mission_summary("SQL", "DB", display_path, "Tables", ", ".join(profiles.keys()), strategy.specialist_type.value, critic, pptx, report.final_report_markdown if report else "")
        
        return {
            "strategy": strategy, 
            "profile_results": profiles, 
            "narrative": report.final_report_markdown if report else "", 
            "pptx_path": pptx
        }