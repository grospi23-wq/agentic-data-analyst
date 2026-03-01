import asyncio
import json
import logfire
import pandas as pd
from pathlib import Path
from typing import Dict, cast

from lib.path_utils import resolve_file_path
from lib.data_discovery_lib import run_dataset_discovery
from generated_modules.data_profiler import ProfilerReport, get_full_profile
from schema import SheetAnalysisPlan
from agents import (
    orchestrator_agent,
    analyst_agent,
    critic_agent,
    multi_orchestrator_agent,
    formatter_agent,
    load_prompt,
)


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

            if hasattr(strategy, "correlation_columns") and strategy.correlation_columns:
                valid_corr_cols = [c for c in strategy.correlation_columns if c in df_clean.columns]
                if len(valid_corr_cols) >= 2:
                    corr_only_df = cast(pd.DataFrame, df_clean[valid_corr_cols])
                    raw_results.correlations = corr_only_df.corr(numeric_only=True).to_dict()

            combined_data = {sheet_meta.name: raw_results.model_dump()}

        # Phase 4: Synthesis + Critic Loop (Max 2 Attempts)
        with logfire.span("Phase 4: Synthesis & Validation"):
            revision_hint = ""
            final_narrative = ""
            final_report_obj = None
            critic_report = None

            for attempt in range(2):
                with logfire.span(f"Attempt {attempt + 1}"):
                    mission_prompt = load_prompt("analyst_mission").format(
                        hypothesis=strategy.business_hypothesis,
                        questions="\n".join(f"- {q}" for q in strategy.primary_questions),
                        data=json.dumps(combined_data, indent=2),
                        analysis_mode="separate",
                        cross_sheet_hypothesis="N/A",
                    )

                    if revision_hint:
                        mission_prompt += f"\n\nCRITICAL REVISION REQUIRED:\n{revision_hint}"

                    report_res = await analyst_agent.run(mission_prompt)
                    final_report_obj = report_res.output
                    final_narrative = final_report_obj.final_report_markdown

                    critic_input = (
                        f"PROFILER_DATA:\n{raw_results.model_dump_json()}\n\n"
                        f"ANALYST_NARRATIVE:\n{final_narrative}"
                    )
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output

                    if critic_report.approved:
                        logfire.info("Report approved by Critic", score=critic_report.score)
                        logfire.info(
                            "Analyst CoT", thoughts=final_report_obj.internal_thought_process
                        )
                        print(f"✅ Report Approved (Score: {critic_report.score})")
                        break

                    revision_hint = critic_report.revision_instructions
                    logfire.warn(f"Attempt {attempt + 1} blocked", findings=critic_report.findings)
                    print(f"⚠️ Revision Required (Attempt {attempt + 1}): {revision_hint}")
            else:
                logfire.warn(
                    "Published after max retries",
                    final_score=critic_report.score if critic_report else 0,
                )
                print("🚨 PUBLISHED WITH WARNING: Critic did not fully approve after 2 attempts.")

            print(f"\n{'═'*50}\n📜 FINAL VALIDATED REPORT\n{'═'*50}\n{final_narrative}")

        # Phase 5: Formatting (PowerPoint Generation)
        pptx_path_str = None
        with logfire.span("Phase 5: Formatting"):
            if critic_report and critic_report.approved and final_report_obj:
                from pptx_renderer import render_pptx

                format_prompt = (
                    f"AnalystOutput:\n{final_report_obj.model_dump_json()}\n\n"
                    f"deck_title: {sheet_meta.name} Analysis\n"
                    f"critic_score: {critic_report.score}\n"
                    f"analysis_mode: separate\n"
                )
                try:
                    format_res = await formatter_agent.run(format_prompt)
                    out_path = f"output_{path.stem}_{target_sheet}.pptx"
                    pptx_path = render_pptx(format_res.output, out_path)
                    pptx_path_str = str(pptx_path)
                    logfire.info("PPTX generated", path=pptx_path_str)
                    print(f"📊 Presentation saved to: {pptx_path_str}")
                except Exception as e:
                    logfire.error("Failed to generate PPTX", error=str(e))
                    print(f"❌ Failed to generate presentation: {e}")
            else:
                print("⏭️ Skipping PPTX generation because the report was not approved.")

        return {
            "strategy": strategy,
            "results": raw_results,
            "narrative": final_narrative,
            "hypothesis_validation": (
                final_report_obj.hypothesis_validation if final_report_obj else ""
            ),
            "reasoning": (
                final_report_obj.internal_thought_process if final_report_obj else []
            ),
            "critic_score": critic_report.score if critic_report else None,
            "pptx_path": pptx_path_str,
        }


async def execute_multi_sheet_mission(file_path: str) -> dict | None:
    """
    Execute a multi-sheet analysis mission.
    Discovery → MultiSheetStrategy → Joins → Parallel Profiling → Synthesis → PPTX.

    Args:
        file_path: User-provided path (relative or absolute). Resolved via resolve_file_path.

    Returns:
        A result dict on success, or None if the file cannot be found.
    """
    path = resolve_file_path(file_path)
    if not path.exists():
        logfire.error("File not found: {path}", path=str(path))
        return None

    with logfire.span("Multi-Sheet Mission: {file}", file=path.name):

        # Phase 1: Discovery
        with logfire.span("Phase 1: Discovery") as span:
            global_map = run_dataset_discovery(path)
            span.set_attribute("discovery_result", global_map.model_dump())
            logfire.info("Discovered {n} relationships", n=len(global_map.relationships))

        # Phase 2: CDO Multi-Sheet Strategy
        with logfire.span("Phase 2: Multi-Sheet Strategy") as span:
            strategy_res = await multi_orchestrator_agent.run(
                global_map.model_dump_json(indent=2)
            )
            strategy = strategy_res.output
            span.set_attribute("analysis_mode", strategy.analysis_mode)
            span.set_attribute("joins", len(strategy.joins_to_execute))
            print(f"🎯 Multi-Sheet Hypothesis: {strategy.business_hypothesis}")

        # Phase 3: Data Loading & Joins
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

        # Phase 4: Parallel Profiling
        with logfire.span("Phase 4: Parallel Profiling"):
            active_plans = [p for p in strategy.per_sheet_plans if p.should_profile]
            tasks = [
                _profile_sheet_async(dfs[p.sheet_name], p)
                for p in active_plans
                if p.sheet_name in dfs
            ]
            profile_results_list = await asyncio.gather(*tasks)
            profile_results: Dict[str, ProfilerReport] = dict(profile_results_list)

        # Phase 5: Synthesis & Validation Loop (Max 2 Attempts)
        with logfire.span("Phase 5: Synthesis & Validation"):
            combined_data = {k: v.model_dump() for k, v in profile_results.items()}
            revision_hint = ""
            final_report_obj = None
            critic_report = None

            for attempt in range(2):
                with logfire.span(f"Attempt {attempt + 1}"):
                    mission_prompt = load_prompt("analyst_mission").format(
                        hypothesis=strategy.business_hypothesis,
                        analysis_mode=strategy.analysis_mode,
                        cross_sheet_hypothesis=strategy.cross_sheet_hypothesis,
                        questions="\n".join(f"- {q}" for q in strategy.primary_questions),
                        data=json.dumps(combined_data, indent=2),
                    )

                    if revision_hint:
                        mission_prompt += f"\n\nCRITICAL REVISION REQUIRED:\n{revision_hint}"

                    report_res = await analyst_agent.run(mission_prompt)
                    final_report_obj = report_res.output

                    critic_input = (
                        f"PROFILER_DATA:\n{json.dumps(combined_data)}\n\n"
                        f"ANALYST_NARRATIVE:\n{final_report_obj.final_report_markdown}"
                    )
                    critic_res = await critic_agent.run(critic_input)
                    critic_report = critic_res.output

                    if critic_report.approved:
                        logfire.info("Multi-sheet report approved", score=critic_report.score)
                        print(f"✅ Report Approved (Score: {critic_report.score})")
                        break

                    revision_hint = critic_report.revision_instructions
                    logfire.warn(f"Attempt {attempt + 1} blocked", findings=critic_report.findings)
                    print(f"⚠️ Revision Required (Attempt {attempt + 1}): {revision_hint}")

            print(
                f"\n{'═'*50}\n📜 FINAL VALIDATED MULTI-SHEET REPORT\n{'═'*50}\n"
                f"{final_report_obj.final_report_markdown if final_report_obj else ''}"
            )

        # Phase 6: Formatting (PowerPoint Generation)
        pptx_path_str = None
        with logfire.span("Phase 6: Formatting"):
            if critic_report and critic_report.approved and final_report_obj:
                from pptx_renderer import render_pptx

                format_prompt = (
                    f"AnalystOutput:\n{final_report_obj.model_dump_json()}\n\n"
                    f"deck_title: {path.stem} Comprehensive Analysis\n"
                    f"critic_score: {critic_report.score}\n"
                    f"analysis_mode: {strategy.analysis_mode}\n"
                )
                try:
                    format_res = await formatter_agent.run(format_prompt)
                    out_path = f"output_{path.stem}_multisheet.pptx"
                    pptx_path = render_pptx(format_res.output, out_path)
                    pptx_path_str = str(pptx_path)
                    logfire.info("PPTX generated", path=pptx_path_str)
                    print(f"📊 Presentation saved to: {pptx_path_str}")
                except Exception as e:
                    logfire.error("Failed to generate PPTX", error=str(e))
                    print(f"❌ Failed to generate presentation: {e}")
            else:
                print("⏭️ Skipping PPTX generation because the multi-sheet report was not approved.")

        return {
            "strategy": strategy,
            "profile_results": profile_results,
            "narrative": final_report_obj.final_report_markdown if final_report_obj else "",
            "critic_score": critic_report.score if critic_report else 0.0,
            "reasoning": (
                final_report_obj.internal_thought_process if final_report_obj else []
            ),
            "pptx_path": pptx_path_str,
        }


__all__ = ["execute_analysis_mission", "execute_multi_sheet_mission"]
