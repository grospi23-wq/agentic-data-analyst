"""
agents.py
---------
Defines all PydanticAI agents used in the analysis pipeline.

Phase 4B additions:
  - AgentDeps dataclass — carries the AnalystMemoryStore into every agent run.
  - multi_orchestrator_agent now exposes a system_prompt dynamic tool that
    injects the historical memory context block from AgentDeps.

Phase 7 additions:
  - AnalystDeps dataclass — carries live DataFrames into the analyst agent.
  - execute_python_analysis tool — lets the analyst run arbitrary Python code
    against the loaded data to verify insights before writing the report.
"""

import ast
import asyncio
import contextlib
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from execution_backend import LocalExecBackend

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import logfire

from schema import (
    AnalysisStrategy,
    AnalystOutput,
    CriticReport,
    MultiSheetStrategy,
    PresentationSpec,
)
from memory_schema import AnalystMemoryStore

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Deps — injected into every agent.run() call via RunContext
# ---------------------------------------------------------------------------

@dataclass
class AgentDeps:
    """
    Dependency container passed to agents that need access to persistent
    memory (primarily the Multi-Sheet CDO) or shared run-level context.

    Fields
    ------
    memory_store:
        Loaded AnalystMemoryStore for the current schema_hash family.
        Defaults to an empty store so single-sheet missions can reuse
        the same agent definitions without special-casing.
    schema_hash:
        Propagated for convenience — avoids re-computing it in callbacks.
    """
    memory_store: AnalystMemoryStore = field(
        default_factory=lambda: AnalystMemoryStore(schema_hash="none")
    )
    schema_hash: str = "none"


@dataclass
class AnalystDeps:
    """
    Dependency container for the Analyst agent.

    Carries the live DataFrames so that the execute_python_analysis tool can
    run arbitrary Pandas code against the real data during synthesis.
    Defaults to an empty dict so self-reflection passes can reuse the agent
    without providing data access.

    namespace is the persistent execution environment shared across all
    execute_python_analysis calls within a single analyst run. Variables
    defined in one call (e.g. df = dfs['orders']) survive into the next,
    eliminating the need to redefine them on every invocation.
    """
    dfs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    namespace: dict = field(default_factory=dict)
    executor: LocalExecBackend = field(default_factory=LocalExecBackend)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    """Load a system prompt from the prompts/ directory by filename stem."""
    return Path(f"prompts/{name}.txt").read_text(encoding="utf-8").strip()


openrouter_provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Define a standard cap for tokens to avoid OpenRouter credit pre-auth errors
from pydantic_ai.settings import ModelSettings
STANDARD_SETTINGS = ModelSettings(max_tokens=6000)

# ---------------------------------------------------------------------------
# Agent 1: Chief Data Officer (single-sheet)
# ---------------------------------------------------------------------------
# claude-3.7-sonnet: strongest hypothesis reasoning and schema analysis.
# No memory injection needed — single-sheet missions don't persist joins.

orchestrator_agent: Agent[None, AnalysisStrategy] = Agent(
    OpenAIChatModel("anthropic/claude-3.7-sonnet", provider=openrouter_provider),
    output_type=AnalysisStrategy,
    system_prompt=load_prompt("cdo_strategy"),
    model_settings=STANDARD_SETTINGS,
)


# ---------------------------------------------------------------------------
# Agent 2: Senior Data Analyst
# ---------------------------------------------------------------------------
# gpt-4o: proven at long-form structured synthesis.
# deps_type=AnalystDeps: live DataFrames injected so the agent can run Python.

analyst_agent: Agent[AnalystDeps, AnalystOutput] = Agent(
    OpenAIChatModel("openai/gpt-4o", provider=openrouter_provider),
    deps_type=AnalystDeps,
    output_type=AnalystOutput,
    system_prompt=load_prompt("analyst_synthesis"),
)

_EXEC_TIMEOUT_SECONDS = 30


@analyst_agent.tool
async def execute_python_analysis(ctx: RunContext[AnalystDeps], code: str) -> str:
    """
    Executes Python code on the loaded dataset to verify insights or perform
    custom calculations. Use this to dive deeper than the precomputed Profiler JSON.

    ALLOWED LIBRARIES ONLY: pd (pandas), np (numpy). No seaborn, scipy, sklearn,
    or any other import. Attempting to import an unlisted library will raise a
    ModuleNotFoundError and waste the attempt.

    STRICT TABLE ACCESS: The execution namespace exposes `dfs`, a dict whose keys
    are exactly the sheet/table names discovered in Phase 1. Always call
    print(list(dfs.keys())) first if you are uncertain. Accessing a key that does
    not exist raises a KeyError — do NOT guess table names.

    The execution namespace contains:
      - dfs  : Dict[str, pd.DataFrame] — use exact sheet names from Phase 1.
      - pd   : pandas module.
      - np   : numpy module.

    The last expression in your code is automatically printed (Jupyter-style),
    so `df.describe()` works without an explicit print() wrapper.
    Variables you define persist across calls within the same attempt —
    you do NOT need to redefine `df = dfs['table']` in every block.
    Always use the exact column names present in the DataFrame; do not guess.
    If a tool call returns an error, fix your code — do NOT hallucinate numbers.
    """
    dfs = ctx.deps.dfs

    if not ctx.deps.namespace:
        ctx.deps.namespace.update({"dfs": dfs, "pd": pd, "np": np})
        
    
    try:
        result = await ctx.deps.executor.run(code, ctx.deps.namespace)
        
        logfire.info(
            "execute_python_analysis",
            code_preview=code[:300],
            output_preview=result[:300],
            available_dfs=list(dfs.keys())
        )
        return result

    except Exception as exc:
        error_msg = f"ERROR: {type(exc).__name__}: {exc}"
        logfire.error("execute_python_analysis_failed", error=error_msg)
        return error_msg


# ---------------------------------------------------------------------------
# Agent 3: Ruthless Auditor / Critic
# ---------------------------------------------------------------------------
# gemini-2.0-flash: lives inside a retry loop — speed and cost matter.
# Cross-provider auditing reduces same-family model bias in critiques.

critic_agent: Agent[None, CriticReport] = Agent(
    OpenAIChatModel("anthropic/claude-3.5-sonnet", provider=openrouter_provider),
    output_type=CriticReport,
    system_prompt=load_prompt("critic_auditor"),
    model_settings=STANDARD_SETTINGS,
)


# ---------------------------------------------------------------------------
# Agent 4: Multi-Sheet CDO — memory-aware
# ---------------------------------------------------------------------------
# claude-3.7-sonnet: most complex planning task (join graph reasoning).
# deps_type=AgentDeps: memory context is injected via a dynamic system prompt.

multi_orchestrator_agent: Agent[AgentDeps, MultiSheetStrategy] = Agent(
    OpenAIChatModel("anthropic/claude-3.7-sonnet", provider=openrouter_provider),
    deps_type=AgentDeps,
    output_type=MultiSheetStrategy,
    system_prompt=load_prompt("cdo_multi_sheet"),
    model_settings=STANDARD_SETTINGS,
)


@multi_orchestrator_agent.system_prompt
async def inject_memory_context(ctx: RunContext[AgentDeps]) -> str:
    """
    Dynamically append the historical memory block to the CDO system prompt.

    PydanticAI collects *all* @system_prompt functions and concatenates their
    return values with the static system_prompt string.  This keeps the base
    prompt clean while injecting run-specific memory at call time.
    """
    return ctx.deps.memory_store.to_cdo_context_block()


# ---------------------------------------------------------------------------
# Agent 5: Formatter
# ---------------------------------------------------------------------------
# gpt-4o-mini: pure JSON transformation — no frontier model needed.

formatter_agent: Agent[None, PresentationSpec] = Agent(
    OpenAIChatModel("openai/gpt-4o", provider=openrouter_provider),
    output_type=PresentationSpec,
    retries=5,
    system_prompt=load_prompt("formatter_agent"), 
    model_settings=STANDARD_SETTINGS,
)
