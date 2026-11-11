"""
agents.py
---------
Defines all PydanticAI agents used in the analysis pipeline.
Optimized for modular execution and structured visualization data.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

import logfire
# Import both backends and the protocol for type hinting
from execution_backend import LocalExecBackend, E2BBackend, CodeExecutionBackend
from schema import (
    AnalysisStrategy,
    AnalystOutput,
    CriticReport,
    MultiSheetStrategy,
    PresentationSpec,
)
from memory_schema import AnalystMemoryStore

# Load environment configuration
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Dependency Containers
# ---------------------------------------------------------------------------

@dataclass
class AnalystDeps:
    """
    Container for Analyst Agent dependencies.
    - dfs: Dictionary of DataFrames for analysis.
    - executor: The engine used to run Python code (Local or E2B).
    - namespace: State dictionary maintained between code executions.
    """
    dfs: Dict[str, pd.DataFrame]
    executor: Optional[CodeExecutionBackend] = None
    namespace: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentDeps:
    """
    Shared dependencies for orchestration agents (CDO).
    Carries historical memory context for multi-sheet/SQL missions.
    """
    memory_store: AnalystMemoryStore = field(
        default_factory=lambda: AnalystMemoryStore(schema_hash="none")
    )
    schema_hash: str = "none"

# ---------------------------------------------------------------------------
# Backend Factory
# ---------------------------------------------------------------------------

def get_execution_backend() -> CodeExecutionBackend:
    """
    Composition Root helper: Selects the appropriate execution engine.
    - If E2B_API_KEY is present in environment, returns E2B cloud sandbox.
    - Otherwise, falls back to LocalExecBackend for development.
    """
    api_key = os.getenv("E2B_API_KEY")
    if api_key:
        # Secure cloud execution
        return E2BBackend(api_key=api_key)

    # Standard local execution fallback
    return LocalExecBackend()

# ---------------------------------------------------------------------------
# Provider & Model Configuration
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    """Load a system prompt template from the prompts/ directory."""
    return Path(f"prompts/{name}.txt").read_text(encoding="utf-8").strip()


api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise EnvironmentError("OPENROUTER_API_KEY is not set in .env file.")

openrouter_provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# ---------------------------------------------------------------------------
# Model Configurations (Tiered Resource Allocation)
# ---------------------------------------------------------------------------

# --- Model Selection ---
# Model IDs can be overridden independently without changing source code.
# OpenRouter's Auto Router avoids pinning the project to a retired model.
DEFAULT_MODEL = "openrouter/auto"
CDO_MODEL = os.getenv("OPENROUTER_CDO_MODEL", DEFAULT_MODEL)
ANALYST_MODEL = os.getenv("OPENROUTER_ANALYST_MODEL", DEFAULT_MODEL)
CRITIC_MODEL = os.getenv("OPENROUTER_CRITIC_MODEL", DEFAULT_MODEL)
FORMATTER_MODEL = os.getenv("OPENROUTER_FORMATTER_MODEL", DEFAULT_MODEL)

# --- Heavy: High-capacity planning and synthesis (CDO, Analyst) ---
# Used for complex schema mapping and long-form report writing.
HEAVY_MODEL = ModelSettings(max_tokens=8000)

# --- Medium: Evaluation and validation (Critic) ---
# Balanced for thorough auditing without excessive verbosity.
MEDIUM_CLAUDE = ModelSettings(max_tokens=5000)

# --- Light: Structured output and formatting (Formatter) ---
# Constrained for rigid JSON output to ensure speed and cost-efficiency.
LIGHT_GPT = ModelSettings(max_tokens=3000)

# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

# Agent 1: Chief Data Officer (Single-Sheet)
orchestrator_agent: Agent[None, AnalysisStrategy] = Agent(
    OpenAIChatModel(CDO_MODEL, provider=openrouter_provider),
    output_type=AnalysisStrategy,
    system_prompt=load_prompt("cdo_strategy"),
    model_settings=HEAVY_MODEL,
)

# Agent 2: Senior Data Analyst
analyst_agent: Agent[AnalystDeps, AnalystOutput] = Agent(
    OpenAIChatModel(ANALYST_MODEL, provider=openrouter_provider),
    deps_type=AnalystDeps,
    output_type=AnalystOutput,
    system_prompt=load_prompt("analyst_synthesis"),
    model_settings=HEAVY_MODEL,
)

# Agent 3: Critic / Auditor
critic_agent: Agent[None, CriticReport] = Agent(
    OpenAIChatModel(CRITIC_MODEL, provider=openrouter_provider),
    output_type=CriticReport,
    system_prompt=load_prompt("critic_auditor"),
    model_settings=MEDIUM_CLAUDE,
)

# Agent 4: Multi-Sheet/SQL CDO
multi_orchestrator_agent: Agent[AgentDeps, MultiSheetStrategy] = Agent(
    OpenAIChatModel(CDO_MODEL, provider=openrouter_provider),
    deps_type=AgentDeps,
    output_type=MultiSheetStrategy,
    system_prompt=load_prompt("cdo_multi_sheet"),
    model_settings=HEAVY_MODEL,
)

# Agent 5: Formatter
formatter_agent: Agent[None, PresentationSpec] = Agent(
    OpenAIChatModel(FORMATTER_MODEL, provider=openrouter_provider),
    output_type=PresentationSpec,
    retries=5,
    system_prompt=load_prompt("formatter_agent"), 
    model_settings=LIGHT_GPT,
)

# ---------------------------------------------------------------------------
# Analyst Tools
# ---------------------------------------------------------------------------

@analyst_agent.tool
async def execute_python_analysis(ctx: RunContext[AnalystDeps], code: str) -> str:
    """
    Executes Python (Pandas/NumPy) on live DataFrames to extract numeric facts.
    """
    # Fix for "OptionalMemberAccess": Explicitly check if executor is None
    if ctx.deps.executor is None:
        return "ERROR: No execution backend (executor) is configured in dependencies."

    dfs = ctx.deps.dfs

    # Ensure namespace is initialized with base libraries
    if not ctx.deps.namespace:
        ctx.deps.namespace.update({"dfs": dfs, "pd": pd, "np": np})
    
    try:
        # Now Pyright knows executor is not None here
        result = await ctx.deps.executor.run(code, ctx.deps.namespace)
        
        logfire.info(
            "execute_python_analysis",
            code_preview=code[:200],
            available_dfs=list(dfs.keys())
        )
        return result

    except Exception as exc:
        error_msg = f"ERROR: {type(exc).__name__}: {exc}"
        logfire.error("python_execution_failed", error=error_msg)
        return error_msg

# ---------------------------------------------------------------------------
# Dynamic Context Injection
# ---------------------------------------------------------------------------

@multi_orchestrator_agent.system_prompt
async def inject_memory_context(ctx: RunContext[AgentDeps]) -> str:
    """Injects historical join and quality data from AgentDeps memory store."""
    return ctx.deps.memory_store.to_cdo_context_block()
