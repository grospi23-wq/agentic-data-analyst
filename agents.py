import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from schema import (
    AnalysisStrategy,
    AnalystOutput,
    CriticReport,
    MultiSheetStrategy,
    PresentationSpec,
)

load_dotenv(override=True)


def load_prompt(name: str) -> str:
    """Load a system prompt from the prompts/ directory by filename stem."""
    return Path(f"prompts/{name}.txt").read_text(encoding="utf-8").strip()


openrouter_provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Agent 1: Chief Data Officer — defines strategy and enforces ID-column exclusion
# claude-3.7-sonnet: strongest hypothesis reasoning and schema analysis
orchestrator_agent = Agent(
    OpenAIChatModel("anthropic/claude-3.7-sonnet", provider=openrouter_provider),
    output_type=AnalysisStrategy,
    system_prompt=load_prompt("cdo_strategy"),
)

# Agent 2: Senior Data Analyst — structured CoT + business-centric narrative
# gpt-4o: proven at long-form structured synthesis; kept as anchor model
analyst_agent = Agent(
    OpenAIChatModel("openai/gpt-4o", provider=openrouter_provider),
    output_type=AnalystOutput,
    system_prompt=load_prompt("analyst_synthesis"),
)

# Agent 3: Ruthless Auditor — validates narrative against raw profiler data
# gemini-2.0-flash: lives inside a retry loop, so speed and cost matter;
# cross-provider auditing also reduces Anthropic family bias in the critique
critic_agent = Agent(
    OpenAIChatModel("google/gemini-2.0-flash", provider=openrouter_provider),
    output_type=CriticReport,
    system_prompt=load_prompt("critic_auditor"),
)

# Agent 4: Multi-sheet CDO — plans joins and per-sheet focus across an entire workbook
# claude-3.7-sonnet: most complex planning task in the pipeline (join graph reasoning)
multi_orchestrator_agent = Agent(
    OpenAIChatModel("anthropic/claude-3.7-sonnet", provider=openrouter_provider),
    output_type=MultiSheetStrategy,
    system_prompt=load_prompt("cdo_multi_sheet"),
)

# Agent 5: Formatter — converts analyst output into a structured presentation spec
# gpt-4o-mini: pure JSON transformation task; no frontier model needed
formatter_agent = Agent(
    OpenAIChatModel("openai/gpt-4o-mini", provider=openrouter_provider),
    output_type=PresentationSpec,
    system_prompt=load_prompt("formatter_agent"),
)
