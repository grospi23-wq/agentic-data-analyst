"""
agent_foundry.py
================
Scaffolding pipeline for generating and validating data analysis modules.

This script runs an Architect → Sandbox → QA feedback loop that iteratively refines
generated code until tests pass and QA approves. Successful artifacts are persisted
to `generated_modules/` for use by the notebook-driven Phase 3 pipeline.

Run this file directly to regenerate artifacts in generated_modules/:
    python agent_foundry.py
"""

import os
import sys
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import List, Literal, Optional

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# --- Environment Setup ---

load_dotenv(override=True)

# Silence noisy Pydantic logs while keeping Logfire visible
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("logfire").setLevel(logging.INFO)
logfire.configure()
logfire.instrument_pydantic(record="off")

# Destination directory for all agent-generated artifacts
OUTPUT_DIR = Path("generated_modules")
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Prompt Loader ---

def load_prompt(name: str) -> str:
    """Load a system prompt from the prompts/ directory by filename stem."""
    return Path(f"prompts/{name}.txt").read_text(encoding="utf-8").strip()


# --- Pydantic Models ---

class GeneratedModule(BaseModel):
    """Blueprint returned by the Architect agent after code generation."""
    module_name: str = Field(description="The library filename, e.g. 'data_profiler.py'")
    library_code: str = Field(description="The production-ready library code")
    test_file_name: str = Field(description="The test filename, e.g. 'test_data_profiler.py'")
    test_code: str = Field(description="Pytest code that validates the library functions")
    validation_summary: str = Field(description="A brief explanation of the design decisions")


class QAValidationIssue(BaseModel):
    """A single issue identified by the QA agent."""
    severity: Literal["low", "medium", "high"]
    location: str = Field(description="Function or line where the issue exists")
    description: str
    suggested_fix: Optional[str] = None


class QAFeedback(BaseModel):
    """Structured feedback returned by the QA agent (decoupled from Architect)."""
    is_passed: bool
    issues: List[QAValidationIssue] = Field(default_factory=list)
    summary: str


# --- Sandbox Runner ---

def run_in_local_sandbox(file_to_run: str) -> str:
    """Run a pytest file in an isolated subprocess and return the combined output."""
    print(f"🧪 Running {file_to_run} in Local Sandbox...")
    try:
        # Include both project root and generated_modules/ so pytest can find all imports
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([os.getcwd(), str(OUTPUT_DIR.resolve())])

        result = subprocess.run(
            [sys.executable, "-m", "pytest", file_to_run],
            capture_output=True,
            text=True,
            env=env,
        )
        return f"{result.stdout}\n{result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"


# --- Agent Definitions ---

openrouter_provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Architect: generates library code and matching pytest suite
architect_agent = Agent(
    OpenAIChatModel("anthropic/claude-3.5-sonnet", provider=openrouter_provider),
    output_type=GeneratedModule,
    system_prompt=load_prompt("architect"),
)

# QA Engineer: reviews code + pytest report, returns structured feedback only
qa_agent = Agent(
    OpenAIChatModel("openai/gpt-4o-mini", provider=openrouter_provider),
    output_type=QAFeedback,
    system_prompt=load_prompt("qa_engineer"),
)


# --- Production Line ---

async def run_production_line(task_description: str, max_retries: int = 3) -> GeneratedModule | None:
    """
    Orchestrate the Architect → Sandbox → QA feedback loop.

    Persists successful artifacts to OUTPUT_DIR (generated_modules/).

    Args:
        task_description: Natural-language task specification for the Architect agent.
        max_retries: Maximum refinement attempts before returning failure.

    Returns:
        The validated `GeneratedModule` on success, or `None` if retries are exhausted.
    """
    try:
        with logfire.span("Production Line: {task}", task=task_description):

            # Initial generation phase
            with logfire.span("Initial Architecting"):
                try:
                    arch_res = await architect_agent.run(f"Task: {task_description}")
                    current_module = arch_res.output
                except asyncio.CancelledError:
                    logfire.warn("Architect phase cancelled")
                    raise

            if not current_module:
                logfire.error("Architect failed to produce a module")
                return None

            last_qa_feedback: QAFeedback | None = None

            # Iterative self-healing loop: refine until tests pass and QA approves
            for attempt in range(max_retries):
                attempt_num = attempt + 1
                with logfire.span(f"Attempt {attempt_num}") as span:
                    try:
                        # Step 1: Persist generated files to generated_modules/
                        with logfire.span("File Persistence"):
                            module_path = OUTPUT_DIR / current_module.module_name
                            test_path = OUTPUT_DIR / current_module.test_file_name
                            module_path.write_text(current_module.library_code, encoding="utf-8")
                            test_path.write_text(current_module.test_code, encoding="utf-8")

                        # Step 2: Run pytest against the generated test file
                        with logfire.span("Sandbox Validation"):
                            pytest_report = run_in_local_sandbox(str(test_path))
                            pytest_passed = (
                                "passed" in pytest_report.lower()
                                and "failed" not in pytest_report.lower()
                            )

                        # Step 3: Collect structured QA feedback
                        with logfire.span("QA Criticism"):
                            qa_input = (
                                f"Review this code and the sandbox report.\n"
                                f"Code:\n{current_module.model_dump_json()}\n"
                                f"Pytest Results:\n{pytest_report}"
                            )
                            qa_res = await qa_agent.run(qa_input)
                            last_qa_feedback = qa_res.output

                        # Step 4: Both pytest and QA must pass to accept the module
                        if pytest_passed and last_qa_feedback.is_passed:
                            span.set_attribute("status", "success")
                            print(f"🏆 SUCCESS on attempt {attempt_num}!")
                            return current_module

                        # Step 5: Feed all failure signals back to the Architect for refinement
                        span.set_attribute("status", "refining")
                        print(
                            f"❌ Attempt {attempt_num} needs refinement. "
                            f"QA says: {last_qa_feedback.summary}"
                        )

                        with logfire.span("Architect Refinement"):
                            refine_input = (
                                f"Your previous code failed validation.\n"
                                f"QA Feedback: {last_qa_feedback.model_dump_json()}\n"
                                f"Pytest Report: {pytest_report}\n"
                                f"Please provide a corrected version of the module."
                            )
                            arch_res = await architect_agent.run(refine_input)
                            current_module = arch_res.output

                    except asyncio.CancelledError:
                        logfire.warn("Iteration cancelled", attempt=attempt_num)
                        raise

            logfire.error("Max retries reached without success", task=task_description)
            return None

    except asyncio.CancelledError:
        logfire.error("Production line cancelled")
        print("⚠️ Task cancelled")
        raise


# --- Task Definitions ---

# Task 1: Statistical Profiling Library
profiling_task = """
Create a professional Data Profiling & Statistical Library using Pandas and Pydantic.

Requirements:
1. Define a Pydantic model 'OutlierReport' with fields: column (str), outlier_count (int),
   outlier_pct (float), q1 (float), q3 (float), sample_values (list[float]),
   and severity (Literal["low", "medium", "high"]).

2. Implement 'detect_outliers(df, column)': It must return an instance of 'OutlierReport'.
   - Calculate outliers using the IQR method.
   - Set severity to 'high' if outlier_pct > 10%, 'medium' if > 5%, otherwise 'low'.

3. Implement 'analyze_missing_data(df)': It must return a 'MissingDataReport' model
   containing total_missing, column_stats (dict with count/pct per column),
   and critical_columns (list of columns with >50% missing).

Focus on strict type safety and high-quality docstrings.
"""

# Task 2: Discovery & Schema Mapping Library
discovery_task = """
Create a 'Discovery & Schema Mapping' module for our Data Agency.
The module must include:
1. run_dataset_discovery(file_path): Support .xlsx and .csv. Return GlobalDiscoveryMap (column names, types, counts, relationships).
2. get_semantic_sample(df, n=5): Provide a meaningful preview of the data.
"""


# --- Entry Point ---

if __name__ == "__main__":
    print("🏭 Agent Foundry: Starting production line...")
    asyncio.run(run_production_line(profiling_task))
