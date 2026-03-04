"""
prompt_builders.py
------------------
A dedicated factory for constructing LLM prompts. 
It encapsulates the string formatting logic, separating data and configuration 
from the raw text templates sent to the agents.
"""

import json
from typing import Dict, Any

def build_analyst_mission_prompt(
    template: str,
    hypothesis: str,
    questions: list[str],
    data_json: str,
    schema_context: str,
    analysis_mode: str = "separate",
    cross_sheet_hypothesis: str = "N/A"
) -> str:
    """
    Constructs the initial analyst mission prompt for Attempt 1.
    Injects the schema guardrail to prevent hallucinated columns.
    """
    questions_block = "\n".join(f"- {q}" for q in questions)
    
    formatted_body = template.format(
        hypothesis=hypothesis,
        questions=questions_block,
        data=data_json,
        analysis_mode=analysis_mode,
        cross_sheet_hypothesis=cross_sheet_hypothesis,
    )
    
    return (
        f"CRITICAL — VERIFIED SCHEMA (use these EXACT names, do not guess):\n"
        f"{schema_context}\n\n"
        f"{formatted_body}"
    )

def build_retry_prompt(
    template: str,
    revision_instructions: str,
    previous_narrative: str
) -> str:
    """
    Constructs the retry prompt based on Critic feedback.
    This is intentionally lean, sending only the required rewrites and the past attempt.
    """
    return template.format(
        revision_instructions=revision_instructions,
        previous_narrative=previous_narrative
    )

def build_self_validation_prompt(
    template: str,
    schema_context: str,
    chain_of_thought: list[str],
    draft_narrative: str
) -> str:
    """
    Constructs the self-reflection prompt where the Analyst critiques its own draft
    before sending it to the formal Critic.
    """
    cot_block = "\n".join(chain_of_thought)
    
    return (
        f"CRITICAL — VERIFIED SCHEMA (use these EXACT names, do not guess):\n"
        f"{schema_context}\n\n"
        + template.format(
            chain_of_thought=cot_block,
            draft_narrative=draft_narrative
        )
    )

def build_cdo_multisheet_prompt(
    pruned_map: Dict[str, Any],
    memory_context: str,
    lean_mode: bool
) -> str:
    """
    Constructs the input context for the CDO multi-sheet strategy agent,
    combining the current discovery map with historical memory.
    """
    return (
        f"DATASET STRUCTURE ({'Lean' if lean_mode else 'Full'} Context):\n"
        f"{json.dumps(pruned_map, indent=2)}\n\n"
        f"{memory_context}"
    )

def build_formatter_prompt(
    report_json: str,
    deck_title: str,
    critic_score: float,
    analysis_mode: str
) -> str:
    """
    Constructs the prompt for the Formatter agent to generate the PresentationSpec.
    """
    return (
        f"AnalystOutput:\n{report_json}\n\n"
        f"deck_title: {deck_title}\n"
        f"critic_score: {critic_score}\n"
        f"analysis_mode: {analysis_mode}\n"
    )