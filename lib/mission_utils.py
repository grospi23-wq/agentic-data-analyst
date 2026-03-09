"""
mission_utils.py
----------------
UI and formatting utilities for mission orchestration.
Extracted from missions.py to keep orchestration logic separate from presentation.
"""

from schema import CriticReport


def format_rewrites(critic_report: CriticReport | None) -> str:
    """Converts structured Critic rewrites into a readable prompt string."""
    # Handle the None case safely to satisfy the type checker
    if not critic_report:
        return "Revise the report to address the findings."

    if critic_report.required_rewrites:
        lines = [
            f"- [{rw.insight_id}] ({rw.issue_type}): {rw.fix_instruction}"
            for rw in critic_report.required_rewrites
        ]
        return "\n".join(lines)

    return critic_report.revision_instructions or "Revise the report to address the Critic's findings."


def print_rejection_detail(
    critic_report: CriticReport,
    attempt: int,
    max_attempts: int,
) -> None:
    """Prints a structured summary of Critic rejections."""
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
