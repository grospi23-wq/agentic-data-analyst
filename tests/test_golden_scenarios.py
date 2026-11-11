import pytest
from missions import execute_analysis_mission

pytestmark = pytest.mark.integration

# Define ranges instead of absolute numbers (Models are stochastic)
SCENARIOS = [
    {
        "id": "pareto_sales",
        "file": "tests/golden/datasets/sales_pareto.csv",
        "min_score": 0.75,
        "keywords": ["pareto", "80/20", "concentration"]
    },
    {
        "id": "null_limitation",
        "file": "tests/golden/datasets/high_nulls.csv",
        "min_score": 0.70,
        "keywords": ["limitation", "missing data", "null"]
    }
]

@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
@pytest.mark.asyncio
async def test_golden_pipeline(scenario):
    """
    Runs the full analysis pipeline against a known 'Golden' dataset.
    Checks if the Critic score and the narrative meet the expected quality.
    """
    # Using our fixed mission logic
    result = await execute_analysis_mission(scenario["file"])

    assert result is not None, "Mission failed to return a result"

    score = result.get("critic_score", 0)
    narrative = result.get("narrative", "").lower()

    # 1. Check Score Range
    assert score >= scenario["min_score"], f"Score {score} is too low for {scenario['id']}"

    # 2. Check for Hallucinations vs. Facts
    for word in scenario.get("keywords", []):
        assert word in narrative, f"The narrative missed the key insight: {word}"
