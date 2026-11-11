import asyncio
import logfire
import pytest
from missions import execute_sql_mission

pytestmark = pytest.mark.integration

# Optional: Configure logfire to see the live trace
logfire.configure()

@pytest.mark.asyncio
async def test_sql_mission_smoke():
    print("🚀 Starting SQL Mission Test on 'analyst_memory.db'...")
    
    # We pass the relative path to our SQLite file
    result = await execute_sql_mission("analyst_memory.db")
    
    if result:
        print("\n" + "━"*40)
        print("✅ TEST SUCCESSFUL!")
        print(f"📊 Critic Score: {result.get('critic_score')}")
        print(f"💾 PPTX Created: {result.get('pptx_path')}")
        print("━"*40)
    else:
        pytest.fail("SQL mission did not return a result. Check the model provider configuration.")

if __name__ == "__main__":
    asyncio.run(test_sql_mission_smoke())
