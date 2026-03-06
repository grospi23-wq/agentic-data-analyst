import asyncio
import logfire
from missions import execute_sql_mission

# Optional: Configure logfire to see the live trace
logfire.configure()

async def test_run():
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
        print("\n❌ TEST FAILED - Check logs for errors.")

if __name__ == "__main__":
    asyncio.run(test_run())