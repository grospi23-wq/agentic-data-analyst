import asyncio
import argparse
import logging
import platform
import subprocess
import logfire
from dotenv import load_dotenv
from service_layer import run_analysis_pipeline

async def main() -> None:
    load_dotenv(override=True)
    logging.getLogger("pydantic").setLevel(logging.ERROR)
    logfire.configure()

    parser = argparse.ArgumentParser(description="Agentic Data Analyst — Executive Analysis.")
    # Generic 'source' instead of 'file_path'
    parser.add_argument("source", help="Path to dataset (.xlsx, .csv) OR Database URL (sqlite://...)")
    parser.add_argument("--sheet", help="Target sheet name")
    args = parser.parse_args()

    # Detect if it's a SQL Connection String
    is_sql = any(args.source.startswith(p) for p in ["sqlite://", "postgresql://", "mysql://"])
    
    print(f"\n🚀 Starting mission for: {args.source}")
    result = await run_analysis_pipeline(args.source, target_sheet=args.sheet, is_sql=is_sql)

    if result:
        print(f"\n✅ Mission complete! Score: {result.get('critic_score', 0.0)}/1.0")
        if result.get("pptx_path"):
            print(f"Presentation saved: {result['pptx_path']}")
    else:
        print("\n❌ Mission failed.")

if __name__ == "__main__":
    asyncio.run(main())