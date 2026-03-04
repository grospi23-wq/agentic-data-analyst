"""
main.py
-------
Entry point for the Agentic Data Analyst. 
Handles CLI arguments, environment initialization, and OS interactions.
"""

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
    logging.getLogger("logfire").setLevel(logging.INFO)
    logfire.configure()
    logfire.instrument_pydantic(record="off")

    parser = argparse.ArgumentParser(
        description="Agentic Data Analyst — run an analysis mission on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py titanic.xlsx --sheet titanic_data  # single-sheet\n"
            "  python main.py bikes.xlsx                          # multi-sheet\n"
            "  python main.py data.csv                            # CSV (single-sheet)\n"
        ),
    )
    parser.add_argument("file_path", help="Path to the dataset (.xlsx or .csv)")
    parser.add_argument(
        "--sheet",
        metavar="SHEET_NAME",
        help="Target sheet name — forces single-sheet mode for .xlsx files",
    )
    args = parser.parse_args()

    # Delegate core logic to the Service Layer
    result = await run_analysis_pipeline(args.file_path, target_sheet=args.sheet)

    # Handle UI/OS Output
    if result:
        score = result.get("critic_score", 0.0)
        pptx = result.get("pptx_path")
        print(f"\n{'═'*50}")
        print(f"Mission complete — Critic score: {score}/1.0")
        
        if pptx:
            print(f"Presentation saved: {pptx}")
            try:
                print("🚀 Opening presentation in Windows...")
                if "microsoft" in platform.uname().release.lower():
                    # WSL environment
                    windows_friendly_path = pptx.replace('/', '\\')
                    subprocess.run(["explorer.exe", windows_friendly_path])
                elif platform.system() == "Windows":
                    # Native Windows
                    import os
                    os.startfile(pptx)
            except Exception as e:
                print(f"⚠️ Could not auto-open presentation: {e}")
                
        print(f"{'═'*50}")
    else:
        print("\n❌ Mission failed. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())