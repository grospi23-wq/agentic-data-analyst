import asyncio
import argparse
import logging
from pathlib import Path

import logfire
from dotenv import load_dotenv

from missions import execute_analysis_mission, execute_multi_sheet_mission


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

    is_csv = Path(args.file_path).suffix.lower() == ".csv"

    if args.sheet or is_csv:
        sheet = args.sheet or Path(args.file_path).stem
        result = await execute_analysis_mission(args.file_path, target_sheet=sheet)
    else:
        result = await execute_multi_sheet_mission(args.file_path)

    if result:
        score = result.get("critic_score", 0.0)
        pptx = result.get("pptx_path")
        print(f"\n{'═'*50}")
        print(f"Mission complete — Critic score: {score}/1.0")
        if pptx:
            print(f"Presentation: {pptx}")
        print(f"{'═'*50}")
    else:
        print("\n❌ Mission failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())
