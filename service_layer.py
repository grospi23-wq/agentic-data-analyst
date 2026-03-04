"""
service_layer.py
----------------
The core application service that orchestrates the data analysis pipeline.
Handles the routing logic based on file type and sheet count.
"""

import os
import pandas as pd
import logfire
from pathlib import Path
from lib.path_utils import resolve_file_path
from missions import execute_analysis_mission, execute_multi_sheet_mission

# --- CONCURRENCY & SCALE GUARDS ---
MAX_FILE_SIZE_MB = 50.0

def _enforce_size_guard(path: Path) -> None:
    """
    Prevents Out-Of-Memory (OOM) crashes by blocking excessively large files.
    In a production environment, files above this threshold would be routed
    to a distributed background queue (e.g., Celery/Redis) with chunked processing.
    """
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f}MB). The current synchronous architecture "
            f"is capped at {MAX_FILE_SIZE_MB}MB to prevent memory exhaustion."
        )

async def run_analysis_pipeline(file_path: str, target_sheet: str | None = None) -> dict | None:
    """
    Orchestrates the analysis flow by determining the correct mission type.
    """
    path = resolve_file_path(file_path)
    
    # 1. Enforce Production Guards
    _enforce_size_guard(path)

    # 2. Routing Logic
    is_csv = path.suffix.lower() == ".csv"

    if is_csv:
        return await execute_analysis_mission(str(path), target_sheet=path.stem)
    
    if target_sheet:
        return await execute_analysis_mission(str(path), target_sheet=target_sheet)

    excel_file = pd.ExcelFile(path)
    sheet_names = excel_file.sheet_names
    
    if len(sheet_names) == 1:
        logfire.info("Auto-routing to single-sheet mission (1 sheet detected)")
        print(f"📄 Single sheet detected: '{sheet_names[0]}'. Using optimized analyst.")
        return await execute_analysis_mission(str(path), target_sheet=sheet_names[0])
    else:
        logfire.info("Routing to multi-sheet mission ({n} sheets detected)", n=len(sheet_names))
        return await execute_multi_sheet_mission(str(path))