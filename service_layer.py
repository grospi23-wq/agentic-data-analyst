"""
service_layer.py
----------------
Core application service with a Global File Resolver.
Scans Project, WSL, and Windows (Desktop/Documents/Downloads) dynamically.
"""

import os
import pandas as pd
import logfire
import getpass
from pathlib import Path
from missions import (
    execute_analysis_mission, 
    execute_multi_sheet_mission, 
    execute_sql_mission
)

# --- CONFIGURATION ---
MAX_FILE_SIZE_MB = 50.0

def _enforce_size_guard(path: Path) -> None:
    """Prevents OOM by blocking files larger than the threshold."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f}MB). Threshold is {MAX_FILE_SIZE_MB}MB.")

def _resolve_source_path(source: str) -> Path:
    """
    Scans every possible landing zone for the file.
    Covers: Project Root, Data folders, WSL/Windows Desktop, Documents, and Downloads.
    """
    linux_user = getpass.getuser()
    
    # 1. Base Project Paths
    search_dirs = [
        Path("."), 
        Path("data")
    ]

    # 2. WSL Home Directories
    home = Path.home()
    search_dirs.extend([home / "Downloads", home / "Desktop", home / "Documents"])

    # 3. Dynamic Windows Path Discovery
    win_users_base = Path("/mnt/c/Users")
    if win_users_base.exists():
        try:
            # Iterate through all Windows users to find where the file might be
            for user_dir in win_users_base.iterdir():
                if not user_dir.is_dir(): continue
                
                # Check all standard "Landing Zones" for each user found
                for folder_name in ["Downloads", "Desktop", "Documents"]:
                    candidate_dir = user_dir / folder_name
                    if candidate_dir.exists():
                        search_dirs.append(candidate_dir)
        except (PermissionError, FileNotFoundError):
            pass

    # 4. Search Loop: Exact Match & Case-Insensitive
    for folder in search_dirs:
        if not folder.exists(): continue
        
        # Exact match
        exact = folder / source
        if exact.exists(): return exact
        
        # Case-insensitive scan
        try:
            for item in folder.iterdir():
                if item.name.lower() == source.lower():
                    return item
        except (PermissionError, FileNotFoundError):
            continue

    # 5. Last Resort: Deep recursive search in current project directory
    for p in Path(".").rglob(source):
        return p

    raise FileNotFoundError(f"Could not find '{source}' in Project, Desktop, Documents, or Downloads (WSL/Windows).")

async def run_analysis_pipeline(source: str, target_sheet: str | None = None, is_sql: bool = False) -> dict | None:
    """
    Routes the analysis mission based on the resolved source.
    """
    # SQL ROUTE
    if is_sql:
        logfire.info("Routing to SQL mission: {url}", url=source)
        return await execute_sql_mission(source)

    # FILE ROUTE
    try:
        path = _resolve_source_path(source)
        path = path.resolve()
        _enforce_size_guard(path)
        
        print(f"📂 Resolved: {path}")
        logfire.info("File resolved", path=str(path))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    # Processing Logic
    is_csv = path.suffix.lower() == ".csv"
    if is_csv:
        return await execute_analysis_mission(str(path), target_sheet=path.stem)
    
    if target_sheet:
        return await execute_analysis_mission(str(path), target_sheet=target_sheet)

    # Excel Logic
    excel_file = pd.ExcelFile(path)
    sheet_names = excel_file.sheet_names
    
    if len(sheet_names) == 1:
        return await execute_analysis_mission(str(path), target_sheet=sheet_names[0])
    else:
        return await execute_multi_sheet_mission(str(path))