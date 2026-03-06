"""
Path resolution utilities.

This module provides a single entry point, `resolve_file_path`, used by Missions to
normalize user-provided paths and resolve them to an absolute `Path` in a WSL-friendly
way. The search strategy is intentionally pragmatic (project-relative first, then
common Windows mount locations) to reduce notebook friction.
"""

import os
import re
import getpass
from pathlib import Path
from sqlalchemy import create_engine, Engine

def resolve_file_path(path_input: str) -> Path:
    """
    Smart path resolver that bridges WSL and Windows file systems.
    Searches in CWD, project root, and common Windows locations (Downloads/Desktop).
    """
    # 1. Basic normalization (fix slashes)
    clean_path = path_input.replace("\\", "/").strip()
    path_obj = Path(clean_path)

    # 2. If it's absolute and exists, we are good
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj.resolve()

    # 3. Define search candidates.
    # Note: Windows mount locations are project-specific convenience defaults.
    user = os.getenv("WSL_USER", getpass.getuser())

    candidates = [
        Path.cwd() / path_obj,                          # Relative to where script is running
        Path(__file__).parent.parent / path_obj,        # Relative to project root
        Path(f"/mnt/c/Users/{user}/Downloads/{clean_path}"), # Your specific Windows Downloads
        Path(f"/mnt/c/Users/{user}/Desktop/{clean_path}"),   # Your specific Windows Desktop
    ]

    # 4. Check if file exists in any of these locations
    for candidate in candidates:
        if candidate.exists():
            print(f"✅ Found file at: {candidate}")  # Visible hint when resolving from fallbacks
            return candidate.resolve()

    # 5. Fallback: Return absolute path of CWD version (even if missing)
    return (Path.cwd() / path_obj).resolve()   

def resolve_sql_connection(path_input: str) -> tuple[Engine, str]:
    """
    Resolves a connection string or local SQLite file path into a SQLAlchemy Engine.
    Returns the Engine and a sanitized display path (passwords redacted).
    """
    if "://" in path_input:
        # It's a URL (e.g., postgresql://user:pass@host/db)
        engine = create_engine(path_input)
        
        # Redact password for logs using a regex
        display_path = re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", path_input)
        return engine, display_path
    
    # It's a local file path (SQLite)
    resolved_path = resolve_file_path(path_input)
    engine = create_engine(f"sqlite:///{resolved_path}")
    return engine, f"sqlite:///{resolved_path.name}"    