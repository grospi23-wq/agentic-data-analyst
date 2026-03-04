"""
conftest.py
-----------
Root test configuration.

Adds the project root, lib/, and generated_modules/ to sys.path so all
modules are importable without installation. This also fixes the legacy
flat imports used in the existing test_data_discovery_lib.py and
test_data_profiler.py test files.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Project root — schema, agents, missions, service_layer, etc.
sys.path.insert(0, str(ROOT))

# lib/ sub-package — allows "from data_discovery_lib import ..." in legacy tests
sys.path.insert(0, str(ROOT / "lib"))

# generated_modules/ — allows "from data_profiler import ..." in legacy tests
sys.path.insert(0, str(ROOT / "generated_modules"))
