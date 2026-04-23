from __future__ import annotations

import sys
from pathlib import Path


# Ensure tests can import workspace packages (e.g., src.*) regardless of
# how pytest resolves rootdir on CI runners.
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)