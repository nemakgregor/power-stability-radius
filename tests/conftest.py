from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root and `src/` are importable when tests are run without installing the package.
#
# Why both are needed:
# - library code lives under `src/`
# - verification helpers live under top-level `verification/`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

paths: list[str] = []
if SRC.is_dir():
    paths.append(str(SRC))
paths.append(str(ROOT))

# Deterministic ordering: src first, then repo root.
for p in reversed(paths):
    if p not in sys.path:
        sys.path.insert(0, p)
