from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is importable when tests are run without installing the package.
#
# IMPORTANT:
# We do NOT add repository root to sys.path. Adding repo root silently re-enables
# direct source imports and hides packaging mistakes.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if not SRC.is_dir():
    raise RuntimeError(f"Expected src/ directory at: {SRC}")

src_str = str(SRC)
if src_str not in sys.path:
    sys.path.insert(0, src_str)
