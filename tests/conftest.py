import sys
from pathlib import Path

# Ensure `src/` is importable when tests are run without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
