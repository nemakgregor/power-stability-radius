from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (ROOT / "src" / "stability_radius", ROOT / "entry_points")
TEXT_ROOTS = (
    ROOT / "src" / "stability_radius",
    ROOT / "entry_points",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / ".codex" / "docs",
)
TEXT_FILES = (ROOT / "UNITS_CONTRACT.md",)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml"}
DISALLOWED_DESIGN_TERMS = (
    "fall" + "back",
    "back" + "ward",
    "leg" + "acy",
    "wrap" + "per",
    "spag" + "hetti",
)


def _iter_python_files() -> list[Path]:
    """Return Python files that belong to the application surface."""
    files: list[Path] = []
    for root in PYTHON_ROOTS:
        files.extend(
            path for path in root.rglob("*.py") if "__pycache__" not in path.parts
        )
    return sorted(files)


def _iter_text_files() -> list[Path]:
    """Return text files covered by repository quality wording checks."""
    files: list[Path] = []
    for root in TEXT_ROOTS:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and "__pycache__" not in path.parts
        )
    files.extend(path for path in TEXT_FILES if path.exists())
    return sorted(set(files))


def test_all_application_functions_have_docstrings() -> None:
    """Every function in application Python code must carry a docstring."""
    missing: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if ast.get_docstring(node) is None:
                rel = path.relative_to(ROOT)
                missing.append(f"{rel}:{node.lineno}:{node.name}")

    assert missing == []


def test_quality_policy_words_do_not_reenter_project_text() -> None:
    """Reject banned design-policy terms in code and docs."""
    hits: list[str] = []
    for path in _iter_text_files():
        rel_path = str(path.relative_to(ROOT)).lower()
        text = path.read_text(encoding="utf-8").lower()
        for term in DISALLOWED_DESIGN_TERMS:
            if term in rel_path:
                hits.append(f"{path.relative_to(ROOT)}:{term}:path")
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")

    assert hits == []
