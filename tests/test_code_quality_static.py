from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (ROOT / "src" / "stability_radius", ROOT / "entry_points")
TEXT_ROOTS = (
    ROOT / "src" / "stability_radius",
    ROOT / "entry_points",
    ROOT / "tests",
    ROOT / "docs",
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


class _FunctionBodyNormalizer(ast.NodeTransformer):
    """Normalize function bodies before structural repetition checks."""

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Ignore local argument names and annotations."""
        node.arg = "_"
        node.annotation = None
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Ignore local variable names."""
        node.id = "_"
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Ignore attribute spelling while preserving access structure."""
        self.generic_visit(node)
        node.attr = "_"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Ignore literal spelling in repeated body detection."""
        if isinstance(node.value, str):
            node.value = "<str>"
        elif isinstance(node.value, (int, float, complex)):
            node.value = 0
        return node


def _normalized_body_digest(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a stable digest for a normalized function body."""
    normalizer = _FunctionBodyNormalizer()
    module = ast.Module(body=node.body, type_ignores=[])
    normalized = ast.fix_missing_locations(normalizer.visit(module))
    dump = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def test_application_function_bodies_are_not_copied_between_modules() -> None:
    """Reject repeated non-trivial function bodies across application code."""
    seen: dict[str, str] = {}
    repeats: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(node.body) < 4:
                continue
            digest = _normalized_body_digest(node)
            rel = path.relative_to(ROOT)
            location = f"{rel}:{node.lineno}:{node.name}"
            if digest in seen:
                repeats.append(f"{seen[digest]} == {location}")
            else:
                seen[digest] = location

    assert repeats == []


def test_entry_points_do_not_import_private_library_symbols() -> None:
    """Entry-point scripts must use the public library surface."""
    hits: list[str] = []
    entry_points_dir = ROOT / "entry_points"
    for path in sorted(entry_points_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("stability_radius"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    hits.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}.{alias.name}"
                    )

    assert hits == []


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
