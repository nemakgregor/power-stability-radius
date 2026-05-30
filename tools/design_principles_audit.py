from __future__ import annotations

"""
Heuristic audit for DRY, KISS, YAGNI, and SOLID design signals.

The script intentionally reports candidates instead of rewriting code. Design
principles are context-sensitive; the correct repair is a small reviewed patch
plus tests, not a blind mechanical edit.
"""

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCAN_ROOTS = ("src/stability_radius", "entry_points", "tests")
IGNORED_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.IfExp,
    ast.ExceptHandler,
)


@dataclass(frozen=True)
class AuditThresholds:
    """Thresholds for design-principle audit candidates."""

    max_function_lines: int = 90
    max_branch_nodes: int = 18
    max_nested_blocks: int = 5
    max_parameters: int = 8
    max_public_methods: int = 16
    max_module_definitions: int = 55
    min_duplicate_lines: int = 8


@dataclass(frozen=True)
class Finding:
    """One design-principle audit finding."""

    principle: str
    code: str
    path: str
    line: int
    message: str


class _BodyNormalizer(ast.NodeTransformer):
    """Normalize function bodies before duplicate-body comparison."""

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
        """Ignore literal values in structural duplicate detection."""
        if isinstance(node.value, str):
            node.value = "<str>"
        elif isinstance(node.value, (int, float, complex)):
            node.value = 0
        return node


def iter_python_files(
    root: Path, scan_roots: Iterable[str] = DEFAULT_SCAN_ROOTS
) -> list[Path]:
    """Return Python files under the configured repository scan roots."""
    files: list[Path] = []
    for rel_root in scan_roots:
        target = root / rel_root
        if target.is_file() and target.suffix == ".py":
            files.append(target)
            continue
        if not target.exists():
            continue
        for path in target.rglob("*.py"):
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            files.append(path)
    return sorted(set(files))


def _relative(path: Path, root: Path) -> str:
    """Return a stable POSIX-style path for audit output."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _node_line_span(node: ast.AST) -> int:
    """Return the source-line span for an AST node."""
    start = int(getattr(node, "lineno", 0) or 0)
    end = int(getattr(node, "end_lineno", start) or start)
    return max(1, end - start + 1)


def _function_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count explicit function parameters, including keyword-only parameters."""
    args = node.args
    return (
        len(args.posonlyargs)
        + len(args.args)
        + len(args.kwonlyargs)
        + (1 if args.vararg else 0)
        + (1 if args.kwarg else 0)
    )


def _branch_count(node: ast.AST) -> int:
    """Count branch-like nodes inside a function or method."""
    return sum(1 for child in ast.walk(node) if isinstance(child, BRANCH_NODES))


def _max_nested_blocks(node: ast.AST) -> int:
    """Measure maximum nesting depth for block-like statements."""

    def visit(current: ast.AST, depth: int) -> int:
        next_depth = depth + 1 if isinstance(current, BRANCH_NODES) else depth
        best = next_depth
        for child in ast.iter_child_nodes(current):
            best = max(best, visit(child, next_depth))
        return best

    return visit(node, 0)


def _body_without_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    """Return a function body without the leading docstring statement."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _normalized_body_digest(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a stable digest for a normalized function body."""
    normalizer = _BodyNormalizer()
    module = ast.Module(body=_body_without_docstring(node), type_ignores=[])
    normalized = ast.fix_missing_locations(normalizer.visit(module))
    dump = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _collect_text_references(files: list[Path]) -> dict[str, int]:
    """Count rough textual references to private top-level names."""
    counts: dict[str, int] = {}
    identifier = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in identifier.findall(text):
            if token.startswith("_") and not token.startswith("__"):
                counts[token] = counts.get(token, 0) + 1
    return counts


def audit_files(
    *,
    root: Path,
    files: list[Path],
    thresholds: AuditThresholds,
) -> list[Finding]:
    """Audit Python files and return design-principle candidate findings."""
    findings: list[Finding] = []
    duplicate_bodies: dict[str, tuple[str, int, int]] = {}
    text_refs = _collect_text_references(files)

    for path in files:
        rel = _relative(path, root)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(
                Finding(
                    principle="KISS",
                    code="syntax-error",
                    path=rel,
                    line=int(exc.lineno or 1),
                    message=str(exc),
                )
            )
            continue

        top_defs = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if len(top_defs) > thresholds.max_module_definitions:
            findings.append(
                Finding(
                    principle="SOLID",
                    code="module-too-many-definitions",
                    path=rel,
                    line=1,
                    message=(
                        f"module has {len(top_defs)} top-level definitions; "
                        "check single-responsibility boundaries"
                    ),
                )
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = _node_line_span(node)
                branches = _branch_count(node)
                nesting = _max_nested_blocks(node)
                params = _function_parameters(node)

                if span > thresholds.max_function_lines:
                    findings.append(
                        Finding(
                            principle="KISS",
                            code="function-too-long",
                            path=rel,
                            line=int(node.lineno),
                            message=f"{node.name} spans {span} lines",
                        )
                    )
                if branches > thresholds.max_branch_nodes:
                    findings.append(
                        Finding(
                            principle="KISS",
                            code="function-too-branchy",
                            path=rel,
                            line=int(node.lineno),
                            message=f"{node.name} has {branches} branch-like nodes",
                        )
                    )
                if nesting > thresholds.max_nested_blocks:
                    findings.append(
                        Finding(
                            principle="KISS",
                            code="function-too-nested",
                            path=rel,
                            line=int(node.lineno),
                            message=f"{node.name} has nesting depth {nesting}",
                        )
                    )
                if params > thresholds.max_parameters:
                    findings.append(
                        Finding(
                            principle="SOLID",
                            code="too-many-parameters",
                            path=rel,
                            line=int(node.lineno),
                            message=f"{node.name} has {params} parameters",
                        )
                    )

                body = _body_without_docstring(node)
                if len(body) >= 4 and span >= thresholds.min_duplicate_lines:
                    digest = _normalized_body_digest(node)
                    current = (rel, int(node.lineno), span)
                    if digest in duplicate_bodies:
                        prev_path, prev_line, _prev_span = duplicate_bodies[digest]
                        findings.append(
                            Finding(
                                principle="DRY",
                                code="duplicate-function-body",
                                path=rel,
                                line=int(node.lineno),
                                message=(
                                    f"{node.name} duplicates function body at "
                                    f"{prev_path}:{prev_line}"
                                ),
                            )
                        )
                    else:
                        duplicate_bodies[digest] = current

                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("_")
                    and not node.name.startswith("__")
                    and text_refs.get(node.name, 0) <= 1
                ):
                    findings.append(
                        Finding(
                            principle="YAGNI",
                            code="private-definition-unreferenced",
                            path=rel,
                            line=int(node.lineno),
                            message=(
                                f"{node.name} is not referenced by name in scanned files; "
                                "verify before keeping it"
                            ),
                        )
                    )

            elif isinstance(node, ast.ClassDef):
                public_methods = [
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                ]
                if len(public_methods) > thresholds.max_public_methods:
                    findings.append(
                        Finding(
                            principle="SOLID",
                            code="class-too-many-public-methods",
                            path=rel,
                            line=int(node.lineno),
                            message=(
                                f"{node.name} has {len(public_methods)} public methods; "
                                "check responsibility split"
                            ),
                        )
                    )

    return sorted(findings, key=lambda item: (item.path, item.line, item.code))


def audit_repository(root: Path, thresholds: AuditThresholds) -> list[Finding]:
    """Audit the default repository Python surfaces."""
    files = iter_python_files(root)
    return audit_files(root=root, files=files, thresholds=thresholds)


def _format_text(findings: list[Finding]) -> str:
    """Format findings for terminal output."""
    if not findings:
        return "No DRY/KISS/YAGNI/SOLID candidates found."

    lines = ["DRY/KISS/YAGNI/SOLID audit candidates:"]
    for item in findings:
        lines.append(
            f"{item.path}:{item.line}: {item.principle} {item.code}: {item.message}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Audit Python code for DRY, KISS, YAGNI, and SOLID candidates."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--max-function-lines", type=int, default=90)
    parser.add_argument("--max-branch-nodes", type=int, default=18)
    parser.add_argument("--max-nested-blocks", type=int, default=5)
    parser.add_argument("--max-parameters", type=int, default=8)
    parser.add_argument("--max-public-methods", type=int, default=16)
    parser.add_argument("--max-module-definitions", type=int, default=55)
    parser.add_argument("--min-duplicate-lines", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the design-principles audit CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    thresholds = AuditThresholds(
        max_function_lines=int(args.max_function_lines),
        max_branch_nodes=int(args.max_branch_nodes),
        max_nested_blocks=int(args.max_nested_blocks),
        max_parameters=int(args.max_parameters),
        max_public_methods=int(args.max_public_methods),
        max_module_definitions=int(args.max_module_definitions),
        min_duplicate_lines=int(args.min_duplicate_lines),
    )
    findings = audit_repository(root=args.root.resolve(), thresholds=thresholds)

    if args.format == "json":
        payload: dict[str, Any] = {
            "root": str(args.root.resolve()),
            "thresholds": asdict(thresholds),
            "findings": [asdict(item) for item in findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text(findings))

    return 1 if args.fail_on_findings and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
