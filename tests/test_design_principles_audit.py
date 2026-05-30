from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "design_principles_audit.py"


def _load_audit_module():
    """Load the design-principles audit script as a test module."""
    spec = importlib.util.spec_from_file_location("design_principles_audit", AUDIT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_design_principles_audit_finds_core_smells(tmp_path: Path) -> None:
    """The audit reports representative DRY, KISS, YAGNI, and SOLID candidates."""
    source = """
def _unused_private_helper():
    value = 1
    return value


def duplicate_one(a):
    total = 0
    for value in a:
        total += value
    total *= 1
    return total


def duplicate_two(b):
    result = 0
    for item in b:
        result += item
    result *= 1
    return result


def many_parameters(a, b, c, d):
    return a + b + c + d


def branchy(flag):
    if flag:
        if flag:
            if flag:
                return 1
    return 0
"""
    package = tmp_path / "src" / "stability_radius"
    package.mkdir(parents=True)
    target = package / "sample.py"
    target.write_text(source, encoding="utf-8")

    audit = _load_audit_module()
    thresholds = audit.AuditThresholds(
        max_function_lines=20,
        max_branch_nodes=2,
        max_nested_blocks=2,
        max_parameters=3,
        min_duplicate_lines=3,
    )
    findings = audit.audit_repository(root=tmp_path, thresholds=thresholds)
    codes = {item.code for item in findings}

    assert "duplicate-function-body" in codes
    assert "too-many-parameters" in codes
    assert "function-too-nested" in codes
    assert "private-definition-unreferenced" in codes


def test_design_principles_audit_cli_json(tmp_path: Path) -> None:
    """The command-line audit emits machine-readable JSON."""
    package = tmp_path / "src" / "stability_radius"
    package.mkdir(parents=True)
    (package / "sample.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT_PATH),
            "--root",
            str(tmp_path),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["root"] == str(tmp_path.resolve())
    assert isinstance(payload["findings"], list)
