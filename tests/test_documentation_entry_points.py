from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_entry_point_reference_mentions_every_entry_point_module() -> None:
    entry_points_dir = ROOT / "entry_points"
    content = _read(ROOT / "docs" / "entry_points.md")

    entry_point_names = sorted(
        path.name
        for path in entry_points_dir.glob("*.py")
        if path.name != "__init__.py"
    )

    for name in entry_point_names:
        assert f"`entry_points/{name}`" in content


def test_entry_point_reference_contains_structured_cards() -> None:
    content = _read(ROOT / "docs" / "entry_points.md")

    # One card per runnable module.
    assert content.count("**What It Does**") >= 12
    assert content.count("**Inputs**") >= 12
    assert content.count("**Outputs**") >= 12
    assert content.count("**Example Invocation**") >= 12
    assert "`compute`" in content
    assert "`monte-carlo`" in content
    assert "`report`" in content
    assert "`table`" in content


def test_repository_text_files_do_not_use_cyrillic() -> None:
    cyrillic_codepoints = {0x0401, 0x0451}
    text_suffixes = {".md", ".py", ".toml", ".yml", ".yaml"}
    roots = [
        ROOT / "docs",
        ROOT / "entry_points",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "conf",
        ROOT / "experiments",
        ROOT / ".github",
    ]
    single_files = [
        ROOT / "README.md",
        ROOT / "UNITS_CONTRACT.md",
        ROOT / "pyproject.toml",
    ]

    def _contains_cyrillic(text: str) -> bool:
        for ch in text:
            codepoint = ord(ch)
            if 0x0410 <= codepoint <= 0x044F or codepoint in cyrillic_codepoints:
                return True
        return False

    offenders: list[str] = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            if _contains_cyrillic(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))

    for path in single_files:
        if _contains_cyrillic(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_docs_index_links_to_entry_points_and_testing_docs() -> None:
    content = _read(ROOT / "docs" / "index.md")

    assert "[entry_points.md](entry_points.md)" in content
    assert "[testing_and_ci.md](testing_and_ci.md)" in content


def test_readme_links_to_docs_index_and_primary_cli() -> None:
    content = _read(ROOT / "README.md")

    assert "[docs/index.md](docs/index.md)" in content
    assert (
        "`python entry_points/power_stability_radius.py --config conf/config.yaml <command>`"
        in content
    )
