from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NON_ENTRY_HELPERS = {
    "collect_results.py",
    "plot_radius_distribution.py",
    "plot_sigma_vs_time.py",
    "plot_worst_case_heatmap.py",
    "table.py",
}


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
    entry_point_names = sorted(
        path.name
        for path in (ROOT / "entry_points").glob("*.py")
        if path.name != "__init__.py"
    )

    # One card per runnable module.
    assert content.count("**What It Does**") == len(entry_point_names)
    assert content.count("**Inputs**") == len(entry_point_names)
    assert content.count("**Outputs**") == len(entry_point_names)
    assert content.count("**Example Invocation**") == len(entry_point_names)
    assert "`compute`" in content
    assert "`monte-carlo`" in content
    assert "`report`" in content
    assert "`table`" in content


def test_entry_points_directory_contains_only_runnable_fronts() -> None:
    entry_point_names = {
        path.name
        for path in (ROOT / "entry_points").glob("*.py")
        if path.name != "__init__.py"
    }

    assert NON_ENTRY_HELPERS.isdisjoint(entry_point_names)
    for helper in NON_ENTRY_HELPERS:
        assert (ROOT / "src" / "stability_radius" / "postprocess" / helper).exists()


def test_main_entry_point_is_a_thin_application_wrapper() -> None:
    content = _read(ROOT / "entry_points" / "power_stability_radius.py")
    application_content = _read(
        ROOT / "src" / "stability_radius" / "application" / "cli.py"
    )

    assert "from stability_radius.application.cli import (" in content
    assert "argparse" not in content
    assert "from stability_radius.postprocess.table import (" in application_content
    assert (
        'logger = logging.getLogger("stability_radius.application.cli")'
        in application_content
    )


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


def test_architecture_docs_describe_application_and_domain_layers() -> None:
    repository_structure = _read(ROOT / "docs" / "repository_structure.md")
    architecture = _read(ROOT / "docs" / "architecture.md")
    readme = _read(ROOT / "README.md")

    assert "`application/`" in repository_structure
    assert "`domain/`" in repository_structure
    assert "`src/stability_radius/application/cli.py`" in architecture
    assert "`src/stability_radius/domain/reporting.py`" in architecture
    assert "`src/stability_radius/application/`" in readme
    assert "`src/stability_radius/domain/`" in readme
