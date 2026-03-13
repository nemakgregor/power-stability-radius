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
