import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Allow running the example without installing the package:
# `python examples/script_demo.py` or `poetry run python examples/script_demo.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))

from stability_radius.dc.dc_model import build_dc_matrices
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.l2 import compute_l2_radius
from stability_radius.statistics.table import print_radius_summary, print_results_table
from stability_radius.utils import setup_logging
from stability_radius.utils.download import download_ieee_case

logger = logging.getLogger(__name__)

_CONFIG_DIR = (_PROJECT_ROOT / "config").resolve()


def _infer_case_number_from_path(path: str, default: int = 30) -> int:
    """
    Infer IEEE/MATPOWER case number from a filename like 'ieee30.m' or 'case118.m'.

    Parameters
    ----------
    path:
        Input file path.
    default:
        Returned when inference fails.

    Returns
    -------
    int
        Parsed case number or `default`.
    """
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else default


def _run_tests_or_exit(project_root: Path) -> None:
    """
    Run repository tests before executing the demo.

    This matches the current demo behavior. To skip (e.g., for speed), set:
        SR_SKIP_TESTS=1
    """
    if os.environ.get("SR_SKIP_TESTS", "").strip() == "1":
        return

    cmd = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(cmd, cwd=str(project_root), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _resolve_under_project_root(p: str) -> str:
    """Resolve a potentially-relative path under the repository root."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


@hydra.main(config_path=str(_CONFIG_DIR), config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    # Configure logs and create runs/<timestamp> directory
    run_dir = setup_logging(cfg)

    # 1) Download case file if it doesn't exist
    case_number = _infer_case_number_from_path(cfg.paths.input, default=30)
    target_path = _resolve_under_project_root(str(cfg.paths.input))
    input_path = download_ieee_case(case_number=case_number, target_path=target_path)

    # 2) Load network (with fallback .m parser if needed)
    net = load_network(input_path)

    # 3) Build DC model matrices (PTDF-like sensitivities)
    H_full, _ = build_dc_matrices(net, slack_bus=int(cfg.settings.slack_bus))

    # 4) Compute radii
    radii_results = compute_l2_radius(net, H_full)

    # 5) Print results table + summary to stdout
    print_results_table(radii_results)
    print_radius_summary(radii_results)

    # 6) Save results
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(radii_results, f, indent=4, ensure_ascii=False)

    logger.info("Done. Results written to: %s", results_path)


if __name__ == "__main__":
    _run_tests_or_exit(_PROJECT_ROOT)
    main()
