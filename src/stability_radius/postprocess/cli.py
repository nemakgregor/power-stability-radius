from __future__ import annotations

"""Shared command-line helpers for post-processing plot modules."""

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from stability_radius.utils import create_module_output_dir, setup_output_dir_logging


def run_single_input_plot_cli(
    *,
    argv: Sequence[str] | None,
    description: str,
    input_flag: str,
    input_default: Path,
    input_help: str,
    module_name: str,
    plot_func: Callable[[Path, Path], None],
) -> int:
    """Parse a single-input plot command and run the supplied plot function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        input_flag,
        type=Path,
        default=input_default,
        help=input_help,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(""),
        help="Directory where plots are saved.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_dir = create_module_output_dir(
        module_name=module_name,
        requested_output_dir=args.output_dir,
    )
    setup_output_dir_logging(output_dir)
    input_dir = getattr(args, input_flag.removeprefix("--").replace("-", "_"))
    plot_func(input_dir, output_dir)
    return 0
