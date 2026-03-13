# Execution Flow

This document describes the current runtime flow of the repository's main entry points. For a file-by-file inventory, see [entry_points.md](entry_points.md). For the repository map, see [repository_structure.md](repository_structure.md).

## Primary CLI: `entry_points/power_stability_radius.py`

The main CLI is the operational front door:

```bash
python entry_points/power_stability_radius.py --config conf/config.yaml <command>
```

`entry_points/power_stability_radius.py` is an interface wrapper. The actual CLI orchestration lives in `src/stability_radius/application/cli.py`.

Common startup path:

1. Pre-parse `--config` so YAML defaults are available before full CLI parsing.
2. Load the composed project config via `stability_radius.config.load_project_config(...)`.
3. Build the argparse parser with global logging, OPF, download, and command-specific flags.
4. Optionally run the repository test suite when `--run-tests 1`.
5. Dispatch to one of four command handlers: `compute`, `monte-carlo`, `report`, or `table`.
6. For `report`, convert YAML `report.cases` into typed `ReportCaseSpec` objects from `stability_radius.domain.reporting`.

## `compute`

`compute` is the main deterministic pipeline:

1. Create a per-run artifact directory via `setup_logging(...)`.
2. Persist the effective config, source config, and argv into the run directory.
3. Translate CLI flags into `DCExtensionsConfig` and `ACExtensionsConfig`.
4. Call `stability_radius.workflows.compute_results_for_case(...)`.
5. Write `results.json`, optional `h_vectors.npz`, and formatted tables.
6. Optionally export a copy of `results.json` to a user-requested location.

At a high level, `compute_results_for_case(...)` performs:

1. Load the MATPOWER case and resolve the slack bus.
2. Build the DC base point and compute selected DC radius variants.
3. Build the AC base point and compute selected AC radius variants.
4. Assemble per-line outputs plus `__meta__` provenance.

## `monte-carlo`

The `monte-carlo` command verifies a prior `results.json`:

1. Validate that both `--results` and `--input` were provided.
2. Create a run directory and write the effective config snapshot.
3. Call `stability_radius.verification.monte_carlo.run_monte_carlo_verification(...)`.
4. Serialize the result to `monte_carlo_stats.json`.
5. Print the same JSON payload to stdout.

## `report`

The `report` command generates a multi-case Markdown verification report:

1. Read `report.cases` from the YAML config.
2. Resolve each case's input case path and result path relative to the configured base directories.
3. Build `ReportCaseSpec` objects in the application layer, before calling verification code.
4. Call `stability_radius.verification.generate_report.generate_report_text(...)`.
5. Write the report both to the requested output path and to the run directory as `verification_report.md`.

## `table`

The `table` command reformats an existing `results.json`:

1. Load the results object.
2. Choose either flat or sectioned output.
3. Infer default columns when the user does not provide them.
4. Print the table and radius summary, then save `results_table.txt` to the run directory.

## Standalone Experiment Entry Points

Standalone scripts in `entry_points/` bypass the main CLI when they need dedicated workflows or paper-style outputs:

- `entry_points/run_pglib_sweep.py`: multi-case DC/AC sweep over PGLib cases
- `entry_points/run_sigma_radius.py`: AC sigma-radius experiment around an average operating point
- `entry_points/run_worst_case_verify.py`: nonlinear verification of analytical worst-case perturbations
- `entry_points/run_scalability.py`: repeated timing runs for DC and AC pipelines
- `entry_points/metrics_analysis.py`: compare radii against baseline and practical metrics
- `entry_points/n1_stability_demo.py`: three-regime N-1 demonstration pipeline

Each of these scripts creates artifacts under `run_artifacts/<module>/` unless an output directory is explicitly provided.

## Reusable Post-Processing Modules

Table formatting, CSV aggregation, and plotting helpers now live under `src/stability_radius/postprocess/`:

- `stability_radius.postprocess.table`
- `stability_radius.postprocess.collect_results`
- `stability_radius.postprocess.plot_radius_distribution`
- `stability_radius.postprocess.plot_sigma_vs_time`
- `stability_radius.postprocess.plot_worst_case_heatmap`

These modules are library-side helpers rather than primary repository entry points.

## Artifact Pattern

Artifact writing follows two patterns:

- Main CLI commands: `run_artifacts/<command>/<timestamp or run_name>/...`
- Standalone scripts: `run_artifacts/<module>/...` or a normalized requested directory under the same root

The logging and artifact helpers live in `src/stability_radius/utils/__init__.py`, which is why entry-point documentation and tests focus on the module name used in `create_module_output_dir(...)` or `setup_logging(...)`.
