# Entry Points Reference

This file is the authoritative inventory of everything under `entry_points/`. Keep it updated whenever a runnable script is added, removed, renamed, or materially changes its inputs or outputs.

## Shared Rules

- The primary operational entry point is `entry_points/power_stability_radius.py`.
- Standalone scripts are used for experiments, paper figures, or focused post-processing tasks.
- Artifact directories default to `run_artifacts/`.
- The docs-as-code test suite checks that every `entry_points/*.py` module is listed here.

## Main CLI

### `entry_points/power_stability_radius.py`

Use this when you want the repository's supported operational interface.

```bash
python entry_points/power_stability_radius.py --config conf/config.yaml <command>
```

Subcommands:

| Command | Purpose | Main outputs |
|--------|---------|--------------|
| `compute` / `demo` | Run deterministic DC and/or AC radius computation for one case | `results.json`, tables, optional `h_vectors.npz` |
| `monte-carlo` | Verify an existing result with random perturbations | `monte_carlo_stats.json` |
| `report` | Build a multi-case Markdown verification report | `verification_report.md` |
| `table` | Reformat an existing `results.json` for terminal or file output | `results_table.txt` |

## Standalone Analysis and Experiment Scripts

| File | When to use it | Typical inputs | Main outputs |
|------|----------------|----------------|--------------|
| `entry_points/table.py` | Format a single `results.json` without using the main CLI | `results.json` | Console table, `results_table.txt` |
| `entry_points/metrics_analysis.py` | Compare AC radii against baseline and practical metrics | MATPOWER case, sigma values, Monte Carlo settings | Unified CSV tables, plots, metrics JSON |
| `entry_points/n1_stability_demo.py` | Run the dedicated three-regime N-1 demonstration workflow | MATPOWER case plus regime parameters | CSV tables, plots, summary text, debug log |
| `entry_points/run_pglib_sweep.py` | Run the multi-case DC vs AC sweep over PGLib-style cases | `experiments/configs/pglib_sweep.yaml` | Per-case JSON files, `summary.json`, comparison plots |
| `entry_points/run_sigma_radius.py` | Run the AC sigma-radius experiment around the average operating point | Experiment YAML, MATPOWER case, UC.jl or synthetic sigma source | `results.json`, `summary.json`, Table 2 CSV, plots, verification outputs |
| `entry_points/run_worst_case_verify.py` | Recheck analytical worst-case directions with nonlinear AC power flow | Prior sweep results or per-case result JSON plus h-vectors | Per-case worst-case JSON, validation checks, plots |
| `entry_points/run_scalability.py` | Measure repeated DC and AC runtime against case size | `experiments/configs/pglib_sweep.yaml`, repeat count | `scalability.json` |
| `entry_points/collect_results.py` | Aggregate experiment outputs into one CSV summary | A `run_artifacts/` tree | `all_results.csv` |
| `entry_points/plot_radius_distribution.py` | Plot DC and AC radius distributions across cases | Sweep result JSON files | `radius_distribution.pdf`, `radius_distribution.png` |
| `entry_points/plot_sigma_vs_time.py` | Plot sigma-radius summaries and optional scalability timings | Sigma-radius result JSON files, optional `scalability.json` | `sigma_radius_sorted.pdf`, `sigma_vs_time.pdf` |
| `entry_points/plot_worst_case_heatmap.py` | Plot verification error and violation heatmaps | `*_worst_case.json` files | `worst_case_heatmap.pdf`, `worst_case_scatter.pdf` |

## Which Entry Point to Pick

- Use `entry_points/power_stability_radius.py` for normal compute, verification, reporting, and table formatting workflows.
- Use `entry_points/run_pglib_sweep.py`, `entry_points/run_sigma_radius.py`, `entry_points/run_worst_case_verify.py`, and `entry_points/run_scalability.py` for paper-style experiments and aggregate studies.
- Use `entry_points/metrics_analysis.py` and `entry_points/n1_stability_demo.py` for dedicated analysis pipelines that have their own CLI surface and artifact conventions.
- Use `entry_points/collect_results.py`, `entry_points/plot_radius_distribution.py`, `entry_points/plot_sigma_vs_time.py`, and `entry_points/plot_worst_case_heatmap.py` after experiment runs when you want summaries or visualizations.

## Artifact Naming

The entry points use two artifact patterns:

- Main CLI commands create per-run directories such as `run_artifacts/compute/<timestamp>/`.
- Standalone scripts generally write to `run_artifacts/<module>/`, where `<module>` matches the module name passed to `create_module_output_dir(...)`.

That mapping is what lets the tests and CI keep the documentation aligned with the code.
