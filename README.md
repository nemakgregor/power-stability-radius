# power-stability-radius

Compute simple per-line “stability radius” (robustness margins) for power grids using a DC sensitivity (PTDF-like) model:

- `radius_l2` (L2 ball)
- `radius_metric` (weighted/metric; in the default workflow equals `radius_l2`)
- `radius_sigma` + overload probability (Gaussian injections)
- `radius_nminus1` (effective N-1 via LODF; requires materializing `H_full`)

All workflows are launched via a **single entrypoint**:
`src/power_stability_radius.py`

Defaults are stored in an **OmegaConf/Hydra-style YAML config**:
`conf/config.yaml`

CLI flags override config values (config → defaults, CLI → override).

---

## Installation

```bash
poetry install
```

---

## Configuration (OmegaConf/Hydra-style + minimal inheritance)

Default config file: `conf/config.yaml`

### Using experiment configs (extends main config)

In addition to plain YAML configs, the CLI supports a small deterministic inheritance mechanism:

- `extends: ../config.yaml` at the top level of a YAML file.
- The base config is loaded first, then the experiment file overrides it.
- `extends` paths are resolved relative to the experiment file location.

Examples are provided under `conf/experiments/`.

Example (run case118 with its experiment config):

```bash
poetry run python src/power_stability_radius.py --config conf/experiments/case118.yaml demo
```

### Optional: default command in config (omit subcommand)

You can specify a default subcommand at the top level of a config:

```yaml
command: report   # or demo / monte-carlo / table
```

Then you can run without typing the subcommand explicitly:

```bash
poetry run python src/power_stability_radius.py --config conf/experiments/report.yaml
```

The explicit form still works and overrides config:

```bash
poetry run python src/power_stability_radius.py --config conf/experiments/report.yaml report
```

---

## Outputs (runs directory)

Each command creates a run folder under `runs/` and writes:

- `run.log` — full log
- `config.yaml` — effective config used for the run (after CLI overrides)
- `config_source.yaml` — the original config file used (copied)
- `argv.txt` — the exact CLI argv for reproducibility

Output folder behavior is configurable:

- `logging.run_dir_mode: timestamp` → `runs/<timestamp>/` (default)
- `logging.run_dir_mode: overwrite` → `runs/<run_name>/` (folder is deleted/recreated)

---

## Quickstart (single case)

```bash
poetry run python src/power_stability_radius.py demo --input data/input/pglib_opf_case30_ieee.m
```

What it does:
1) Ensures an input MATPOWER/PGLib `.m` case file exists (downloads if needed)
2) Loads the case into pandapower
3) Solves a single-snapshot **DC OPF via PyPSA + HiGHS** to get a feasible base point
4) Builds a DC sensitivity model (`DCOperator` or dense `H_full`)
5) Computes radii per monitored line
6) Writes outputs under the run folder

---

## Demo options

```bash
poetry run python src/power_stability_radius.py demo --help
```

Key options:
- `--dc-mode operator|materialize`
  - `operator` (default): fast, does not materialize `H_full`, no N-1
  - `materialize`: builds dense `H_full` (memory heavy), enables N-1

- `--compute-nminus1 1` (requires `--dc-mode materialize`)
- `--margin-factor` (e.g. `0.9` more conservative)
- `--inj-std-mw` for probabilistic radii

---

## Monte Carlo verification (single case)

You can provide required paths either via CLI flags or via config keys
(`monte_carlo.results`, `monte_carlo.input`).

Example via CLI flags:

```bash
poetry run python src/power_stability_radius.py monte-carlo \
  --results verification/results/case30.json \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --n-samples 50000 \
  --seed 0 \
  --chunk-size 256
```

Example via experiment config:

```bash
poetry run python src/power_stability_radius.py --config conf/experiments/case30.yaml monte-carlo
```

The command prints JSON to stdout and also saves `monte_carlo_stats.json` into the run folder.

---

## Verification report (multiple cases)

```bash
poetry run python src/power_stability_radius.py report \
  --results-dir verification/results \
  --out verification/report.md
```

Or using the provided experiment config (subcommand can be omitted because it has `command: report`):

```bash
poetry run python src/power_stability_radius.py --config conf/experiments/report.yaml
```

Also writes a copy to `<run_dir>/verification_report.md`.

---

## Print/export a table from an existing results.json

```bash
poetry run python src/power_stability_radius.py table runs/<timestamp>/results.json
```

Options:
- `--max-rows N`
- `--radius-field radius_l2|radius_metric|radius_sigma|radius_nminus1`
- `--columns a,b,c` (comma-separated)
- `--table-out path/to/table.txt`
- `--csv-out path/to/table.csv`
```