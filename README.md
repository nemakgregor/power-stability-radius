# power-stability-radius

Compute simple “stability radius” (robustness margins) for power grids using a DC sensitivity (PTDF-like) model and per-line radii:
- `radius_l2` (L2 ball)
- `radius_metric` (weighted/metric)
- `radius_sigma` + overload probability (Gaussian injections)
- `radius_nminus1` (effective N-1 via LODF)

All workflows are launched via a **single entrypoint**:
`src/power_stability_radius.py`

The main demo writes results into `runs/<timestamp>/`.

---

## Installation

```bash
poetry install
```

---

## Quickstart (single case)

```bash
poetry run python src/power_stability_radius.py demo --input data/input/pglib_opf_case30_ieee.m
```

What it does:
1) Ensures an input MATPOWER/PGLib `.m` case file exists (downloads if needed)
2) Loads the case into pandapower
3) Runs AC PF once and extracts base flows / estimated limits
4) Builds DC sensitivity matrix `H_full`
5) Computes all radii per line
6) Writes outputs under `runs/<timestamp>/`

### Outputs in `runs/<timestamp>/`

- `results.json` — all per-line fields + `__meta__`
- `results_table.csv` — CSV export (same columns/order as the internal table)
- `run.log` — file log (**all project log levels**; third-party logs are filtered).  
  The large ASCII results table is written **only to run.log** (never printed to console, never saved as `.txt`).

---

## Demo CLI options

```bash
poetry run python src/power_stability_radius.py demo --help
```

Common options:

### Input / outputs
- `--input` — path to case file  
- `--export-results` — additionally copy `results.json` to a fixed path (useful for verification)
  ```bash
  poetry run python src/power_stability_radius.py demo \
    --input data/input/pglib_opf_case30_ieee.m \
    --export-results verification/results/case30.json
  ```

### Power flow / model
- `--slack-bus` — slack bus id/position (consistent with `build_dc_matrices`)
- `--margin-factor` — multiplies estimated line limits (e.g. 0.9 more conservative)

### Probabilistic
- `--inj-std-mw` — per-bus stddev for diagonal Sigma (MW)

### N-1
- `--nminus1-update-sensitivities` — 1/0
- `--nminus1-islanding` — `skip` or `raise`

### Table export/logging
- `--table-columns a,b,c` — comma-separated list of table columns (default: full set)
- `--max-rows N` — limit rows in the logged/saved table
- `--save-csv 1/0`

### Logging
- `--log-level` — console log level
- `--log-file-level` — file log level (keep DEBUG to include the full table)

---

## Printing/exporting a table from an existing `results.json`

```bash
poetry run python src/power_stability_radius.py table runs/<timestamp>/results.json
```

Options:
- `--max-rows N`
- `--radius-field radius_l2|radius_metric|radius_sigma|radius_nminus1`
- `--columns a,b,c` (comma-separated)
- `--table-out path/to/table.txt`
- `--csv-out path/to/table.csv`

---

## Verification workflow (multiple cases + report)

### 1) Generate per-case results files

Example (case30 + case118):
```bash
poetry run python src/power_stability_radius.py demo \
  --input data/input/pglib_opf_case30_ieee.m \
  --export-results verification/results/case30.json

poetry run python src/power_stability_radius.py demo \
  --input data/input/pglib_opf_case118_ieee.m \
  --export-results verification/results/case118.json
```

Tip: for multiple cases, a shell loop is usually simplest:
```bash
for c in 30 118 300; do
  poetry run python src/power_stability_radius.py demo \
    --input data/input/pglib_opf_case${c}_ieee.m \
    --export-results verification/results/case${c}.json
done
```

### 2) Monte Carlo coverage for a single case

```bash
poetry run python src/power_stability_radius.py monte-carlo \
  --results verification/results/case30.json \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --n-samples 50000 \
  --seed 0 \
  --chunk-size 256
```

### 3) Generate the aggregated verification report

```bash
poetry run python src/power_stability_radius.py report \
  --results-dir verification/results \
  --out verification/report.md
```

A copy is also written into `runs/<timestamp>/verification_report.md` together with `run.log`.