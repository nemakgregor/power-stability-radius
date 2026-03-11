# N-1 Demo

## Purpose

`python -m stability_radius.n1_stability_demo` compares three operating regimes
on the same MATPOWER case:

1. `Cost OPF` - validated AC cost-minimizing OPF.
2. `Radius OPF` - the same AC cost OPF with stability-radius-driven line-limit tightening.
3. `SCOPF` - a screening-based SCOPF proxy that iteratively tightens pre-contingency
   line limits using the worst AC N-1 screening overloads.

The goal is not just to report costs, but to show how the stability radius moves
the operating point toward N-1-secure behavior and how close it gets to the
screening-based SCOPF regime.

## Output Location

All demo artifacts are written under the unified `runs/` tree:

```text
runs/n1_stability_demo/<requested-name>/
```

Examples:

- `--output-dir n1_demo_case118` -> `runs/n1_stability_demo/n1_demo_case118/`
- `--output-dir analysis_output/legacy_case118` -> normalized to
  `runs/n1_stability_demo/legacy_case118/`
- `--output-dir runs/custom_bucket/demo_a` -> preserved as-is under `runs/`

Each run directory contains `run.log`, CSV artifacts, `comparison_summary.txt`,
and the generated plots.

## Command

```bash
python -m stability_radius.n1_stability_demo \
    --input data/input/pglib_opf_case118_ieee.m \
    --output-dir n1_demo_case118 \
    --n-iter 2 \
    --scopf-iter 2
```

Useful flags:

- `--n-iter`: maximum radius-OPF tightening rounds.
- `--scopf-iter`: maximum screening-based SCOPF tightening rounds.
- `--skip-dc-n1`: skip DC N-1 post-processing.
- `--skip-ac-n1-radius`: skip AC N-1 radius computation.
- `--skip-n1-screening`: skip the extra end-of-run AC N-1 screening for Cost/Radius OPF.

## Main Artifacts

- `comparison_summary.txt`: side-by-side comparison of all three regimes.
- `*_radii.csv`: per-line AC L2 radius tables for each regime.
- `n1_screening_*.csv`: brute-force AC N-1 screening results.
- `ac_n1_radii_*.csv`: per-line AC N-1 radius tables when enabled.
- `dc_n1_*.csv`: DC N-1 effective radius tables when enabled.
- `plot_radius_cdf.png`: AC L2 radius CDF for all three regimes.
- `plot_n1_overloads.png`: overload count per contingency for all three regimes.
- `plot_cost_security_tradeoff.png`: cost increase versus N-1 security trade-off.
- `plot_ac_n1_radius_cdf.png`: AC N-1 radius CDF when AC N-1 radius is enabled.

## Important Interpretation Note

`min_headroom_mva` in the summary is a **stability-radius proxy margin**:

```text
headroom_proxy = S_limit_proxy - |S|
```

This is not exactly the same quantity as pandapower's AC OPF thermal constraint.
The AC OPF is enforced through pandapower's **current/loading** limits
(`loading_percent`), while the radius workflow uses an **MVA proxy** derived from
the line rating metadata. Because of that mismatch:

- a feasible AC OPF point can still show slightly negative `min_headroom_mva`;
- the authoritative feasibility numbers for the OPF are the
  `max_line_loading_pct` and `min_line_loading_headroom_pct` values in the
  `AC OPF Constraints (current-based)` section.

## SCOPF Caveat

The demo's `SCOPF` regime is intentionally labeled as a **screening-based SCOPF
proxy**. The repository stack does not expose a native full AC SCOPF solver
through pandapower, so the demo uses an iterative tighten-screen-resolve loop:

1. solve validated AC cost OPF;
2. run brute-force AC N-1 screening;
3. tighten pre-contingency limits on the lines with the worst post-contingency overloads;
4. repeat until the screening stops requesting tighter limits or the iteration budget is exhausted.

This keeps the demo reproducible and comparable to the radius-guided regime while
still giving a practical SCOPF-style baseline.
