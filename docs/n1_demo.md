# N-1 Demo

## Purpose

`python entry_points/n1_stability_demo.py` compares three operating regimes
on the same MATPOWER case:

1. `Cost OPF` - AC cost-minimizing OPF, then replayed with AC PF for a
   physically consistent base point.
2. `Radius OPF` - the same AC cost OPF with stability-radius-driven line-limit tightening.
3. `SCOPF` - a screening-based SCOPF proxy that iteratively tightens pre-contingency
   line limits using the worst AC N-1 screening overloads.

The goal is not just to report costs, but to show how the stability radius moves
the operating point toward N-1-secure behavior and how close it gets to the
screening-based SCOPF regime.

## Output Location

All demo artifacts are written under the unified `run_artifacts/` tree:

```text
run_artifacts/n1_stability_demo/<requested-name>/
```

Examples:

- `--output-dir n1_demo_case118` -> `run_artifacts/n1_stability_demo/n1_demo_case118/`
- `--output-dir analysis_output/case118` -> normalized to
  `run_artifacts/n1_stability_demo/case118/`
- `--output-dir run_artifacts/custom_bucket/demo_a` -> preserved as-is under `run_artifacts/`

Each run directory contains `debug.log`, CSV artifacts, `comparison_summary.txt`,
and the generated plots.

## Command

```bash
python entry_points/n1_stability_demo.py \
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
- `opf_line_limit_consistency_*.csv`: per-line check that the stability-radius
  proxy limit matches the reconstructed pandapower OPF line-limit model.
- `n1_screening_*.csv`: brute-force AC N-1 screening results.
- `ac_n1_radii_*.csv`: per-line AC N-1 radius tables when enabled.
- `dc_n1_*.csv`: DC N-1 effective radius tables when enabled.
- `plot_radius_cdf.png`: AC L2 radius CDF for all three regimes.
- `plot_n1_overloads.png`: four-panel AC N-1 screening summary with sorted
  overload-count curves, sorted peak-loading curves, adaptive top contingencies
  (overload counts or peak loading when all contingencies pass), and
  pass/fail/diverged shares.
- `plot_cost_security_tradeoff.png`: cost increase versus N-1 security trade-off.
- `plot_ac_n1_radius_cdf.png`: AC N-1 radius CDF when AC N-1 radius is enabled.

## Important Interpretation Note

`min_headroom_mva` in the summary is a **stability-radius proxy margin**:

```text
headroom_proxy = S_limit_proxy - |S|
```

This is not exactly the same quantity as pandapower's post-PF current loading
diagnostic in this demo. `runopp` is configured with `OPF_FLOW_LIM=0`, so the
solver enforces an **apparent-power** branch limit model. The summary also
reports a separate post-PF **current/loading** diagnostic (`loading_percent`).

To keep the demo internally consistent, the script now rewrites the line-level
explicit MVA rating used by the radius utilities so that it matches the same
current-based branch model that pandapower OPF uses. The
`opf_line_limit_consistency_*.csv` artifacts and the `Line Limit Consistency
(proxy vs OPF)` section in the summary verify that this alignment actually holds
for each solved regime using the same `sqrt(3) * vn_kv * max_i_ka * df * parallel`
formula and `max_loading_percent` scaling that pandapower uses for line
constraints.

Because the post-PF diagnostic still uses a different loading metric:

- a feasible AC OPF point can still show slightly negative `min_headroom_mva`;
- a converged AC OPF point can also show slightly negative
  `min_line_loading_headroom_pct` in the post-PF diagnostic table;
- the current-loading numbers should be interpreted as replay diagnostics, not as
  the exact branch metric used inside `runopp`.

## SCOPF Caveat

The demo's `SCOPF` regime is intentionally labeled as a **screening-based SCOPF
proxy**. The repository stack does not expose a native full AC SCOPF solver
through pandapower, so the demo uses an iterative tighten-screen-resolve loop:

1. solve AC cost OPF and replay it with AC PF;
2. run brute-force AC N-1 screening;
3. tighten pre-contingency limits on the lines with the worst post-contingency overloads;
4. repeat until the screening stops requesting tighter limits or the iteration budget is exhausted.

This keeps the demo reproducible and comparable to the radius-guided regime while
still giving a practical SCOPF-style baseline.

