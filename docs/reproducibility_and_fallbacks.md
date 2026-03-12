# Reproducibility And Fallbacks

This document is the source of truth for runtime behaviors that are deterministic
but can still change the chosen operating point, feasibility status, or metadata.
These behaviors are not random; they are explicit repair rules, fallback chains,
and surrogate values that the code applies in well-defined situations.

## Deterministic Tie-Breaks

| Behavior | Location | Rule |
|----------|----------|------|
| Bus ordering | `dc_model.py`, `ac_model.py`, `workflows.py` | Always `sorted(net.bus.index)` |
| Line ordering | `dc_model.py`, `ac.py`, `workflows.py` | Always `sorted(net.line.index)` |
| Slack auto-detect with multiple `ext_grid` rows | `base_point/pandapower_tools.py`, `metrics_analysis.py` | Use the smallest in-service `ext_grid` bus id |
| Load-proportional sigma arrays | `metrics_analysis.py` | Reindex bus loads to `sorted(net.bus.index)` before building per-bus sigma arrays |
| Download mirror selection | `utils/download.py` | Try candidate URLs in a stable order; env overrides replace defaults exactly |

The slack-bus rule matters because MATPOWER conversions can produce multiple
`ext_grid` rows. The project now treats this as a deterministic tie-break rather
than "first row wins".

## Base-Point Repair And Fallback Chains

### HiGHS deterministic defaults

Location: `config.py`, `conf/config_shared.yaml`

The project default for `opf.threads` is `4` in both Python and YAML. This is a
performance-oriented default; set `opf.threads=1` when the strictest
reproducibility matters more than runtime. Multi-threaded HiGHS can change
pivot order and produce different dispatch choices on tied or weakly conditioned
LPs. The default `random_seed=42` is aligned across the same entrypoints.

### DC OPF adaptive headroom

Location: `workflows.py`

When `base_dispatch=dc_opf`, the solver first tries the configured
`opf.headroom_factor`. If the OPF is infeasible, the workflow relaxes toward
`1.0` using this deterministic schedule:

1. Configured `headroom_factor`
2. `0.92`
3. `0.95`
4. `0.98`
5. `1.0`

Only values strictly larger than the configured headroom are appended. The
actual value that succeeded is recorded in `__meta__.opf.headroom_factor_used`.

### AC PF repair cascade

Location: `base_point/pypsa_pf.py`

For `solver="pandapower"`, the Newton-Raphson solve uses a fixed three-stage
repair cascade:

1. Primary attempt with the requested init
2. Retry with the opposite init (`flat` <-> `dc`)
3. Relaxed retry with `enforce_q_lims=False`, `distributed_slack=False`,
   `init="flat"`

The winning stage is recorded in `pf_attempt`. Any modifications are recorded in
`pf_repairs`.

### Large-network distributed-slack guard

Location: `base_point/pypsa_pf.py`

If distributed slack is requested on a network with at least `300` buses, the
project disables it before calling `pandapower.runpp()`. This is a protective
guard against a known pandapower crash in the distributed-slack path on large
systems.

Metadata:

- `distributed_slack_requested`: user/config request
- `distributed_slack`: effective setting used by the successful solve
- `pf_repairs` may include `distributed_slack_auto_disabled_large_network`

### AC FPF repair cascade

Location: `base_point/pandapower_opp.py`

The AC feasibility solve (`pandapower.runopp`) uses at most three attempts:

1. `primary`: configured voltage bounds and init
2. `relaxed_v`: voltage bounds widened to `[0.85, 1.15]`, `init="flat"`
3. `relaxed_all`: voltage bounds widened to `[0.80, 1.20]`, `init="flat"`

If the OPP solve succeeds, the code performs a post-OPP `runpp()` validation.
If that PF also converges, the OPP operating point is overwritten with the PF
solution and `pf_repairs` gets `post_opp_pf_applied`.

## Deterministic Numeric Surrogates

### Line thermal limits

Location: `radii/common.py`

If a line rating is `0`, `NaN`, or `+inf`, the project treats the line as
"unconstrained" and replaces the limit with the finite surrogate
`DEFAULT_OPF.unconstrained_line_nom_mw` (default: `1e5`).

### AC FPF line data repair

Location: `base_point/pandapower_opp.py`

Before `runopp()`, the code applies these deterministic defaults:

- Missing `max_loading_percent` -> `100.0`
- Missing or non-positive `max_i_ka` -> `100.0` kA surrogate

These repairs prevent unconstrained lines from being interpreted as zero-capacity
elements by pandapower.

### AC FPF generator/ext-grid bounds

Location: `base_point/pandapower_opp.py`

When generator bounds are incomplete, the OPP setup uses fixed surrogates:

- Missing/non-positive `max_p_mw` -> `max(min_p_mw + 1.0, 100.0)`
- Missing `min_q_mvar` -> `-999.0`
- Missing `max_q_mvar` -> `999.0`
- `ext_grid` active/reactive bounds -> `max(1000.0, 2.0 * total_load)`

### Distributed-slack weights

Location: `base_point/pypsa_pf.py`

When distributed slack is enabled, participation weights are based on generator
headroom `max_p_mw - p_mw`. If no generator has positive headroom, the
`ext_grid` receives a fixed fallback weight of `100.0` MW.

### Auto-created slack source

Location: `base_point/pandapower_tools.py`

If the resolved slack bus has no in-service `ext_grid`, the project creates one
automatically with `vm_pu=1.0` and `va_degree=0.0`.

## Test Coverage

The following tests lock these behaviors:

- `tests/test_pandapower_tools.py`
- `tests/test_workflows_helpers.py`
- `tests/test_ac_pf_repair_cascade.py`
- `tests/test_ac_fpf_base_dispatch.py`
- `tests/test_metrics_analysis.py`
- `tests/test_config_project_defaults.py`

These tests complement the existing AC/DC pipeline tests and are intended to
catch silent changes in repair logic, surrogate values, or metadata semantics.

## Intentional Non-Result Variability

Some outputs are intentionally run-dependent but do not change the computed
certificate values:

- `logging.run_dir_mode=timestamp` creates time-based run directories
- `__meta__.compute_time_sec` records wall-clock time

These fields affect artifact names and performance reporting, not the radii or
base-point metadata.
