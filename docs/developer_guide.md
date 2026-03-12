# Developer Guide

This document provides guidance for extending and maintaining the codebase.

> Cross-references: [architecture.md](architecture.md) for component overview, [repository_structure.md](repository_structure.md) for file locations.

---

## 1. Development Environment

### Setup

```bash
# Clone repository
git clone <repo-url>
cd power-stability-radius

# Install dependencies with Poetry
poetry install

# Activate virtualenv
poetry shell
```

### Dependencies

From `pyproject.toml`:
- **Core**: `numpy`, `scipy`, `pandas`, `pandapower`, `pypsa`, `linopy`, `pyyaml`
- **Plotting**: `matplotlib`
- **Statistics**: `scipy` (for Spearman correlation)
- **Optional**: `highspy` (HiGHS LP/MIP solver, used by PyPSA for DC OPF)
- **Testing**: `pytest`

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_dc_model.py

# Verbose with output
pytest tests/ -v -s
```

### CI

GitHub Actions CI is configured in `.github/workflows/ci.yml`. It runs the test suite on push and pull requests.

---

## 2. Adding a New Radius Variant

### Step-by-step

1. **Create a new module** in `src/stability_radius/radii/`:

   ```python
   # src/stability_radius/radii/my_radius.py
   from __future__ import annotations

   import numpy as np
   from .common import LineBaseQuantities, get_line_base_quantities, line_key

   def compute_my_radius(
       net,
       H_full: np.ndarray,
       *,
       custom_param: float,
       base: LineBaseQuantities | None = None,
   ) -> dict[str, dict[str, Any]]:
       base_q = base or get_line_base_quantities(net)
       results = {}
       for pos, lid in enumerate(base_q.line_indices):
           g_l = H_full[pos, :]
           margin = float(base_q.margin_mw[pos])
           # Your radius computation here
           my_r = margin / some_norm(g_l, custom_param)
           results[line_key(lid)] = {
               "radius_my_variant": float(my_r),
               "margin_mw": margin,
           }
       return results
   ```

2. **Export** from `radii/__init__.py`:
   ```python
   from .my_radius import compute_my_radius
   __all__.append("compute_my_radius")
   ```

3. **Integrate** into `workflows.py`:
   - Add parameter to `DCExtensionsConfig` or `ACExtensionsConfig`
   - Call `compute_my_radius()` in the appropriate phase of `compute_results_for_case()`
   - Merge results via `_merge_line_results()`

4. **Add CLI support** in `cli.py`:
   - Add `--my-radius-param` argument to the `compute` subparser
   - Pass to `compute_results_for_case()`

5. **Add tests** in `tests/test_my_radius.py`:
   - Test with synthetic data (3-bus network)
   - Test edge cases (zero margin, zero sensitivity, inf limit)
   - Test consistency with existing radii where applicable

### Key patterns

- Use `LineBaseQuantities` for per-line base data (avoids redundant OPF calls)
- Use `line_key(lid)` for consistent result keys: `"line_0"`, `"line_1"`, etc.
- Return dict of dicts (consistent with all other radius functions)
- Use `float()` wrapping for all numeric values (JSON serialization)

---

## 3. Adding a New Base Dispatch Mode

### Step-by-step

1. **Create the base point function** in `base_point/`:
   ```python
   # src/stability_radius/base_point/my_dispatch.py
   def compute_my_dispatch(net, *, slack_bus: int, **kwargs) -> BasePointDC:
       # Your dispatch logic
       return BasePointDC(
           p_gen_mw=...,
           injections_mw=...,
           flows_mw=...,
           slack_absorption_mw=...,
       )
   ```

2. **Add dispatch mode** to `workflows.py`:
   - Add `"my_dispatch"` as a valid `base_dispatch` option
   - Add handling in the Phase 2 (DC base) or Phase 4 (AC base) sections

3. **Update CLI** in `cli.py`:
   - Add `"my_dispatch"` to the `choices` for `--base-dispatch`

4. **Add test** verifying the dispatch produces valid base flows

---

## 4. Adding a New Input Format Parser

### Step-by-step

1. **Create parser module** in `parsers/`:
   ```python
   # src/stability_radius/parsers/my_format.py
   import pandapower as pp

   def load_network(path: Path) -> pp.pandapowerNet:
       # Parse your format → pandapower network
       net = pp.create_empty_network()
       # Populate net.bus, net.line, net.gen, net.load, net.ext_grid
       return net
   ```

2. **Register** in the workflow or CLI for auto-detection based on file extension

3. **Key requirements** for the pandapower network:
   - `net.bus` with `vn_kv` column
   - `net.line` with `from_bus`, `to_bus`, `x_ohm_per_km`, `length_km`, `parallel`, `max_i_ka`
   - `net.gen` with `bus`, `p_mw`, `min_p_mw`, `max_p_mw`, `in_service`
   - `net.load` with `bus`, `p_mw`, `q_mvar`, `in_service`
   - `net.ext_grid` with `bus`, `vm_pu` (at least one, at the slack bus)
   - `net.sn_mva` (system base, default 100)

---

## 5. Adding a New Metric for Comparison

### In `metrics/ac_baselines.py`:

```python
def compute_baseline_metrics(results: dict) -> dict:
    baselines = {}
    for k, v in results.items():
        if not k.startswith("line_"):
            continue
        # Add your metric
        baselines[k] = {
            "loading_ratio": ...,
            "headroom_mva": ...,
            "my_new_metric": compute_my_metric(v),
        }
    return baselines
```

### In `metrics_analysis.py`:

1. Add the metric column name to `metric_cols` list
2. If "lower = more dangerous", add to `_NEGATE_FOR_CORRELATION` set
3. It will automatically be included in Spearman correlation and precision-at-k

---

## 6. Adding a New Experiment

1. **Create experiment script** in `experiments/`:
   ```python
   # experiments/run_my_experiment.py
   """Experiment N: Description."""

   from stability_radius.workflows import compute_results_for_case

   def run(config_path: Path) -> None:
       cfg = load_config(config_path)
       # Your experiment logic

   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", ...)
       args = parser.parse_args()
       run(Path(args.config))
   ```

2. **Create experiment config** in `experiments/configs/my_experiment.yaml`

3. **Follow conventions**:
   - Save outputs to `runs/<module>/<name>/` (or directly `runs/<module>/`)
   - Include `debug.log` with detailed logs
   - Save `summary.json` for aggregated results
   - Generate `.png` and `.pdf` versions of plots

---

## 7. Coding Conventions

### Observed patterns in the codebase:

- **Type annotations**: All function signatures use type hints
- **Frozen dataclasses**: Data containers use `@dataclass(frozen=True)`
- **Explicit float conversion**: All numeric results wrapped in `float()` for JSON serialization
- **Conservative error handling**: Explicit validation with informative `ValueError` messages
- **Logging levels**: INFO for user-visible progress, DEBUG for computation details
- **Deterministic ordering**: Always `sorted(net.bus.index)`, `sorted(net.line.index)`
- **Numerical guards**: `eps` thresholds for division-by-zero protection (typically 1e-12)
- **Per-line keys**: `line_key(lid)` → `"line_{lid}"` for consistent naming
- **Units in names**: `_mw`, `_mvar`, `_mva`, `_pu`, `_rad`, `_kv` suffixes
- **Chunked processing**: Large matrix operations processed in configurable chunks

### Style

- Imports: `from __future__ import annotations` at top
- Docstrings: Google-style with Parameters/Returns/Raises
- Line length: ~100 chars (not strict)
- f-strings for formatting
- `noqa: BLE001` for intentionally broad exception catches in defensive code

---

## 8. Debugging

### Common debugging scenarios:

#### AC PF doesn't converge
- Check `ac_pf_attempt` and `ac_pf_repairs` in results `__meta__`
- Try `--ac-pf-init flat` (more robust than DC init)
- Try `--base-dispatch acpf` (uses case data directly)
- Check for disconnected buses or extreme impedance values

#### DC-AC consistency mismatch
- Check `__meta__.consistency` fields in results
- Large mismatches indicate the DC model is inadequate for this network
- Phase-shifting transformers can cause systematic offsets

#### Zero radius for many lines
- Check if thermal limits are being loaded correctly (some cases have rateA=0)
- `unconstrained_line_nom_mw` parameter controls the surrogate limit for such lines
- Check `is_unconstrained` flag in per-line results

#### Jacobian factorization failure
- Usually indicates a disconnected or ill-conditioned network
- Check for isolated buses or very small impedances
- The lossless policy (r=0) can sometimes make the network numerically difficult

### Debug test scripts

Two debug scripts exist in `tests/`:
- `tests/debug_h_vector_case118.py` — Validates h-vector computation for case118
- `tests/debug_jacobian_case118.py` — Validates Jacobian vs pandapower for case118

---

## 9. Test Structure

Tests use `pytest` with fixtures defined in `conftest.py`:

### Key Fixtures

```python
@pytest.fixture
def case5_net():
    """5-bus PJM test case."""
    return load_network(Path("data/input/pglib_opf_case5_pjm.m"))

@pytest.fixture
def case14_net():
    """14-bus IEEE test case."""

@pytest.fixture
def case30_net():
    """30-bus IEEE test case."""
```

### Test Categories

| Category | Files | Purpose |
|----------|-------|---------|
| Unit tests | `test_dc_model.py`, `test_radii_*.py` | Individual component correctness |
| Integration tests | `test_unit_consistency_end_to_end.py` | Full pipeline consistency |
| Concept tests | `test_certificate_concept.py`, `test_radius_concept_synthetic.py` | Mathematical concept validation |
| Smoke tests | `test_ac_radius_smoke.py` | Quick sanity checks on real cases |
| Finite difference | `test_h_vector_fd.py`, `test_ac_jacobian_vs_pandapower.py` | Numerical derivative validation |
| Config tests | `test_config_extends.py`, `test_config_project_defaults.py` | Configuration system |
| CLI tests | `test_cli_*.py` | CLI argument parsing |
| Verification tests | `test_verify_worst_case.py`, `test_verification_*.py` | Verification pipeline |
