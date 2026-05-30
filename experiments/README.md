# Experiments

This directory stores experiment configuration files only. Runnable experiment
fronts live in `entry_points/`, and reusable result aggregation or plotting
logic lives in `src/stability_radius/postprocess/`.

## Layout

```text
experiments/
  configs/
    pglib_sweep.yaml
    sigma_case2000_goc.yaml
    sigma_case2736sp_k.yaml
    sigma_case2869_pegase.yaml
    uc_jl_case118.yaml
  README.md
```

Generated experiment outputs belong under `run_artifacts/` by default. Historical
workspace outputs under `experiments/output/` are ignored by git and are not part
of the source layout.

## Runnable Fronts

Use the documented entry points:

```bash
python entry_points/run_pglib_sweep.py --config experiments/configs/pglib_sweep.yaml
python entry_points/run_sigma_radius.py --config experiments/configs/uc_jl_case118.yaml
python entry_points/run_worst_case_verify.py --sweep-dir run_artifacts/run_pglib_sweep
python entry_points/run_scalability.py --config experiments/configs/pglib_sweep.yaml
```

The full inventory and output contracts are maintained in
`docs/entry_points.md`.

## Post-Processing Modules

Run post-processing through the package modules:

```bash
python -m stability_radius.postprocess.collect_results
python -m stability_radius.postprocess.plot_radius_distribution
python -m stability_radius.postprocess.plot_sigma_vs_time
python -m stability_radius.postprocess.plot_worst_case_heatmap
```
