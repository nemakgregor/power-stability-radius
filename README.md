# power-stability-radius

Compute simple “stability radius” (robustness margins) for power grids using a DC sensitivity (PTDF-like) model and per-line L2 radii.

## Quickstart

```bash
poetry install
poetry run python examples/script_demo.py
```

The demo uses Hydra config from `config/main.yaml`, downloads a MATPOWER IEEE case if needed, runs power flow, builds a DC sensitivity matrix, and writes results into `runs/<timestamp>/results.json`.

## Tests

```bash
poetry run pytest
```