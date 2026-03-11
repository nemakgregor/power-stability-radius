# Power Stability Radius — Documentation Index

## Project Overview

**Power Stability Radius** is a Python toolkit for computing **robustness certificates** for power system operating points with respect to **line thermal limit violations**. It answers the practical question:

> *How far (in the norm of bus injection perturbations) can the system deviate from a given base dispatch before any transmission line exceeds its thermal rating?*

The answer is produced **per line** and aggregated as a global minimum across all lines. The project supports two physical models:

| Model | Description | Scalability |
|-------|-------------|-------------|
| **DC (linear)** | Lossless DC power flow with PTDF sensitivities | Fast, scales to 10 000+ bus networks |
| **AC (linearized)** | AC power flow Jacobian / adjoint sensitivities around an AC PF base point | Moderate, requires sparse LU factorization |

Several **radius variants** are implemented:

- **L2 radius** — worst-case under Euclidean-norm-bounded perturbations (Cauchy–Schwarz certificate)
- **Sigma radius** — worst-case in "number of standard deviations" units, accounting for heterogeneous per-bus uncertainty
- **Metric radius** — worst-case under an arbitrary symmetric positive-definite (SPD) weight matrix
- **Probabilistic (DC)** — Gaussian overload probability via the Q-function
- **N-1 (DC)** — L2 radius under single-line contingencies with updated sensitivities

Verification is provided through:

- **Monte Carlo simulation** (both DC and AC)
- **Deterministic certificate verification** (soundness check against analytic worst-case perturbations)
- **Comparative metrics analysis** (Spearman rank correlation with empirical overload probabilities)

---

## Documentation Files

| File | Contents |
|------|----------|
| [repository_structure.md](repository_structure.md) | Directory tree, file roles, and module responsibilities |
| [architecture.md](architecture.md) | Software architecture, component interactions, and data flow |
| [mathematical_foundations.md](mathematical_foundations.md) | Formal problem statement, variables, constraints, and all radius formulations with equations |
| [algorithms_and_models.md](algorithms_and_models.md) | Step-by-step algorithmic descriptions for every solver, heuristic, and computation pipeline |
| [scientific_concepts.md](scientific_concepts.md) | Research motivation, hypotheses, trade-offs, and scientific interpretation of results |
| [data_formats.md](data_formats.md) | Input file schemas (MATPOWER, UnitCommitment.jl), output JSON structure, CSV/plots |
| [configuration.md](configuration.md) | YAML config system, all parameters, defaults, and interactions |
| [execution_flow.md](execution_flow.md) | End-to-end execution traces for CLI commands, experiment scripts, and entry points |
| [reproducibility_and_fallbacks.md](reproducibility_and_fallbacks.md) | Deterministic repair paths, surrogate values, tie-break rules, and how they are surfaced in metadata |
| [experiments_and_evaluation.md](experiments_and_evaluation.md) | Experimental pipeline, benchmark cases, metrics, and reproducibility procedures |
| [developer_guide.md](developer_guide.md) | Extension points, coding conventions, adding new algorithms/metrics/parsers |
| [limitations_and_assumptions.md](limitations_and_assumptions.md) | Mathematical and implementation assumptions, scalability, known limitations |
| [metrics.md](metrics.md) | Complete reference for all metrics: formulas, inputs, predictive vs a posteriori classification |
| [glossary.md](glossary.md) | Definitions of domain terms, abbreviations, and key variable names |

---

## Recommended Reading Paths

### For a new developer
1. [repository_structure.md](repository_structure.md) — understand where code lives
2. [architecture.md](architecture.md) — understand component interactions
3. [execution_flow.md](execution_flow.md) — trace a computation end-to-end
4. [developer_guide.md](developer_guide.md) — learn how to extend

### For a researcher
1. [mathematical_foundations.md](mathematical_foundations.md) — formal problem definition and all formulas
2. [scientific_concepts.md](scientific_concepts.md) — research motivation and hypotheses
3. [algorithms_and_models.md](algorithms_and_models.md) — algorithmic details
4. [experiments_and_evaluation.md](experiments_and_evaluation.md) — experimental methodology

### For a technical reviewer
1. [mathematical_foundations.md](mathematical_foundations.md) — verify correctness of formulations
2. [limitations_and_assumptions.md](limitations_and_assumptions.md) — understand what is and isn't modeled
3. [experiments_and_evaluation.md](experiments_and_evaluation.md) — reproduce results
4. [data_formats.md](data_formats.md) — understand inputs and outputs

---

## Quick Start

### Installation

```bash
poetry install
```

### Run a basic computation

```bash
poetry run python src/power_stability_radius.py \
    --config conf/config.yaml \
    compute \
    --input data/input/pglib_opf_case30_ieee.m \
    --slack-bus 0 \
    --base-dispatch case \
    --output results.json
```

### Run Monte Carlo verification

```bash
poetry run python src/power_stability_radius.py \
    --config conf/config.yaml \
    monte-carlo \
    --results-path results.json \
    --input data/input/pglib_opf_case30_ieee.m \
    --slack-bus 0 \
    --mode dc \
    --n-samples 10000
```

### Run experiments for paper

```bash
python -m experiments.run_pglib_sweep
python -m experiments.run_sigma_radius --config experiments/configs/sigma_case2000_goc.yaml
```
