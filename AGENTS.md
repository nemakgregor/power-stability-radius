# AGENTS.md

Instructions for AI coding agents working in this repository.

## Start Here

At the beginning of a task, read:

1. `.codex/docs/README.md`
2. `.codex/docs/project-memory.md`
3. `.codex/docs/session-log.md`

Then open only the project docs or source files needed for the task. The public
documentation source of truth starts at `docs/index.md`; the units and schema
contract is `UNITS_CONTRACT.md`.

For broad maintainability work, also read
`.codex/docs/design-principles-agent.md` and run the local audit:

```bash
python tools/design_principles_audit.py --root .
```

## Memory Rule

After each meaningful conversation or code change:

- Add a concise dated entry to `.codex/docs/session-log.md`.
- Update `.codex/docs/project-memory.md` only when a durable rule, decision,
  pitfall, or result summary changed.
- Keep `.codex/docs` compact. Summarize and replace stale notes instead of
  appending endlessly.
- Do not paste long logs, CSVs, JSON outputs, or generated tables into Codex
  memory files.

## Project Contracts

- Preserve deterministic ordering of bus and line coordinates.
- Do not introduce implicit downloads; missing inputs should fail unless
  `allow_download=true`.
- Keep AC certificates tied to an AC PF/FPF base point. DC OPF is a dispatch
  source, not the AC linearization point.
- Preserve balanced-disturbance projection in radius computations.
- Keep verification as independent from computation as practical.
- Results schema is currently `__meta__.schema_version = 3`.
- Treat unit semantics in `UNITS_CONTRACT.md` as binding.
- Generated artifacts in `analysis_output/`, `runs/`, `run_artifacts/`,
  `verification/`, and `data/` are git-ignored/user state; do not delete or
  rewrite them unless explicitly asked.

## Development Commands

Install:

```bash
poetry install
```

Run tests:

```bash
poetry run python -m pytest -q
```

Run formatting check:

```bash
poetry run ruff format --check .
```

Run DRY/KISS/YAGNI/SOLID design audit:

```bash
python tools/design_principles_audit.py --root .
```

Typical compute command:

```bash
poetry run python entry_points/power_stability_radius.py --config conf/config.yaml --run-tests 0 compute --input data/input/pglib_opf_case30_ieee.m --slack-bus 0 --base-dispatch case
```

## Documentation Updates

- New or renamed runnable entry point: update `docs/entry_points.md`.
- Changes to units, result fields, schema, base-point semantics, or fail-fast
  behavior: update `UNITS_CONTRACT.md` and tests.
- Changes to CLI/config behavior: update `docs/configuration.md`,
  `docs/execution_flow.md`, and relevant tests.
- Keep `.codex/docs` as a compact memory layer, not a duplicate of `docs/`.
