# Testing and CI

This repository treats tests, documentation, and workflow configuration as one system. The goal is not only to verify numerical logic, but also to keep the runnable surface area documented and continuously checked.

## Local Commands

Install dependencies:

```bash
poetry install
```

Run the full test suite:

```bash
poetry run python -m pytest -q
```

Run the formatting check used by CI:

```bash
poetry run ruff format --check .
```

Run the local design-principles audit used by the Codex agent:

```bash
python tools/design_principles_audit.py --root .
```

If you want to rewrite files to Ruff's formatting locally:

```bash
poetry run ruff format .
```

## What the Tests Cover

The suite in `tests/` includes:

- Core numerical and workflow tests for `src/stability_radius/`
- CLI and entry-point wiring tests for `entry_points/`
- Smoke tests for reporting and plotting entry points
- Docs-as-code checks that verify:
  - every `entry_points/*.py` module is documented in [entry_points.md](entry_points.md)
  - [docs/index.md](index.md) links to the key operational docs
  - `README.md` points to the docs hub and primary CLI
  - `.github/workflows/ci.yml` still runs formatting checks and pytest
- A Codex-facing design-principles audit in `tools/design_principles_audit.py`
  that reports DRY, KISS, YAGNI, and SOLID candidates for human review

That last group is what keeps the entry-point inventory and CI behavior from silently drifting out of sync with the codebase.

## GitHub Actions

The repository workflow lives at `.github/workflows/ci.yml`.

Current CI behavior:

1. Run on `push`, `pull_request`, and `workflow_dispatch`.
2. Install Python.
3. Install Poetry explicitly with `pip install poetry`.
4. Install project dependencies with `poetry install --no-interaction`.
5. Run `poetry run ruff format --check .`.
6. Run `poetry run python -m pytest -q --cov=src/stability_radius --cov-report=term-missing --cov-fail-under=45`.

We intentionally do not use `actions/setup-python`'s built-in Poetry cache here, because that mode expects the `poetry` executable to already exist during the Python setup step and can fail before Poetry is installed.

Coverage is reported in the CI log, and CI currently enforces a conservative `45%` floor on `src/stability_radius`.

That scope is intentional:

- the coverage gate protects the maintained library and application layers;
- `entry_points/` is still tested, but mostly through smoke and wiring tests rather than a blanket line-coverage floor;
- experiment-heavy scripts remain documented and checked, but they do not dominate the repository-wide coverage percentage.

## Notes on "CD"

There is no deployment target configured in this repository right now, so the GitHub automation is CI-focused. The project is still kept continuously releasable by checking formatting, tests, documentation references, and coverage on every change.
