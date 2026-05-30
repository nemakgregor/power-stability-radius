# Design Principles Agent

Purpose: review Python changes for DRY, KISS, YAGNI, and SOLID issues, then
make the smallest verified repair that improves maintainability without
changing scientific behavior.

Run this audit before broad cleanup or after substantial code changes:

```bash
python tools/design_principles_audit.py --root .
```

Use stricter behavior for CI-like local checks:

```bash
python tools/design_principles_audit.py --root . --fail-on-findings
```

## Review Rules

- DRY: factor repeated non-trivial logic only when the shared abstraction has a
  stable meaning in the domain. Do not merge code that merely looks similar but
  represents different contracts.
- KISS: split long, branch-heavy, or deeply nested functions when the split
  reveals existing concepts such as base-point construction, geometry,
  certificate classification, or result serialization.
- YAGNI: remove unused private helpers and speculative options only after
  verifying references with `rg`, tests, and docs. Keep research hooks that are
  documented and exercised.
- SOLID: keep entry points thin, preserve domain boundaries, and move reusable
  behavior into `src/stability_radius` public modules instead of scripts.

## Repair Policy

- Treat audit output as candidates, not automatic truth.
- Fix confirmed issues with targeted edits and focused tests.
- Do not change units, schema fields, solver policy, or artifact paths while
  doing design cleanup unless the task explicitly requires that contract change.
- After repairs, run `python -m ruff format --check .`, focused tests, and
  `python -m pytest -q` when feasible.
