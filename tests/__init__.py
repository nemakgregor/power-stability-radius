from __future__ import annotations

"""
Test package marker.

Why this file is necessary
--------------------------
Pytest can treat the `tests/` directory as a (namespace) package depending on the
project/pytest configuration (e.g. `consider_namespace_packages=true`) and then
will try to import `tests/__init__.py` during collection/setup.

If the file is missing, collection fails with:

    ImportError while importing test module '.../tests/__init__.py'
    ImportError: .../tests/__init__.py

This file is intentionally empty (no side effects, no sys.path modifications).
All test import-path setup remains in `tests/conftest.py` (adds `src/` to sys.path).
"""
