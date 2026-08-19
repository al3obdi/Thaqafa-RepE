"""Command-line entry points for Thaqafa-RepE.

This package marker exists so that mypy maps ``scripts/*.py`` to a single
module path; without it, a script imported by tests as
``scripts.run_space_extraction`` is also seen as top-level
``run_space_extraction`` and mypy aborts with a duplicate-module error.
"""
