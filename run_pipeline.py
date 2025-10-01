#!/usr/bin/env python
"""Convenience wrapper for the pipeline.

Why this file exists:
You adopted a modern *src/* layout (code lives in src/crypto_analysis), but you are
invoking the script with a direct path (e.g. `python run_pipeline.py` or via Jenkins).

Python does NOT automatically add the `src` directory to sys.path when you execute a
script from the project root unless the package is installed (pip install -e .) or
you export PYTHONPATH=src. This wrapper safely amends sys.path at runtime so the
package `crypto_analysis` can be imported, then delegates to
`crypto_analysis.run_pipeline.main`.

Recommended alternatives (either also work):
  1. Install editable:  pip install -e .   (then use: python -m crypto_analysis.run_pipeline)
  2. Set env var before running:  export PYTHONPATH=src
  3. Use this wrapper:  python run_pipeline.py

This wrapper keeps CI/CD (Makefile, Jenkins, cron) simple without forcing an install step.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Prepend src to sys.path if not already present
if os.path.isdir(SRC_PATH) and SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)

try:
	from crypto_analysis.run_pipeline import main  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
	details = [
		"[FATAL] Could not import package 'crypto_analysis'.",
		f"        Checked that src/ exists at: {SRC_PATH}",
		"        sys.path currently:",
		*[f"          - {p}" for p in sys.path],
		f"        Original error: {e}",
		"",
		"Suggested fixes:",
		"  * pip install -e .",
		"  * or: export PYTHONPATH=src",
		"  * or ensure you are running from the project root directory.",
	]
	print("\n".join(details))
	raise SystemExit(1)


if __name__ == "__main__":
	main()

