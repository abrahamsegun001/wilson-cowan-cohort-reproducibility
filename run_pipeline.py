#!/usr/bin/env python3
"""Compact public entry point for the cohort-scale Wilson-Cowan workflow."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

SCRIPTS = {
    "model-selection": ROOT / "src/model_selection.py",
    "dynamics": ROOT / "src/dynamics.py",
    "preflight": ROOT / "src/reconstruction_preflight.py",
    "reconstruct": ROOT / "src/reconstruction.py",
    "aggregate-reconstruction": ROOT / "src/reconstruction_aggregate.py",
    "longitudinal": ROOT / "src/longitudinal.py",
    "hierarchical": ROOT / "src/hierarchical.py",
    "report": ROOT / "src/reporting.py",
    "supplementary-figure": ROOT / "src/supplementary_figure.py",
    "validate-release": ROOT / "scripts/validate_release.py",
}

def verify() -> int:
    required = [
        ROOT / "config.yaml",
        ROOT / "data/preprocessing_manifest_40.csv",
        ROOT / "src/model_selection.py",
        ROOT / "src/reconstruction.py",
        ROOT / "src/dynamics.py",
        ROOT / "src/longitudinal.py",
        ROOT / "src/hierarchical.py",
        ROOT / "src/reporting.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("Repository check failed. Missing:")
        for item in missing:
            print(" -", item)
        return 2
    print("Repository structure check passed.")
    print("Use `python run_pipeline.py <command> --help` for a stage-specific interface.")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact entry point for the cohort-scale Wilson-Cowan reproducibility pipeline."
    )
    parser.add_argument("command", choices=["verify", *SCRIPTS.keys()])
    args, rest = parser.parse_known_args()
    if args.command == "verify":
        return verify()
    cmd = [PYTHON, str(SCRIPTS[args.command]), *rest]
    return subprocess.call(cmd, cwd=ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
