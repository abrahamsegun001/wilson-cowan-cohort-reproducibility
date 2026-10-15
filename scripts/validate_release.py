#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
required = {
    ROOT / "results/full/model_selection/cohort_model_selection_800.csv": 800,
    ROOT / "results/full/dynamics/cohort_dynamics_800.csv": 800,
    ROOT / "results/full/reconstruction/cohort_gpinn_reconstruction_metrics_40.csv": 40,
}

errors = []
for path, n in required.items():
    if not path.exists():
        errors.append(f"missing: {path.relative_to(ROOT)}")
        continue
    try:
        rows = len(pd.read_csv(path))
    except Exception as exc:
        errors.append(f"unreadable: {path.relative_to(ROOT)} ({exc})")
        continue
    if rows != n:
        errors.append(f"{path.relative_to(ROOT)}: expected {n} rows, found {rows}")

long_path = ROOT / "results/full/longitudinal/chunk_dynamics_nearest_stable_equilibrium.csv"
if not long_path.exists():
    errors.append(f"missing: {long_path.relative_to(ROOT)}")

hier_dir = ROOT / "results/full/hierarchical"
if not hier_dir.exists() or not any(hier_dir.glob("*.csv")):
    errors.append("missing hierarchical CSV outputs")

if errors:
    print("RELEASE VALIDATION FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(2)

print("RELEASE VALIDATION PASSED")
print("Core model-parameter and reconstruction tables are present.")
