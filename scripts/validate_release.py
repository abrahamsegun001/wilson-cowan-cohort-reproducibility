#!/usr/bin/env python3

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

expected_rows = {
    ROOT / "results/full/model_selection/cohort_model_selection_800.csv": 800,
    ROOT / "results/full/dynamics/primary_valid/cohort_dynamics_primary_valid_738.csv": 738,
    ROOT / "results/full/dynamics/sensitivity_all800/cohort_dynamics_sensitivity_all800.csv": 800,
    ROOT / "results/full/reconstruction/cohort_gpinn_reconstruction_metrics_40.csv": 40,
    ROOT / "results/full/longitudinal/subject_session_dynamics_40.csv": 40,
    ROOT / "results/full/longitudinal/subject_longitudinal_wide_10.csv": 10,
    ROOT / "results/full/hierarchical/mouse_overall_bistability_10.csv": 10,
}

errors = []

for path, expected in expected_rows.items():
    if not path.exists():
        errors.append(f"missing: {path.relative_to(ROOT)}")
        continue

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(
            f"unreadable: {path.relative_to(ROOT)} ({exc})"
        )
        continue

    found = len(df)

    if found != expected:
        errors.append(
            f"{path.relative_to(ROOT)}: "
            f"expected {expected} rows, found {found}"
        )


long_path = (
    ROOT
    / "results/full/longitudinal"
    / "chunk_dynamics_nearest_stable_equilibrium.csv"
)

if not long_path.exists():
    errors.append(
        f"missing: {long_path.relative_to(ROOT)}"
    )


hier_dir = ROOT / "results/full/hierarchical"

if not hier_dir.exists() or not any(hier_dir.glob("*.csv")):
    errors.append("missing hierarchical CSV outputs")


if errors:
    print("RELEASE VALIDATION FAILED")

    for error in errors:
        print(" -", error)

    sys.exit(2)


print("RELEASE VALIDATION PASSED")
print()
print("Expected release populations confirmed:")
print(" - 800 model-selection chunks")
print(" - 738 inference-valid primary-dynamics chunks")
print(" - 800 sensitivity-analysis dynamics chunks")
print(" - 40 cohort reconstruction recordings")
print(" - 10 biological subjects")
