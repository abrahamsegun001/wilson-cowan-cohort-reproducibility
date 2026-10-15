#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-$HOME/scratch/gpinn_project}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/results/full"

mkdir -p \
  "$DEST/model_selection" \
  "$DEST/dynamics/primary_valid" \
  "$DEST/dynamics/sensitivity_all800" \
  "$DEST/longitudinal" \
  "$DEST/hierarchical" \
  "$DEST/reconstruction"

copy_required() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "$src" ]]; then
    echo "MISSING: $src" >&2
    return 1
  fi

  cp "$src" "$dst"
  echo "COPIED: $src"
}

copy_optional_glob() {
  local pattern="$1"
  local dst="$2"

  shopt -s nullglob
  local files=( $pattern )

  if (( ${#files[@]} )); then
    cp "${files[@]}" "$dst"/
  fi

  shopt -u nullglob
}

status=0

# ------------------------------------------------------------
# Model selection: all 800 chunks
# ------------------------------------------------------------

copy_required \
  "$SRC_ROOT/outputs/cohort_model_selection_40/cohort_model_selection_800.csv" \
  "$DEST/model_selection/" || status=1

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_model_selection_40/*.json" \
  "$DEST/model_selection"

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_model_selection_40/*summary*.csv" \
  "$DEST/model_selection"

# ------------------------------------------------------------
# Primary dynamics: 738 inference-valid chunks
# ------------------------------------------------------------

copy_required \
  "$SRC_ROOT/outputs/cohort_dynamics_40_primary/cohort_dynamics_800.csv" \
  "$DEST/dynamics/primary_valid/cohort_dynamics_primary_valid_738.csv" || status=1

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_dynamics_40_primary/*.json" \
  "$DEST/dynamics/primary_valid"

shopt -s nullglob
for f in "$SRC_ROOT"/outputs/cohort_dynamics_40_primary/*.csv; do
  [[ "$(basename "$f")" == "cohort_dynamics_800.csv" ]] && continue
  cp "$f" "$DEST/dynamics/primary_valid/"
done
shopt -u nullglob

# ------------------------------------------------------------
# Sensitivity dynamics: all 800 chunks
# ------------------------------------------------------------

copy_required \
  "$SRC_ROOT/outputs/cohort_dynamics_40_sensitivity_all800/cohort_dynamics_800.csv" \
  "$DEST/dynamics/sensitivity_all800/cohort_dynamics_sensitivity_all800.csv" || status=1

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_dynamics_40_sensitivity_all800/*.json" \
  "$DEST/dynamics/sensitivity_all800"

shopt -s nullglob
for f in "$SRC_ROOT"/outputs/cohort_dynamics_40_sensitivity_all800/*.csv; do
  [[ "$(basename "$f")" == "cohort_dynamics_800.csv" ]] && continue
  cp "$f" "$DEST/dynamics/sensitivity_all800/"
done
shopt -u nullglob

# ------------------------------------------------------------
# Longitudinal analysis
# ------------------------------------------------------------

copy_required \
  "$SRC_ROOT/outputs/cohort_longitudinal_40_primary/chunk_dynamics_nearest_stable_equilibrium.csv" \
  "$DEST/longitudinal/" || status=1

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_longitudinal_40_primary/*.csv" \
  "$DEST/longitudinal"

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_longitudinal_40_primary/*.json" \
  "$DEST/longitudinal"

# ------------------------------------------------------------
# Hierarchical analysis
# ------------------------------------------------------------

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_hierarchical_40_primary/*.csv" \
  "$DEST/hierarchical"

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_hierarchical_40_primary/*.json" \
  "$DEST/hierarchical"

# ------------------------------------------------------------
# Cohort gPINN reconstruction
# ------------------------------------------------------------

copy_required \
  "$SRC_ROOT/outputs/cohort_gpinn_reconstruction_40/cohort_gpinn_reconstruction_metrics_40.csv" \
  "$DEST/reconstruction/" || status=1

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_gpinn_reconstruction_40/*.csv" \
  "$DEST/reconstruction"

copy_optional_glob \
  "$SRC_ROOT/outputs/cohort_gpinn_reconstruction_40/*.json" \
  "$DEST/reconstruction"

if (( status != 0 )); then
  echo "Release collection is incomplete. See missing files above." >&2
  exit 2
fi

echo
echo "Release tables collected under:"
echo "$DEST"
