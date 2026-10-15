#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-$HOME/scratch/gpinn_project}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/results/full"
mkdir -p "$DEST"/{model_selection,dynamics,longitudinal,hierarchical,reconstruction}

copy_required() {
  local src="$1"; local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "MISSING: $src" >&2
    return 1
  fi
  cp "$src" "$dst"
  echo "COPIED: $src"
}

copy_optional_glob() {
  local pattern="$1"; local dst="$2"
  shopt -s nullglob
  local files=( $pattern )
  if (( ${#files[@]} )); then cp "${files[@]}" "$dst"/; fi
  shopt -u nullglob
}

status=0
copy_required "$SRC_ROOT/outputs/cohort_model_selection_40/cohort_model_selection_800.csv" "$DEST/model_selection/" || status=1
copy_optional_glob "$SRC_ROOT/outputs/cohort_model_selection_40/*.json" "$DEST/model_selection"
copy_optional_glob "$SRC_ROOT/outputs/cohort_model_selection_40/*summary*.csv" "$DEST/model_selection"

copy_required "$SRC_ROOT/outputs/cohort_dynamics_40_primary/cohort_dynamics_800.csv" "$DEST/dynamics/" || status=1
copy_optional_glob "$SRC_ROOT/outputs/cohort_dynamics_40_primary/*.json" "$DEST/dynamics"
copy_optional_glob "$SRC_ROOT/outputs/cohort_dynamics_40_primary/*.csv" "$DEST/dynamics"

copy_required "$SRC_ROOT/outputs/cohort_longitudinal_40_primary/chunk_dynamics_nearest_stable_equilibrium.csv" "$DEST/longitudinal/" || status=1
copy_optional_glob "$SRC_ROOT/outputs/cohort_longitudinal_40_primary/*.csv" "$DEST/longitudinal"
copy_optional_glob "$SRC_ROOT/outputs/cohort_longitudinal_40_primary/*.json" "$DEST/longitudinal"

copy_optional_glob "$SRC_ROOT/outputs/cohort_hierarchical_40_primary/*.csv" "$DEST/hierarchical"
copy_optional_glob "$SRC_ROOT/outputs/cohort_hierarchical_40_primary/*.json" "$DEST/hierarchical"

copy_required "$SRC_ROOT/outputs/cohort_gpinn_reconstruction_40/cohort_gpinn_reconstruction_metrics_40.csv" "$DEST/reconstruction/" || status=1
copy_optional_glob "$SRC_ROOT/outputs/cohort_gpinn_reconstruction_40/*.csv" "$DEST/reconstruction"
copy_optional_glob "$SRC_ROOT/outputs/cohort_gpinn_reconstruction_40/*.json" "$DEST/reconstruction"

if (( status != 0 )); then
  echo "Release collection is incomplete. See missing files above." >&2
  exit 2
fi

echo "Release tables collected under: $DEST"
