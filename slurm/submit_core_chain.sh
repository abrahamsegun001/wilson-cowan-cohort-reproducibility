#!/usr/bin/env bash
set -euo pipefail
CPU_ACCOUNT="${CPU_ACCOUNT:-}"
GPU_ACCOUNT="${GPU_ACCOUNT:-$CPU_ACCOUNT}"
if [[ -z "$CPU_ACCOUNT" || -z "$GPU_ACCOUNT" ]]; then
  echo "Set CPU_ACCOUNT and GPU_ACCOUNT before running." >&2
  echo "Example: CPU_ACCOUNT=def-example GPU_ACCOUNT=rpp-example VENV_PATH=\$HOME/gpinn_env bash slurm/submit_core_chain.sh" >&2
  exit 2
fi
mkdir -p logs
sel=$(sbatch --parsable --account="$CPU_ACCOUNT" slurm/model_selection_array.sh)
agg=$(sbatch --parsable --account="$CPU_ACCOUNT" --dependency=afterok:$sel slurm/model_selection_aggregate.sh)
dyn=$(sbatch --parsable --account="$CPU_ACCOUNT" --dependency=afterok:$agg slurm/dynamics_primary.sh)
gpu=$(sbatch --parsable --account="$GPU_ACCOUNT" --dependency=afterok:$agg slurm/gpinn_array.sh)
gpa=$(sbatch --parsable --account="$CPU_ACCOUNT" --dependency=afterok:$gpu slurm/gpinn_aggregate.sh)
lon=$(sbatch --parsable --account="$CPU_ACCOUNT" --dependency=afterok:$dyn slurm/longitudinal.sh)
hie=$(sbatch --parsable --account="$CPU_ACCOUNT" --dependency=afterok:$lon slurm/hierarchical.sh)
printf 'model selection array: %s\nmodel selection aggregate: %s\ndynamics: %s\nreconstruction array: %s\nreconstruction aggregate: %s\nlongitudinal: %s\nhierarchical: %s\n' "$sel" "$agg" "$dyn" "$gpu" "$gpa" "$lon" "$hie"
