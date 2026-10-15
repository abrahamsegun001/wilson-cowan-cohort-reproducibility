#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./submit_pipeline.sh {model-selection|model-selection-aggregate|dynamics|reconstruct|reconstruct-aggregate|longitudinal|hierarchical|core-chain} [sbatch options...]"
  exit 2
fi

stage="$1"; shift
case "$stage" in
  model-selection)          job="slurm/model_selection_array.sh" ;;
  model-selection-aggregate) job="slurm/model_selection_aggregate.sh" ;;
  dynamics)                 job="slurm/dynamics_primary.sh" ;;
  reconstruct)              job="slurm/gpinn_array.sh" ;;
  reconstruct-aggregate)    job="slurm/gpinn_aggregate.sh" ;;
  longitudinal)             job="slurm/longitudinal.sh" ;;
  hierarchical)             job="slurm/hierarchical.sh" ;;
  core-chain)               exec bash slurm/submit_core_chain.sh "$@" ;;
  *) echo "Unknown stage: $stage"; exit 2 ;;
esac
exec sbatch "$@" "$job"
