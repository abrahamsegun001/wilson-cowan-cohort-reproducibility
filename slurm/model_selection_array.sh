#!/usr/bin/env bash
#SBATCH --job-name=wc_select40
#SBATCH --array=0-39%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/wc_select40_%A_%a.out
#SBATCH --error=logs/wc_select40_%A_%a.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.10 gcc
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/cohort_model_selection_40/per_recording
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
python src/model_selection.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --manifest data/preprocessing_manifest_40.csv \
  --data-dir data/processed_64_linear \
  --output-dir outputs/cohort_model_selection_40
