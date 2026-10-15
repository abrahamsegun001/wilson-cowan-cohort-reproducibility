#!/usr/bin/env bash
#SBATCH --job-name=wc_recon40
#SBATCH --array=0-39%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_recon40_%A_%a.out
#SBATCH --error=logs/wc_recon40_%A_%a.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.10 gcc cuda
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/cohort_gpinn_reconstruction_40
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/reconstruction.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --manifest data/preprocessing_manifest_40.csv \
  --cohort-csv outputs/cohort_model_selection_40/cohort_model_selection_800.csv \
  --data-dir data/processed_64_linear \
  --output-dir outputs/cohort_gpinn_reconstruction_40 \
  --target-chunk 10 \
  --seed 42
