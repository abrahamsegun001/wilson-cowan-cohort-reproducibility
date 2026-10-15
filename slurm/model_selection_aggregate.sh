#!/usr/bin/env bash
#SBATCH --job-name=wc_select_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_select_agg_%j.out
#SBATCH --error=logs/wc_select_agg_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.10 gcc
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
python src/model_selection.py \
  --aggregate \
  --manifest data/preprocessing_manifest_40.csv \
  --data-dir data/processed_64_linear \
  --output-dir outputs/cohort_model_selection_40
