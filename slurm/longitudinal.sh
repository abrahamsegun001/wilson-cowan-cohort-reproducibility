#!/usr/bin/env bash
#SBATCH --job-name=wc_longitudinal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wc_longitudinal_%j.out
#SBATCH --error=logs/wc_longitudinal_%j.err
set -euo pipefail
module purge
module load StdEnv/2023 python/3.10 gcc
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
python src/longitudinal.py \
  --input outputs/cohort_dynamics_40_primary/cohort_dynamics_800.csv \
  --output-dir outputs/cohort_longitudinal_40_primary
