#!/bin/bash
#SBATCH --job-name=mmsi_film_sweep
#SBATCH --partition=gpuL
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"
echo "[INFO] Submit dir: ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run from the directory where sbatch was submitted.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate conda environment (matches your current setup).
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate mmsi
else
  echo "[WARN] conda.sh not found. Using current shell Python."
fi

echo "[INFO] Python: $(which python)"
python --version || true

# Default experiment configuration (can be overridden with sbatch --export=...)
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-mmsi-fiml-centerloss}"
export EPOCHS="${EPOCHS:-200}"
export EPOCHS_WARMUP="${EPOCHS_WARMUP:-10}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export WORKERS="${WORKERS:-0}"
export CONFIGS="${CONFIGS:-v4:4:0:0,f1:0:1:0,f4:0:4:0,v1_center:1:0:1,v2f2:2:2:0,v2f2_center:2:2:1}"

# Dataset defaults (YouTube)
export TXT_DIR="${TXT_DIR:-/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/datasets/mmsi/benchmark/youtube/transcripts/anonymized}"
export TXT_LABELED_DIR="${TXT_LABELED_DIR:-/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/datasets/mmsi/benchmark/youtube/transcripts/anonymized_labeled}"
export KEYPOINT_DIR="${KEYPOINT_DIR:-/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/datasets/mmsi/keypoints/keypoints_youtube}"
export META_DIR="${META_DIR:-/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/datasets/mmsi/benchmark/youtube/meta_data}"
export DATA_SPLIT_FILE="${DATA_SPLIT_FILE:-/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/datasets/mmsi/benchmark/youtube/data_split.json}"

# Output directory for this batch. Includes job id to avoid collisions.
export OUT_ROOT="${OUT_ROOT:-runs/film_ablation_sweep_youtube_part2_e200_slurm_${SLURM_JOB_ID:-manual}}"

mkdir -p "$OUT_ROOT"

echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] CONFIGS=$CONFIGS"
echo "[INFO] W&B mode=$WANDB_MODE project=$WANDB_PROJECT"

# Run the sweep (training + auto eval + summary.csv)
bash run_film_ablation_sweep.sh |& tee "$OUT_ROOT/slurm_driver.log"

echo "[INFO] Sweep finished: $(date)"
echo "[INFO] Generating markdown report..."

python scripts/generate_ablation_report.py \
  --summary_csv "$OUT_ROOT/summary.csv" \
  --output_md "$OUT_ROOT/ablation_report.md" || true

echo "[INFO] End: $(date)"
