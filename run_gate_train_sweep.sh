#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

TASK="${TASK:-STI}"
LANGUAGE_MODEL="${LANGUAGE_MODEL:-bert}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2}"
MAX_PEOPLE_NUM="${MAX_PEOPLE_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"

TXT_DIR="${TXT_DIR:-$HOME/datasets/mmsi/benchmark/youtube/transcripts/anonymized}"
TXT_LABELED_DIR="${TXT_LABELED_DIR:-$HOME/datasets/mmsi/benchmark/youtube/transcripts/anonymized_labeled}"
KEYPOINT_DIR="${KEYPOINT_DIR:-$HOME/datasets/mmsi/keypoints/keypoints_youtube}"
META_DIR="${META_DIR:-$HOME/datasets/mmsi/benchmark/youtube/meta_data}"
DATA_SPLIT_FILE="${DATA_SPLIT_FILE:-$HOME/datasets/mmsi/benchmark/youtube/data_split.json}"

OUT_ROOT="${OUT_ROOT:-runs/gate_train_sweep}"
RESUME_CKPT="${RESUME_CKPT:-runs/baseline_sti/checkpoints/model.pt}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LAMBDA_LIST="${LAMBDA_LIST:-0.002,0.005,0.01}"
ALPHA_LIST="${ALPHA_LIST:-0.05,0.1}"
TAU_LIST="${TAU_LIST:-0.1,0.12}"
MAX_TRIALS="${MAX_TRIALS:-0}"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
echo "trial_id,lambda,alpha,tau,acc,checkpoint" > "$SUMMARY_CSV"

best_acc="-1"
best_cfg=""
best_ckpt=""

trial_idx=0
run_count=0
IFS=',' read -r -a LAMBDAS <<< "$LAMBDA_LIST"
IFS=',' read -r -a ALPHAS <<< "$ALPHA_LIST"
IFS=',' read -r -a TAUS <<< "$TAU_LIST"

for lam in "${LAMBDAS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for tau in "${TAUS[@]}"; do
      if [[ "$MAX_TRIALS" -gt 0 && "$run_count" -ge "$MAX_TRIALS" ]]; then
        break 3
      fi
      trial_idx=$((trial_idx + 1))
      run_count=$((run_count + 1))
      trial_id=$(printf "trial_%03d" "$trial_idx")
      trial_dir="$OUT_ROOT/$trial_id"
      ckpt_dir="$trial_dir/checkpoints"
      eval_dir="$trial_dir/eval"
      mkdir -p "$trial_dir" "$ckpt_dir" "$eval_dir"

      echo "============================================================"
      echo "[TRAIN] $trial_id | lambda=$lam alpha=$alpha tau=$tau"

      train_epochs="$FINETUNE_EPOCHS"
      if [[ -n "${RESUME_CKPT:-}" && -f "$RESUME_CKPT" ]]; then
        resume_epoch=$("$PYTHON_BIN" -c "import torch;print(int(torch.load('$RESUME_CKPT', map_location='cpu').get('epoch', -1)))")
        train_epochs=$((resume_epoch + 1 + FINETUNE_EPOCHS))
      fi

      if ! "$PYTHON_BIN" train.py \
        --task "$TASK" \
        --txt_dir "$TXT_DIR" \
        --txt_labeled_dir "$TXT_LABELED_DIR" \
        --keypoint_dir "$KEYPOINT_DIR" \
        --meta_dir "$META_DIR" \
        --data_split_file "$DATA_SPLIT_FILE" \
        --checkpoint_save_dir "$ckpt_dir" \
        --language_model "$LANGUAGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --epochs "$train_epochs" \
        --workers "$WORKERS" \
        --context_length "$CONTEXT_LENGTH" \
        --max_people_num "$MAX_PEOPLE_NUM" \
        --resume "$RESUME_CKPT" \
        --gate_consistency_lambda "$lam" \
        --gate_consistency_alpha "$alpha" \
        --gate_consistency_tau "$tau" \
        --save_last \
        |& tee "$trial_dir/train.log"; then
        echo "[WARN] train failed for $trial_id, skipping."
        continue
      fi

      ckpt_file="$ckpt_dir/model.pt"
      if [[ ! -f "$ckpt_file" ]]; then
        echo "[WARN] checkpoint missing: $ckpt_file, skipping."
        continue
      fi

      echo "[EVAL] $trial_id"
      if ! RUN_BINARY=0 RUN_GRID_V2=0 OUT_DIR="$eval_dir" CHECKPOINT_FILE="$ckpt_file" PYTHON_BIN="$PYTHON_BIN" ./run_sti_test_all.sh \
        |& tee "$trial_dir/eval.log"; then
        echo "[WARN] eval failed for $trial_id, skipping."
        continue
      fi

      result_json="$eval_dir/results/baseline_multiclass_auto.json"
      if [[ ! -f "$result_json" ]]; then
        echo "[WARN] result json missing: $result_json, skipping."
        continue
      fi

      acc=$("$PYTHON_BIN" -c "import json;print(json.load(open('$result_json'))['multiclass_baseline']['acc'])")
      echo "$trial_id,$lam,$alpha,$tau,$acc,$ckpt_file" >> "$SUMMARY_CSV"
      echo "[DONE] $trial_id acc=$acc"

      better=$("$PYTHON_BIN" -c "print(1 if float('$acc')>float('$best_acc') else 0)")
      if [[ "$better" == "1" ]]; then
        best_acc="$acc"
        best_cfg="lambda=$lam alpha=$alpha tau=$tau"
        best_ckpt="$ckpt_file"
      fi
    done
  done
done

echo "============================================================"
echo "[SUMMARY] saved: $SUMMARY_CSV"
echo "[BEST] acc=$best_acc | $best_cfg"
echo "[BEST] checkpoint=$best_ckpt"
