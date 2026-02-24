#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

TASK="${TASK:-STI}"
LANGUAGE_MODEL="${LANGUAGE_MODEL:-bert}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-5}"
MAX_PEOPLE_NUM="${MAX_PEOPLE_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
EPOCHS="${EPOCHS:-200}"
EPOCHS_WARMUP="${EPOCHS_WARMUP:-10}"

TXT_DIR="${TXT_DIR:-enter_the_path}"
TXT_LABELED_DIR="${TXT_LABELED_DIR:-enter_the_path}"
KEYPOINT_DIR="${KEYPOINT_DIR:-enter_the_path}"
META_DIR="${META_DIR:-enter_the_path}"
DATA_SPLIT_FILE="${DATA_SPLIT_FILE:-enter_the_path}"

OUT_ROOT="${OUT_ROOT:-runs/film_ablation_sweep}"

# WandB (optional)
USE_WANDB="${USE_WANDB:-0}"               # 0 or 1
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-fiml-centerloss}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"        # online/offline/disabled
WANDB_TAGS="${WANDB_TAGS:-mmsi,ablation}"

# Center loss defaults for configurations that enable it
CENTER_LOSS_LAMBDA="${CENTER_LOSS_LAMBDA:-0.01}"
CENTER_LOSS_LR="${CENTER_LOSS_LR:-1e-3}"

# Grad-CAM probe/reg (optional, usually keep disabled for sweep first)
ENABLE_GRADCAM="${ENABLE_GRADCAM:-0}"
GRADCAM_TARGET="${GRADCAM_TARGET:-vision_post_transformer}"
GRADCAM_LOSS_LAMBDA="${GRADCAM_LOSS_LAMBDA:-0.0}"
GRADCAM_LOSS_MODE="${GRADCAM_LOSS_MODE:-entropy}"

# Auto evaluation after each training run (recommended for report prep)
RUN_EVAL="${RUN_EVAL:-1}"                 # 0 or 1
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
EVAL_SAVE_GRADCAM="${EVAL_SAVE_GRADCAM:-0}"   # 0 or 1
EVAL_GRADCAM_TARGET="${EVAL_GRADCAM_TARGET:-vision_post_transformer}"
EVAL_GRADCAM_MAX_BATCHES="${EVAL_GRADCAM_MAX_BATCHES:-10}"

# Format: name:vision_layers:fusion_layers:use_center_loss(0/1)
CONFIGS="${CONFIGS:-baseline:0:0:0,v1:1:0:0,v4:4:0:0,f1:0:1:0,f4:0:4:0,v2f2:2:2:0,v1_center:1:0:1,v2f2_center:2:2:1}"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
echo "trial_id,vision_film_layers,fusion_film_layers,use_center_loss,checkpoint_dir,eval_json,test_acc" > "$SUMMARY_CSV"

IFS=',' read -r -a CFGS <<< "$CONFIGS"

trial_idx=0
for cfg in "${CFGS[@]}"; do
  IFS=':' read -r cfg_name v_layers f_layers use_center <<< "$cfg"
  trial_idx=$((trial_idx + 1))
  trial_id=$(printf "%03d_%s" "$trial_idx" "$cfg_name")
  trial_dir="$OUT_ROOT/$trial_id"
  ckpt_dir="$trial_dir/checkpoints"
  eval_dir="$trial_dir/eval"
  mkdir -p "$trial_dir" "$ckpt_dir" "$eval_dir"

  echo "============================================================"
  echo "[TRAIN] $trial_id | V=$v_layers F=$f_layers Center=$use_center"

  cmd=(
    "$PYTHON_BIN" train.py
    --model_name "$trial_id"
    --task "$TASK"
    --txt_dir "$TXT_DIR"
    --txt_labeled_dir "$TXT_LABELED_DIR"
    --keypoint_dir "$KEYPOINT_DIR"
    --meta_dir "$META_DIR"
    --data_split_file "$DATA_SPLIT_FILE"
    --checkpoint_save_dir "$ckpt_dir"
    --language_model "$LANGUAGE_MODEL"
    --max_people_num "$MAX_PEOPLE_NUM"
    --context_length "$CONTEXT_LENGTH"
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --epochs "$EPOCHS"
    --epochs_warmup "$EPOCHS_WARMUP"
    --workers "$WORKERS"
    --film_vision_layers "$v_layers"
    --film_fusion_layers "$f_layers"
    --save_last
  )

  if [[ "$v_layers" -gt 0 || "$f_layers" -gt 0 ]]; then
    cmd+=(--use_film_fusion)
  fi

  if [[ "$use_center" == "1" ]]; then
    cmd+=(--use_center_loss --center_loss_lambda "$CENTER_LOSS_LAMBDA" --center_loss_lr "$CENTER_LOSS_LR")
  fi

  if [[ "$ENABLE_GRADCAM" == "1" || "$GRADCAM_LOSS_LAMBDA" != "0.0" ]]; then
    cmd+=(--enable_gradcam --gradcam_target "$GRADCAM_TARGET" --gradcam_loss_lambda "$GRADCAM_LOSS_LAMBDA" --gradcam_loss_mode "$GRADCAM_LOSS_MODE")
  fi

  if [[ "$USE_WANDB" == "1" ]]; then
    cmd+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_name "$trial_id" --wandb_tags "$WANDB_TAGS")
    if [[ -n "$WANDB_ENTITY" ]]; then
      cmd+=(--wandb_entity "$WANDB_ENTITY")
    fi
  fi

  if ! "${cmd[@]}" |& tee "$trial_dir/train.log"; then
    echo "[WARN] training failed: $trial_id"
    continue
  fi

  eval_json=""
  test_acc=""
  if [[ "$RUN_EVAL" == "1" ]]; then
    ckpt_file="$ckpt_dir/model.pt"
    if [[ ! -f "$ckpt_file" ]]; then
      echo "[WARN] missing checkpoint for eval: $ckpt_file"
    else
      eval_json="$eval_dir/results.json"
      eval_cmd=(
        "$PYTHON_BIN" test.py
        --model_name "$trial_id"
        --task "$TASK"
        --txt_dir "$TXT_DIR"
        --txt_labeled_dir "$TXT_LABELED_DIR"
        --keypoint_dir "$KEYPOINT_DIR"
        --meta_dir "$META_DIR"
        --data_split_file "$DATA_SPLIT_FILE"
        --checkpoint_file "$ckpt_file"
        --language_model "$LANGUAGE_MODEL"
        --max_people_num "$MAX_PEOPLE_NUM"
        --context_length "$CONTEXT_LENGTH"
        --batch_size "$EVAL_BATCH_SIZE"
        --workers "$WORKERS"
        --save_path "$eval_json"
      )

      if [[ "$EVAL_SAVE_GRADCAM" == "1" ]]; then
        eval_cmd+=(
          --enable_gradcam
          --export_gradcam
          --gradcam_target "$EVAL_GRADCAM_TARGET"
          --gradcam_export_path "$eval_dir/gradcam_tokens.json"
          --gradcam_export_max_batches "$EVAL_GRADCAM_MAX_BATCHES"
        )
      fi

      echo "[EVAL] $trial_id"
      if ! "${eval_cmd[@]}" |& tee "$trial_dir/eval.log"; then
        echo "[WARN] evaluation failed: $trial_id"
      elif [[ -f "$eval_json" ]]; then
        test_acc=$("$PYTHON_BIN" -c "import json; print(json.load(open('$eval_json'))['multiclass_baseline']['acc'])")
        echo "[EVAL] acc=$test_acc"
      fi
    fi
  fi

  echo "$trial_id,$v_layers,$f_layers,$use_center,$ckpt_dir,$eval_json,$test_acc" >> "$SUMMARY_CSV"
done

echo "============================================================"
echo "[DONE] Sweep completed."
echo "[SUMMARY] $SUMMARY_CSV"
