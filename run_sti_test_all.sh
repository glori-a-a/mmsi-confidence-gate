#!/usr/bin/env bash
set -euo pipefail

# One-shot evaluation script:
# 1) multi-class baseline
# 2) confidence gate for K=0..5

PYTHON_BIN="${PYTHON_BIN:-python}"

TASK="${TASK:-STI}"
LANGUAGE_MODEL="${LANGUAGE_MODEL:-bert}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2}"
MAX_PEOPLE_NUM="${MAX_PEOPLE_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-0}"

ALPHA="${ALPHA:-0.2}"
TAU="${TAU:-0.12}"
RUN_GRID_V2="${RUN_GRID_V2:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_BINARY="${RUN_BINARY:-1}"
GRID_ALPHAS="${GRID_ALPHAS:-0.01,0.03,0.05,0.1,0.2}"
GRID_TAUS="${GRID_TAUS:-0.02,0.05,0.08,0.1,0.15}"
GRID_POLICIES="${GRID_POLICIES:-smoothed,safe_flip,safe_gate,raw_on_confident}"
GRID_GAP_RESETS="${GRID_GAP_RESETS:--1,5,10,20,30}"

TXT_DIR="${TXT_DIR:-$HOME/datasets/mmsi/benchmark/youtube/transcripts/anonymized}"
TXT_LABELED_DIR="${TXT_LABELED_DIR:-$HOME/datasets/mmsi/benchmark/youtube/transcripts/anonymized_labeled}"
KEYPOINT_DIR="${KEYPOINT_DIR:-$HOME/datasets/mmsi/keypoints/keypoints_youtube}"
META_DIR="${META_DIR:-$HOME/datasets/mmsi/benchmark/youtube/meta_data}"
DATA_SPLIT_FILE="${DATA_SPLIT_FILE:-$HOME/datasets/mmsi/benchmark/youtube/data_split.json}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-runs/baseline_sti/checkpoints/model.pt}"

OUT_DIR="${OUT_DIR:-runs/sti_test}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
RESULT_DIR="${RESULT_DIR:-$OUT_DIR/results}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

COMMON_ARGS=(
  --task "$TASK"
  --txt_dir "$TXT_DIR"
  --txt_labeled_dir "$TXT_LABELED_DIR"
  --keypoint_dir "$KEYPOINT_DIR"
  --meta_dir "$META_DIR"
  --data_split_file "$DATA_SPLIT_FILE"
  --checkpoint_file "$CHECKPOINT_FILE"
  --language_model "$LANGUAGE_MODEL"
  --context_length "$CONTEXT_LENGTH"
  --max_people_num "$MAX_PEOPLE_NUM"
  --batch_size "$BATCH_SIZE"
  --workers "$WORKERS"
)

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "[1/3] Running multi-class baseline..."
  "$PYTHON_BIN" test.py \
    "${COMMON_ARGS[@]}" \
    --save_path "$RESULT_DIR/baseline_multiclass_auto.json" \
    |& tee "$LOG_DIR/test_baseline_auto.log"
fi

if [[ "$RUN_BINARY" == "1" ]]; then
  echo "[2/3] Running confidence gate for K=0..5..."
  for K in 0 1 2 3 4 5; do
    echo "  - K=$K"
    "$PYTHON_BIN" test.py \
      "${COMMON_ARGS[@]}" \
      --use_confidence_gate \
      --binary_target_label "$K" \
      --alpha "$ALPHA" \
      --tau "$TAU" \
      --save_path "$RESULT_DIR/gate_K${K}_auto.json" \
      |& tee "$LOG_DIR/test_gate_K${K}_auto.log"
  done
fi

if [[ "$RUN_GRID_V2" == "1" ]]; then
  echo "[3/3] Running multi-class gate grid search (v2 strategies)..."
  "$PYTHON_BIN" test.py \
    "${COMMON_ARGS[@]}" \
    --grid_search_gate \
    --grid_alphas "$GRID_ALPHAS" \
    --grid_taus "$GRID_TAUS" \
    --grid_gate_policies "$GRID_POLICIES" \
    --grid_gap_resets="$GRID_GAP_RESETS" \
    --save_path "$RESULT_DIR/multiclass_gate_grid_v2_full_auto.json" \
    |& tee "$LOG_DIR/test_multiclass_gate_grid_v2_full_auto.log"
fi

echo "[DONE] Results: $RESULT_DIR"
echo "[DONE] Logs:    $LOG_DIR"
