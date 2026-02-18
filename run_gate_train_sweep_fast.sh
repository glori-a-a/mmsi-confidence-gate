#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHON_BIN="$PYTHON_BIN" \
OUT_ROOT="${OUT_ROOT:-runs/gate_train_sweep_fast}" \
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1}" \
MAX_TRIALS="${MAX_TRIALS:-4}" \
LAMBDA_LIST="${LAMBDA_LIST:-0.002,0.005}" \
ALPHA_LIST="${ALPHA_LIST:-0.05,0.1}" \
TAU_LIST="${TAU_LIST:-0.1}" \
./run_gate_train_sweep.sh
