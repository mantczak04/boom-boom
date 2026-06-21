#!/usr/bin/env bash
# Training run: correlated 9×9, state+prob (see .ai/plan.md Appendix A).
#
# Full run (default, ~3h CPU):  ./scripts/train.sh
# Fast run (~45min CPU):        FAST=1 ./scripts/train.sh
# Resume from checkpoint:       RESUME_CHECKPOINT=checkpoints/.../model.pt FAST=1 ./scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-correlated-9x9-state+prob}"
mkdir -p "runs/${RUN_ID}"

if [[ "${FAST:-0}" == "1" ]]; then
  T_MAX=50000
  LEARN_START=800
  EVAL_INTERVAL=5000
  EVAL_EPISODES=50
else
  T_MAX=200000
  LEARN_START=1600
  EVAL_INTERVAL=10000
  EVAL_EPISODES=100
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--model "$RESUME_CHECKPOINT")
fi

uv run python -m rainbow.main \
  --id "$RUN_ID" \
  --seed 0 \
  --board-width 9 \
  --board-height 9 \
  --distribution correlated \
  --obs-mode state+prob \
  --T-max "$T_MAX" \
  --learn-start "$LEARN_START" \
  --memory-capacity 100000 \
  --replay-frequency 1 \
  --multi-step 3 \
  --target-update 2000 \
  --batch-size 32 \
  --hidden-size 256 \
  --learning-rate 1e-4 \
  --discount 0.99 \
  --atoms 51 \
  --noisy-std 0.1 \
  --V-min -5 \
  --V-max 40 \
  --evaluation-interval "$EVAL_INTERVAL" \
  --evaluation-episodes "$EVAL_EPISODES" \
  --disable-cuda \
  "${EXTRA_ARGS[@]}"
