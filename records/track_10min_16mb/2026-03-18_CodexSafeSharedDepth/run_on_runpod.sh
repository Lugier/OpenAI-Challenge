#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"

cd "$REPO_ROOT"
python3 data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"

export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export RUN_ID="${RUN_ID:-codex_safe_shared_depth_$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/train.log}"
export ARTIFACT_PATH="${ARTIFACT_PATH:-$SCRIPT_DIR/model.compact.ptz}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" "$SCRIPT_DIR/train_gpt.py" --mode full

if [[ -n "${AUTHOR_NAME:-}" && -n "${GITHUB_ID:-}" ]]; then
  python3 "$SCRIPT_DIR/write_submission_json.py" \
    --record-dir "$SCRIPT_DIR" \
    --author "$AUTHOR_NAME" \
    --github-id "$GITHUB_ID"
fi
