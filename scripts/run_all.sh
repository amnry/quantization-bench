#!/usr/bin/env bash
# Orchestrate quantize -> eval -> bench for every config in configs/matrix.yaml.
# Skips a config if results/<model>/<config>/DONE already exists (resume-safe).
# With --push: commits+pushes results after EACH config finishes, so a mid-run
# crash never costs already-completed work (PLAN.md phase 4 requirement).
#
# Usage:
#   ./scripts/run_all.sh --model qwen2.5-0.5b --profile local
#   ./scripts/run_all.sh --model qwen2.5-0.5b --profile smoke
#   ./scripts/run_all.sh --model qwen2.5-7b   --profile full --push
set -euo pipefail

MODEL=""
PROFILE=""
PUSH=false
DRY_RUN_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --push) PUSH=true; shift ;;
    --dry-run) DRY_RUN_FLAG="--dry-run"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$PROFILE" ]]; then
  echo "usage: $0 --model <name> --profile <local|smoke|full> [--push] [--dry-run]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG_IDS=$(python3 - <<'PY'
import sys
sys.path.insert(0, "scripts")
from common import all_config_ids
print("\n".join(all_config_ids()))
PY
)

# Runs quantize/eval/bench for one config. Each step is `||`-guarded so a
# failure returns 1 here instead of tripping `set -e` at the top level —
# callers must invoke this as the condition of an if/while (see below).
run_config() {
  local model="$1" config_id="$2" profile="$3"
  echo "=== [run_all] ${config_id} : quantize ==="
  python3 scripts/quantize.py --model "$model" --config "$config_id" || return 1
  echo "=== [run_all] ${config_id} : eval ==="
  python3 scripts/eval.py --model "$model" --config "$config_id" --profile "$profile" || return 1
  echo "=== [run_all] ${config_id} : bench ==="
  python3 scripts/serve_bench.py --model "$model" --config "$config_id" --profile "$profile" $DRY_RUN_FLAG || return 1
}

for CONFIG_ID in $CONFIG_IDS; do
  RESULT_DIR="results/${MODEL}/${CONFIG_ID}"
  DONE_MARKER="${RESULT_DIR}/DONE"
  if [[ -f "$DONE_MARKER" ]]; then
    echo "[run_all] ${CONFIG_ID} already DONE, skipping"
    continue
  fi

  mkdir -p "$RESULT_DIR"
  rm -f "${RESULT_DIR}/FAILED"
  STEP_LOG="${RESULT_DIR}/step.log"

  # `if pipeline; then` is exempt from `set -e` regardless of pipefail, so a
  # failure inside run_config lands in the else branch instead of killing
  # the whole script — one bad config no longer aborts the run.
  if run_config "$MODEL" "$CONFIG_ID" "$PROFILE" 2>&1 | tee "$STEP_LOG"; then
    rm -f "$STEP_LOG"
    touch "$DONE_MARKER"
    echo "[run_all] ${CONFIG_ID} DONE"
    COMMIT_MSG="results(${MODEL}): ${CONFIG_ID} (${PROFILE})"
  else
    tail -n 100 "$STEP_LOG" > "${RESULT_DIR}/FAILED"
    rm -f "$STEP_LOG"
    echo "[run_all] ${CONFIG_ID} FAILED, see ${RESULT_DIR}/FAILED — continuing to next config"
    COMMIT_MSG="results(${MODEL}): ${CONFIG_ID} FAILED (${PROFILE})"
  fi

  if [[ "$PUSH" == true ]]; then
    echo "[run_all] committing + pushing results for ${CONFIG_ID}"
    git add "$RESULT_DIR"
    git commit -m "$COMMIT_MSG" || echo "[run_all] nothing to commit"
    git pull --rebase origin main || true
    git push origin main || echo "[run_all] WARNING: push failed for ${CONFIG_ID}, will retry next run"
  fi
done

echo "[run_all] all configs processed for model=${MODEL} profile=${PROFILE}"
