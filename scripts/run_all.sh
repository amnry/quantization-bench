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

for CONFIG_ID in $CONFIG_IDS; do
  DONE_MARKER="results/${MODEL}/${CONFIG_ID}/DONE"
  if [[ -f "$DONE_MARKER" ]]; then
    echo "[run_all] ${CONFIG_ID} already DONE, skipping"
    continue
  fi

  echo "=== [run_all] ${CONFIG_ID} : quantize ==="
  python3 scripts/quantize.py --model "$MODEL" --config "$CONFIG_ID"

  echo "=== [run_all] ${CONFIG_ID} : eval ==="
  python3 scripts/eval.py --model "$MODEL" --config "$CONFIG_ID" --profile "$PROFILE"

  echo "=== [run_all] ${CONFIG_ID} : bench ==="
  python3 scripts/serve_bench.py --model "$MODEL" --config "$CONFIG_ID" --profile "$PROFILE" $DRY_RUN_FLAG

  touch "results/${MODEL}/${CONFIG_ID}/DONE"
  echo "[run_all] ${CONFIG_ID} DONE"

  if [[ "$PUSH" == true ]]; then
    echo "[run_all] committing + pushing results for ${CONFIG_ID}"
    git add "results/${MODEL}/${CONFIG_ID}"
    git commit -m "results(${MODEL}): ${CONFIG_ID} (${PROFILE})" || echo "[run_all] nothing to commit"
    git pull --rebase origin main || true
    git push origin main || echo "[run_all] WARNING: push failed for ${CONFIG_ID}, will retry next run"
  fi
done

echo "[run_all] all configs processed for model=${MODEL} profile=${PROFILE}"
