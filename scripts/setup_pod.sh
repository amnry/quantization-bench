#!/usr/bin/env bash
# Bootstrap a rented GPU pod (RunPod or similar) to run this repo.
#
# Smoke pod (first time, unpinned):
#   ./scripts/setup_pod.sh
# Real pod (after smoke passed and requirements-lock.txt was pushed):
#   ./scripts/setup_pod.sh --locked
set -euo pipefail

LOCKED=false
[[ "${1:-}" == "--locked" ]] && LOCKED=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if [[ "$LOCKED" == true ]]; then
  if [[ ! -f requirements-lock.txt ]]; then
    echo "[setup_pod] --locked requested but requirements-lock.txt is missing" >&2
    exit 1
  fi
  echo "[setup_pod] installing pinned requirements-lock.txt"
  pip install -r requirements-lock.txt
else
  echo "[setup_pod] installing unpinned requirements-pod.txt (smoke test)"
  pip install -r requirements-pod.txt
fi

# GitHub push access for per-config result commits (PLAN.md phase 4).
# Set GITHUB_TOKEN as a pod environment variable/secret before running this script.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git config user.email "amanarya1910@gmail.com"
  git config user.name "amnry"
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/amnry/quantization-bench.git"
  echo "[setup_pod] git remote configured with token auth"
else
  echo "[setup_pod] WARNING: GITHUB_TOKEN not set — per-config pushes in run_all.sh will fail" >&2
fi

echo "[setup_pod] done. Activate with: source .venv/bin/activate"
echo "[setup_pod] then: ./scripts/run_all.sh --model <name> --profile <smoke|full> --push"
