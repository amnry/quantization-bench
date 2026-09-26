#!/usr/bin/env python3
"""Launch `vllm serve` for one config, sweep `vllm bench serve` over workloads x concurrency,
scrape the KV cache size out of the startup log, then tear the server down.

Local (Mac): --dry-run only — vLLM doesn't serve on macOS, so this parses a canned fixture
log/result instead of a real server, just to prove the orchestration + parsing logic.
Pod (GPU):   real run.

Usage:
    python scripts/serve_bench.py --model qwen2.5-0.5b --config bf16 --profile local --dry-run
    python scripts/serve_bench.py --model qwen2.5-7b   --config w4a16 --profile full
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import RunPaths, config_meta, load_profile, load_quant_config, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SERVER_PORT = 8000
KV_SIZE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE)


def start_server(checkpoint_dir: str, quant_cfg: dict, profile: dict, log_path: Path) -> subprocess.Popen:
    server_cfg = profile["server"]
    cmd = [
        "vllm", "serve", checkpoint_dir,
        "--port", str(SERVER_PORT),
        "--max-model-len", str(server_cfg["max_model_len"]),
        "--gpu-memory-utilization", str(server_cfg["gpu_memory_utilization"]),
        "--max-num-seqs", str(server_cfg["max_num_seqs"]),
        "--no-enable-prefix-caching",
        *quant_cfg.get("vllm_extra_args", []),
    ]
    print(f"[serve_bench] launching: {' '.join(cmd)}")
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def wait_for_health(timeout_s: int = 900) -> bool:
    url = f"http://localhost:{SERVER_PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def scrape_kv_cache_tokens(log_path: Path) -> int | None:
    text = log_path.read_text(errors="ignore")
    m = KV_SIZE_RE.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def run_bench_workload(checkpoint_dir: str, workload: str, wl_cfg: dict, concurrency: int,
                        num_prompts: int, out_path: Path) -> None:
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", checkpoint_dir,
        "--port", str(SERVER_PORT),
        "--dataset-name", "random",
        "--random-input-len", str(wl_cfg["input_len"]),
        "--random-output-len", str(wl_cfg["output_len"]),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--ignore-eos",
        "--seed", "42",
        "--save-result",
        "--result-filename", str(out_path),
    ]
    print(f"[serve_bench] {workload} c={concurrency}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def dry_run(paths: RunPaths, profile: dict) -> None:
    print("[serve_bench] --dry-run: no server launched, parsing fixtures instead")
    fixture_log = FIXTURES_DIR / "sample_vllm_startup.log"
    fixture_bench = FIXTURES_DIR / "sample_bench_result.json"

    kv_tokens = scrape_kv_cache_tokens(fixture_log)
    print(f"[serve_bench] (fixture) KV cache size parsed: {kv_tokens} tokens")

    for workload, wl_cfg in profile["bench"]["workloads"].items():
        for concurrency in wl_cfg["concurrency"]:
            out_path = paths.bench_path(workload, concurrency)
            data = json.loads(fixture_bench.read_text())
            data["_dry_run"] = True
            data["workload"] = workload
            data["concurrency"] = concurrency
            write_json(out_path, data)
    meta = config_meta(paths.model_name, paths.config_id)
    meta["kv_cache_tokens"] = kv_tokens
    meta["dry_run"] = True
    write_json(paths.results_dir / "bench_meta.json", meta)
    print(f"[serve_bench] dry-run wrote fixture-derived results to {paths.results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True, choices=["local", "smoke", "full"])
    ap.add_argument("--dry-run", action="store_true", help="don't launch a real server (macOS/local dev)")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    quant_cfg = load_quant_config(args.config)
    paths = RunPaths(model_name=args.model, config_id=args.config)

    if not profile["bench"]["enabled"] and not args.dry_run:
        print(f"[serve_bench] profile '{args.profile}' has bench.enabled=false, nothing to do")
        return

    if args.dry_run or not shutil.which("vllm"):
        if not args.dry_run:
            print("[serve_bench] 'vllm' not found on PATH (expected on macOS) — forcing --dry-run")
        dry_run(paths, profile)
        return

    log_path = paths.results_dir / "vllm_startup.log"
    proc = start_server(str(paths.checkpoint_dir), quant_cfg, profile, log_path)
    try:
        if not wait_for_health():
            proc.terminate()
            raise RuntimeError(f"vLLM server never became healthy, see {log_path}")

        kv_tokens = scrape_kv_cache_tokens(log_path)
        print(f"[serve_bench] KV cache size: {kv_tokens} tokens")

        num_prompts = profile["bench"]["num_prompts"]
        for workload, wl_cfg in profile["bench"]["workloads"].items():
            for concurrency in wl_cfg["concurrency"]:
                out_path = paths.bench_path(workload, concurrency)
                run_bench_workload(str(paths.checkpoint_dir), workload, wl_cfg, concurrency,
                                    num_prompts, out_path)

        meta = config_meta(args.model, args.config)
        meta["kv_cache_tokens"] = kv_tokens
        write_json(paths.results_dir / "bench_meta.json", meta)
    finally:
        print("[serve_bench] shutting down server")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
