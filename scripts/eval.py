#!/usr/bin/env python3
"""Run lm-eval on a quantized checkpoint.

Local (Mac): HF backend, small --limit, CPU/MPS.
Pod (GPU):   vLLM backend, profile-driven limit (smoke=20, full=None).

Usage:
    python scripts/eval.py --model qwen2.5-0.5b --config w4a16 --profile local
    python scripts/eval.py --model qwen2.5-7b   --config w4a16 --profile full
"""
from __future__ import annotations

import argparse
import time

from common import RunPaths, config_meta, load_profile, write_json


def _local_device() -> str:
    """HFLM defaults to device='cuda', which asserts on a Mac with no CUDA build.
    MPS is deliberately NOT used here: lm-eval's generate_until on MPS hit
    'RuntimeError: Invalid buffer size: 11.88 GiB' during local testing (a known class
    of MPS allocator issue triggered by this transformers/torch combo on generation
    tasks) — confirmed by reproducing it locally. CPU is slower but correct; this
    script only ever runs small --limit evals locally anyway."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_hf_backend(checkpoint_dir: str, tasks: list, limit, num_fewshot: int, metadata: dict = None) -> dict:
    """HF backend is local-dev-only (see module docstring): CPU, batch_size=1,
    float32 (bf16 matmul is often SLOWER than fp32 on M-series CPUs — no bf16 SIMD
    path), max_length capped at 2048 so a runaway generation can't blow up compute/
    memory the way the uncapped MPS run did. This is a smoke check that eval.json
    has the right keys, not a real accuracy measurement — profiles/local.yaml keeps
    task list + limit tiny on purpose."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=checkpoint_dir, dtype="float32", batch_size=1, max_length=2048,
              device=_local_device())
    results = simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, limit=limit, metadata=metadata)
    return results["results"]


def run_vllm_backend(checkpoint_dir: str, tasks: list, limit, num_fewshot: int,
                      max_model_len: int, metadata: dict = None) -> dict:
    from lm_eval import simple_evaluate

    model_args = (
        f"pretrained={checkpoint_dir},max_model_len={max_model_len},dtype=auto,"
        f"gpu_memory_utilization=0.70,batch_size=16"
    )
    results = simple_evaluate(
        model="vllm", model_args=model_args, tasks=tasks, num_fewshot=num_fewshot, limit=limit,
        metadata=metadata,
    )
    return results["results"]


def run_ruler(checkpoint_dir: str, backend: str, lengths: list[int], limit, max_model_len: int,
              num_fewshot: int = 0) -> dict:
    """RULER (lm-eval) is a single group task ("ruler") made of 13 synthetic subtasks;
    each subtask reports one metric PER length ("4096,none", "16384,none", ...), not a
    separate task per length. Which lengths actually get generated (and how long that
    takes — full RULER defaults to 500 docs x 6 lengths x 13 subtasks) is controlled via
    `metadata={"max_seq_lengths": [...]}`, and RULER needs a tokenizer name to size its
    synthetic prompts, passed the same way (see lm_eval/tasks/ruler/common_utils.py).
    We then average each length's metric across the 13 subtasks ourselves, since the
    task's built-in group aggregate only rolls up length "4096" by default.
    """
    metadata = {"tokenizer": checkpoint_dir, "max_seq_lengths": lengths}
    try:
        if backend == "hf":
            raw = run_hf_backend(checkpoint_dir, ["ruler"], limit, num_fewshot, metadata=metadata)
        else:
            raw = run_vllm_backend(checkpoint_dir, ["ruler"], limit, num_fewshot, max_model_len, metadata=metadata)
    except Exception as e:  # noqa: BLE001 — record and move on, don't kill the whole eval run
        return {f"ruler_{length}": {"error": str(e)} for length in lengths}

    out = {}
    for length in lengths:
        metric_key = f"{length},none"
        per_subtask = [v[metric_key] for k, v in raw.items()
                        if isinstance(v, dict) and metric_key in v and k != "ruler"]
        out[f"ruler_{length}"] = {
            "score,none": sum(per_subtask) / len(per_subtask) if per_subtask else None,
            "n_subtasks": len(per_subtask),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True, choices=["local", "smoke", "full"])
    args = ap.parse_args()

    profile = load_profile(args.profile)
    paths = RunPaths(model_name=args.model, config_id=args.config)
    checkpoint_dir = str(paths.checkpoint_dir)

    eval_cfg = profile["eval"]
    tasks = [t for t in eval_cfg["tasks"] if t != "ruler"]
    max_model_len = profile.get("server", {}).get("max_model_len", 20480)

    print(f"[eval] {args.model}/{args.config} profile={args.profile} "
          f"backend={eval_cfg['backend']} limit={eval_cfg['limit']} tasks={tasks}")

    t0 = time.time()
    if eval_cfg["backend"] == "hf":
        results = run_hf_backend(checkpoint_dir, tasks, eval_cfg["limit"], eval_cfg["num_fewshot"])
    else:
        results = run_vllm_backend(checkpoint_dir, tasks, eval_cfg["limit"], eval_cfg["num_fewshot"], max_model_len)

    if "ruler" in eval_cfg["tasks"]:
        ruler_limit = eval_cfg.get("ruler_limit", eval_cfg["limit"])
        ruler_results = run_ruler(
            checkpoint_dir, eval_cfg["backend"], eval_cfg.get("ruler_lengths", [4096]),
            ruler_limit, max_model_len,
        )
        results.update(ruler_results)

    elapsed = time.time() - t0
    out = {
        "meta": config_meta(args.model, args.config),
        "profile": args.profile,
        "backend": eval_cfg["backend"],
        "limit": eval_cfg["limit"],
        "eval_seconds": round(elapsed, 1),
        "results": results,
    }
    write_json(paths.eval_path, out)
    print(f"[eval] done in {elapsed:.1f}s -> {paths.eval_path}")


if __name__ == "__main__":
    main()
