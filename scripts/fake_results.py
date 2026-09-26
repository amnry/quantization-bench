#!/usr/bin/env python3
"""Generate synthetic results (matching the real eval.json / bench_*.json schema) for all
10 configs, so collect.py and plot.py can be exercised end to end without a GPU.

Writes under results/<model>/ with model defaulting to 'fake-demo' — deliberately NOT the
real local test-model name, so it can never collide with actual local quantize/eval output.

Numbers are made up to match the hypotheses we're testing (see PLAN.md):
  - weights (w8a16/w4a16/w8a16-fp8): decode throughput up, small accuracy cost (more @ w4).
  - activations (w8a8-fp8/w8a8-int8): prefill throughput up more than decode, INT8 costs
    more accuracy than FP8 at equal 8-bit width (that's the whole point of tier_axis=format).
  - kv (kv-fp8-e4m3/e5m2): longctx throughput up + ~2x KV token capacity, RULER accuracy
    degrades more at e5m2 (fewer mantissa bits).
  - attn-proj (w8/w4): decode throughput up less than full weight quant (fewer params
    touched), but accuracy cost is disproportionately high — the sensitivity story.

Usage:
    python scripts/fake_results.py --model fake-demo
"""
from __future__ import annotations

import argparse
import random

from common import RunPaths, config_meta, all_config_ids, write_json

random.seed(42)

BASE_DECODE_TPS = {1: 45, 4: 160, 16: 520, 64: 900}
BASE_PREFILL_TPS = {1: 900, 4: 3200, 16: 9500, 64: 14000}
BASE_LONGCTX_TPS = {16: 4200, 64: 11000, 128: 15500, 256: 17000}
BASE_KV_TOKENS = 235_000
BASE_GSM8K = 0.42
BASE_MMLU = 0.47
BASE_RULER_4096 = 0.88
BASE_RULER_16384 = 0.71

# (decode_mult, prefill_mult, longctx_mult, kv_mult, gsm8k_delta, mmlu_delta,
#  ruler4096_delta, ruler16384_delta)
CONFIG_EFFECTS = {
    "bf16":         (1.00, 1.00, 1.00, 1.00,  0.000,  0.000,  0.000,  0.000),
    "w8a16":        (1.55, 1.05, 1.05, 1.00, -0.005, -0.003,  0.000,  0.000),
    "w4a16":        (1.95, 1.08, 1.08, 1.00, -0.025, -0.018, -0.005, -0.010),
    "w8a16-fp8":    (1.50, 1.10, 1.05, 1.00, -0.006, -0.004,  0.000,  0.000),
    "w8a8-fp8":     (1.65, 1.85, 1.20, 1.00, -0.010, -0.008, -0.002, -0.004),
    "w8a8-int8":    (1.60, 1.75, 1.15, 1.00, -0.040, -0.030, -0.015, -0.025),
    "kv-fp8-e4m3":  (1.05, 1.02, 1.60, 1.95, -0.003, -0.002, -0.008, -0.015),
    "kv-fp8-e5m2":  (1.05, 1.02, 1.65, 2.00, -0.008, -0.006, -0.030, -0.060),
    "attn-proj-w8": (1.15, 1.03, 1.03, 1.00, -0.018, -0.012, -0.004, -0.006),
    "attn-proj-w4": (1.25, 1.05, 1.05, 1.00, -0.070, -0.050, -0.020, -0.035),
}


def jitter(x: float, pct: float = 0.03) -> float:
    return x * (1 + random.uniform(-pct, pct))


def build_bench(concurrency_map: dict, mult: float) -> dict:
    return {c: jitter(base * mult) for c, base in concurrency_map.items()}


def make_bench_file(output_tps: float, concurrency: int, num_prompts: int = 20) -> dict:
    input_tps = output_tps * 0.9
    return {
        "num_prompts": num_prompts,
        "completed": num_prompts,
        "request_throughput": round(output_tps / 200, 3),
        "output_throughput": round(output_tps, 1),
        "input_throughput": round(input_tps, 1),
        "mean_ttft_ms": round(jitter(150 / max(concurrency, 1) * 4), 1),
        "mean_tpot_ms": round(jitter(2000 / output_tps * 10), 2),
        "mean_itl_ms": round(jitter(2000 / output_tps * 10), 2),
        "_synthetic": True,
    }


def make_eval_file(config_id: str, model_name: str) -> dict:
    d_mult, p_mult, lc_mult, kv_mult, gsm8k_d, mmlu_d, ruler4k_d, ruler16k_d = CONFIG_EFFECTS[config_id]
    results = {
        "gsm8k": {"acc,none": round(jitter(BASE_GSM8K + gsm8k_d), 4)},
        "mmlu": {"acc,none": round(jitter(BASE_MMLU + mmlu_d), 4)},
        "ruler_4096": {"score,none": round(jitter(BASE_RULER_4096 + ruler4k_d), 4)},
        "ruler_16384": {"score,none": round(jitter(BASE_RULER_16384 + ruler16k_d), 4)},
    }
    return {
        "meta": config_meta(model_name, config_id),
        "profile": "fake",
        "backend": "synthetic",
        "limit": None,
        "eval_seconds": 0.0,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake-demo")
    args = ap.parse_args()

    for config_id in all_config_ids():
        d_mult, p_mult, lc_mult, kv_mult, *_ = CONFIG_EFFECTS[config_id]
        paths = RunPaths(model_name=args.model, config_id=config_id)

        meta = config_meta(args.model, config_id)
        meta["quantize_seconds"] = round(random.uniform(5, 60), 1)
        write_json(paths.meta_path, meta)

        write_json(paths.eval_path, make_eval_file(config_id, args.model))

        for concurrency, base in BASE_DECODE_TPS.items():
            write_json(paths.bench_path("decode", concurrency),
                       make_bench_file(jitter(base * d_mult), concurrency))
        for concurrency, base in BASE_PREFILL_TPS.items():
            write_json(paths.bench_path("prefill", concurrency),
                       make_bench_file(jitter(base * p_mult), concurrency))
        for concurrency, base in BASE_LONGCTX_TPS.items():
            write_json(paths.bench_path("longctx", concurrency),
                       make_bench_file(jitter(base * lc_mult), concurrency))

        bench_meta = config_meta(args.model, config_id)
        bench_meta["kv_cache_tokens"] = int(jitter(BASE_KV_TOKENS * kv_mult))
        write_json(paths.results_dir / "bench_meta.json", bench_meta)

        paths.mark_done()
        print(f"[fake_results] wrote synthetic results for {config_id}")

    print(f"[fake_results] done — results/{args.model}/ ready for collect.py + plot.py")


if __name__ == "__main__":
    main()
