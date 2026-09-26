#!/usr/bin/env python3
"""Quantize a model with llm-compressor according to a configs/quant/<id>.yaml recipe.

Usage:
    python scripts/quantize.py --model qwen2.5-0.5b --config w4a16
    python scripts/quantize.py --model qwen2.5-0.5b --config bf16   # control: just saves the base model

Output: checkpoints/<model>/<config>/  (gitignored — never pushed to GitHub, only eval/bench JSON is)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import RunPaths, config_meta, load_model_config, load_quant_config, write_json


def build_modifiers(quant_cfg: dict):
    """Translate our config schema into an llm-compressor recipe (list of modifiers).
    NOTE: exact kwargs get validated against the installed llmcompressor version at
    smoke-test time (see PLAN.md phase 3) — this is the best-effort mapping from the
    documented preset schemes (W8A16/W4A16/W8A8/FP8/FP8_DYNAMIC).
    """
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    modifiers = []

    if quant_cfg.get("smoothquant", {}).get("enabled"):
        modifiers.append(
            SmoothQuantModifier(
                smoothing_strength=quant_cfg["smoothquant"].get("smoothing_strength", 0.8)
            )
        )

    weights = quant_cfg.get("weights")
    kv_cfg = quant_cfg.get("kv_cache", {})
    kv_cache_scheme = None
    if kv_cfg.get("enabled"):
        kv_cache_scheme = {
            "num_bits": kv_cfg["num_bits"],
            "type": kv_cfg["type"],
            "strategy": kv_cfg.get("strategy", "tensor"),
            "dynamic": kv_cfg.get("dynamic", False),
        }

    if weights is not None:
        common_kwargs = dict(
            scheme=weights["scheme"],
            targets=weights.get("targets", ["Linear"]),
            ignore=weights.get("ignore", ["lm_head"]),
        )
        # NOTE: group_size is NOT a modifier kwarg — the W4A16/W8A16 presets already
        # bake in group_size=128 (checked against installed compressed-tensors 0.14).
        # configs/quant/*.yaml still records it for documentation purposes only.
        if kv_cache_scheme:
            common_kwargs["kv_cache_scheme"] = kv_cache_scheme

        if weights.get("algorithm") == "gptq":
            modifiers.append(GPTQModifier(**common_kwargs))
        else:
            # simple (non-GPTQ) static/dynamic quant, e.g. plain FP8 weight-only
            modifiers.append(QuantizationModifier(**common_kwargs))
    elif kv_cache_scheme:
        # KV-only configs: no weight/activation quant, just calibrate KV scales.
        modifiers.append(
            QuantizationModifier(
                targets=[],
                ignore=["Linear"],
                kv_cache_scheme=kv_cache_scheme,
            )
        )

    return modifiers


def run_control(model_hf_id: str, out_dir: Path) -> None:
    """bf16 config: no quantization, just materialize the base model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[quantize] control (bf16) — saving base model to {out_dir}")
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype="bfloat16")
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)


def run_quantize(model_hf_id: str, quant_cfg: dict, calib_cfg: dict, out_dir: Path) -> None:
    from llmcompressor import oneshot
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)

    def preprocess(example):
        messages = example.get("messages") or [{"role": "user", "content": example.get("prompt", "")}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=calib_cfg["max_seq_len"])

    print(f"[quantize] loading calibration set {calib_cfg['dataset']} "
          f"({calib_cfg['num_samples']} samples)")
    ds = load_dataset(calib_cfg["dataset"], split=calib_cfg["split"])
    ds = ds.shuffle(seed=42).select(range(min(calib_cfg["num_samples"], len(ds))))
    ds = ds.map(preprocess, remove_columns=ds.column_names)

    modifiers = build_modifiers(quant_cfg)
    print(f"[quantize] recipe: {[m.__class__.__name__ for m in modifiers]}")

    oneshot(
        model=model_hf_id,
        dataset=ds,
        recipe=modifiers,
        output_dir=str(out_dir),
        max_seq_length=calib_cfg["max_seq_len"],
        num_calibration_samples=calib_cfg["num_samples"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model config name under configs/models/")
    ap.add_argument("--config", required=True, help="quant config id under configs/quant/")
    ap.add_argument("--force", action="store_true", help="re-quantize even if output exists")
    args = ap.parse_args()

    model_cfg = load_model_config(args.model)
    quant_cfg = load_quant_config(args.config)
    paths = RunPaths(model_name=args.model, config_id=args.config)
    out_dir = paths.checkpoint_dir

    if any(out_dir.iterdir()) if out_dir.exists() else False:
        if not args.force:
            print(f"[quantize] {out_dir} already populated, skipping (use --force to redo)")
            return
        print(f"[quantize] --force: re-quantizing into {out_dir}")

    t0 = time.time()
    skipped_reason = None
    try:
        if quant_cfg["target"] == "control":
            run_control(model_cfg["hf_id"], out_dir)
        else:
            run_quantize(model_cfg["hf_id"], quant_cfg, model_cfg["calib"], out_dir)
    except RuntimeError as e:
        involves_fp8 = "FP8" in str(quant_cfg.get("weights") or {}).upper() or \
            "FP8" in str(quant_cfg.get("activations") or {}).upper() or \
            quant_cfg.get("kv_cache", {}).get("format", "").startswith("e")
        no_gpu = not __import__("torch").cuda.is_available()
        if involves_fp8 and no_gpu and "Float8" in str(e):
            # Expected per PLAN.md: FP8 tensor-core ops aren't supported on CPU/MPS.
            # Real validation happens on the Ada/Hopper smoke pod (see phase 3).
            print(f"[quantize] SKIPPED (local, no FP8 hardware): {e}")
            skipped_reason = str(e)
        else:
            raise
    elapsed = time.time() - t0

    meta = config_meta(args.model, args.config)
    meta["hf_id"] = model_cfg["hf_id"]
    meta["checkpoint_dir"] = str(out_dir)
    meta["quantize_seconds"] = round(elapsed, 1)
    if skipped_reason:
        meta["skipped_local"] = True
        meta["skipped_reason"] = skipped_reason
    write_json(paths.meta_path, meta)
    if skipped_reason:
        print(f"[quantize] recorded skip for {args.config} (validate on GPU smoke pod instead)")
    else:
        print(f"[quantize] done in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
