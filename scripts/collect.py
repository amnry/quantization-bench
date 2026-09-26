#!/usr/bin/env python3
"""Walk results/**, flatten eval + bench JSON into one master table, and compute the
isolation deltas defined in configs/matrix.yaml (activation/weight/KV effects).

Usage:
    python scripts/collect.py                      # all models under results/
    python scripts/collect.py --model qwen2.5-0.5b  # one model only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import RESULTS_DIR, load_matrix, read_json


def flatten_eval(eval_data: dict) -> dict:
    out = {}
    for task, metrics in (eval_data.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if metric_name.startswith("alias"):
                continue
            if isinstance(value, (int, float)):
                key = f"eval_{task}_{metric_name}".replace(",", "_")
                out[key] = value
    return out


def flatten_bench(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("bench_*.json")):
        data = read_json(f)
        if not data:
            continue
        # filename pattern: bench_<workload>_c<N>.json
        stem = f.stem  # bench_decode_c4
        parts = stem.split("_")
        workload = "_".join(parts[1:-1])
        concurrency = parts[-1]  # "c4"
        prefix = f"bench_{workload}_{concurrency}"
        for metric in ("request_throughput", "output_throughput", "input_throughput",
                       "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"):
            if metric in data:
                out[f"{prefix}_{metric}"] = data[metric]
    return out


def collect_config(model_name: str, config_id: str) -> dict:
    results_dir = RESULTS_DIR / model_name / config_id
    meta = read_json(results_dir / "meta.json") or {}
    bench_meta = read_json(results_dir / "bench_meta.json") or {}
    eval_data = read_json(results_dir / "eval.json") or {}

    row = {
        "model": model_name,
        "config_id": config_id,
        "target": meta.get("target"),
        "tier": meta.get("tier"),
        "tier_axis": meta.get("tier_axis"),
        "is_isolation_ref": meta.get("is_isolation_ref", False),
        "w_bits": meta.get("w_bits"),
        "a_bits": meta.get("a_bits"),
        "kv_bits": meta.get("kv_bits"),
        "kv_cache_tokens": bench_meta.get("kv_cache_tokens"),
        "done": (results_dir / "DONE").exists(),
    }
    row.update(flatten_eval(eval_data))
    row.update(flatten_bench(results_dir))
    return row


def build_master(model_names: list[str]) -> pd.DataFrame:
    matrix = load_matrix()
    config_ids = [c["id"] for c in matrix["configs"]]
    rows = []
    for model_name in model_names:
        for config_id in config_ids:
            results_dir = RESULTS_DIR / model_name / config_id
            if not results_dir.exists():
                continue
            rows.append(collect_config(model_name, config_id))
    return pd.DataFrame(rows)


def compute_isolation_deltas(df: pd.DataFrame) -> pd.DataFrame:
    matrix = load_matrix()
    delta_specs = matrix.get("isolation_deltas", [])
    metric_cols = [c for c in df.columns if c.startswith("eval_") or c.startswith("bench_")]

    rows = []
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].set_index("config_id")
        for spec in delta_specs:
            a_id, b_id, name = spec["a"], spec["b"], spec["name"]
            if a_id not in sub.index or b_id not in sub.index:
                continue
            row = {"model": model_name, "delta_name": name, "a": a_id, "b": b_id}
            for col in metric_cols:
                va, vb = sub.loc[a_id, col], sub.loc[b_id, col]
                if pd.notna(va) and pd.notna(vb):
                    row[col] = va - vb
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="restrict to this model (repeatable); default = all under results/")
    args = ap.parse_args()

    if args.model:
        model_names = args.model
    else:
        model_names = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir()) if RESULTS_DIR.exists() else []

    if not model_names:
        print(f"[collect] no results found under {RESULTS_DIR}")
        return

    df = build_master(model_names)
    if df.empty:
        print("[collect] no rows collected")
        return

    df = df.sort_values(["model", "config_id"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "master.csv", index=False)
    with open(RESULTS_DIR / "master.md", "w") as f:
        f.write(f"# Master results table\n\n{df.to_markdown(index=False)}\n")
    print(f"[collect] wrote {RESULTS_DIR / 'master.csv'} and master.md ({len(df)} rows)")

    deltas = compute_isolation_deltas(df)
    if not deltas.empty:
        deltas.to_csv(RESULTS_DIR / "isolation_deltas.csv", index=False)
        with open(RESULTS_DIR / "isolation_deltas.md", "w") as f:
            f.write(f"# Isolation deltas (a - b)\n\n{deltas.to_markdown(index=False)}\n")
        print(f"[collect] wrote isolation_deltas.csv/.md ({len(deltas)} rows)")


if __name__ == "__main__":
    main()
