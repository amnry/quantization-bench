#!/usr/bin/env python3
"""Read results/master.csv and produce:
  - results/master_table.md (re-render, for convenience)
  - results/pareto.png            accuracy vs throughput, colored by target
  - results/throughput_vs_conc.png  throughput vs concurrency, one line per config

Usage:
    python scripts/plot.py --model qwen2.5-0.5b
    python scripts/plot.py --model qwen2.5-0.5b --workload decode
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

TARGET_COLORS = {
    "control": "#888888",
    "isolation-ref": "#c9c9c9",
    "weights": "#4C78A8",
    "activations": "#F58518",
    "kv": "#54A24B",
    "attn-proj": "#B279A2",
}

BENCH_COL_RE = re.compile(r"^bench_(?P<workload>.+)_c(?P<concurrency>\d+)_(?P<metric>.+)$")


def load_master(model_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / "master.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/collect.py first")
    df = pd.read_csv(path)
    df = df[df["model"] == model_name].copy()
    if df.empty:
        raise ValueError(f"no rows for model={model_name} in {path}")
    return df


def mean_accuracy(row: pd.Series) -> float | None:
    acc_cols = [c for c in row.index if c.startswith("eval_") and (
        "acc" in c or "score" in c) and pd.notna(row[c])]
    if not acc_cols:
        return None
    return sum(row[c] for c in acc_cols) / len(acc_cols)


def representative_throughput(row: pd.Series, workload: str) -> float | None:
    """Highest-concurrency output_throughput for the given workload, as a single
    representative number per config for the Pareto plot."""
    cols = [c for c in row.index if c.startswith(f"bench_{workload}_") and c.endswith("output_throughput")
            and pd.notna(row[c])]
    if not cols:
        return None

    def conc_of(col: str) -> int:
        m = BENCH_COL_RE.match(col)
        return int(m.group("concurrency")) if m else 0

    best_col = max(cols, key=conc_of)
    return row[best_col]


def plot_pareto(df: pd.DataFrame, out_path: Path, workload: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        acc = mean_accuracy(row)
        tput = representative_throughput(row, workload)
        if acc is None or tput is None:
            continue
        color = TARGET_COLORS.get(row["target"], "#333333")
        ax.scatter(tput, acc, color=color, s=70, edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(row["config_id"], (tput, acc), fontsize=7, xytext=(4, 4), textcoords="offset points")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t)
               for t, c in TARGET_COLORS.items()]
    ax.legend(handles=handles, title="target", fontsize=8, loc="best")
    ax.set_xlabel(f"output throughput @ highest concurrency ({workload}) [tok/s]")
    ax.set_ylabel("mean accuracy across eval tasks")
    ax.set_title("Accuracy vs Throughput (Pareto view)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_throughput_vs_concurrency(df: pd.DataFrame, out_path: Path, workload: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        points = []
        for col in row.index:
            m = BENCH_COL_RE.match(col)
            if m and m.group("workload") == workload and m.group("metric") == "output_throughput" \
                    and pd.notna(row[col]):
                points.append((int(m.group("concurrency")), row[col]))
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        color = TARGET_COLORS.get(row["target"], "#333333")
        ax.plot(xs, ys, marker="o", label=row["config_id"], color=color)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output throughput [tok/s]")
    ax.set_title(f"Throughput vs Concurrency — {workload}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="decode", choices=["decode", "prefill", "longctx"],
                     help="workload used for the Pareto x-axis and throughput-vs-concurrency chart")
    args = ap.parse_args()

    df = load_master(args.model)
    plot_pareto(df, RESULTS_DIR / "pareto.png", args.workload)
    plot_throughput_vs_concurrency(df, RESULTS_DIR / "throughput_vs_conc.png", args.workload)


if __name__ == "__main__":
    main()
