#!/usr/bin/env python3
"""Read results/master.csv and produce:
  - results/pareto.png                        accuracy delta (vs bf16) vs throughput
  - results/throughput_vs_conc_<workload>.png  throughput vs concurrency, one per workload
  - results/speedup_<workload>.png             throughput / bf16 throughput, one per workload
  - results/longctx_kv_capacity.png            GPU KV cache tokens per config (longctx only)

Every config gets a unique (color, linestyle, marker) combo: color = target,
linestyle/marker = tier (tier1 = solid/circle, tier2 = dashed/square,
isolation-ref = dotted/triangle).

Usage:
    python scripts/plot.py --model qwen2.5-0.5b
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - degrade gracefully if not installed
    adjust_text = None

from common import RESULTS_DIR

WORKLOADS = ["decode", "prefill", "longctx"]

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


def style_of(row: pd.Series) -> dict:
    """color = target, linestyle+marker = tier, so every config is visually unique."""
    color = TARGET_COLORS.get(row["target"], "#333333")
    if bool(row.get("is_isolation_ref")):
        return dict(color=color, linestyle=":", marker="^")
    tier = row.get("tier")
    if pd.isna(tier) or tier <= 1:
        return dict(color=color, linestyle="-", marker="o")
    return dict(color=color, linestyle="--", marker="s")


def is_metric_col(col: str) -> bool:
    return col.startswith("eval_") and "_stderr_" not in col and not col.endswith("_sample_len")


def stderr_col_of(value_col: str) -> str:
    """eval_<task>_<metric>_<agg> -> eval_<task>_<metric>_stderr_<agg>."""
    base, _, agg = value_col.rpartition("_")
    return f"{base}_stderr_{agg}"


def accuracy_delta_pp(row: pd.Series, bf16_row: pd.Series, metric_cols: list[str]) -> tuple[float, float] | tuple[None, None]:
    """Mean accuracy delta vs bf16 in percentage points, with combined stderr
    (independent-metric assumption: var of the mean = sum(var_i) / n^2)."""
    deltas, variances = [], []
    for col in metric_cols:
        if col not in row.index or pd.isna(row[col]) or pd.isna(bf16_row.get(col)):
            continue
        deltas.append(row[col] - bf16_row[col])
        se_col = stderr_col_of(col)
        se_a = row.get(se_col, 0.0) or 0.0
        se_b = bf16_row.get(se_col, 0.0) or 0.0
        variances.append((se_a if pd.notna(se_a) else 0.0) ** 2 + (se_b if pd.notna(se_b) else 0.0) ** 2)
    if not deltas:
        return None, None
    n = len(deltas)
    mean_delta = sum(deltas) / n * 100
    combined_se = (sum(variances) ** 0.5) / n * 100
    return mean_delta, combined_se


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


def legend_handles() -> list:
    target_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t)
                       for t, c in TARGET_COLORS.items()]
    tier_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", marker="o", label="tier 1"),
        plt.Line2D([0], [0], color="black", linestyle="--", marker="s", label="tier 2"),
        plt.Line2D([0], [0], color="black", linestyle=":", marker="^", label="isolation ref"),
    ]
    return target_handles + tier_handles


def plot_pareto(df: pd.DataFrame, out_path: Path, workload: str) -> None:
    if "bf16" not in df["config_id"].values:
        print("[plot] skip pareto.png: no bf16 row to compute accuracy delta against")
        return
    bf16_row = df[df["config_id"] == "bf16"].iloc[0]
    metric_cols = [c for c in df.columns if is_metric_col(c)]

    fig, ax = plt.subplots(figsize=(7, 5))
    texts = []
    for _, row in df.iterrows():
        delta, err = accuracy_delta_pp(row, bf16_row, metric_cols)
        tput = representative_throughput(row, workload)
        if delta is None or tput is None:
            continue
        style = style_of(row)
        ax.errorbar(tput, delta, yerr=err, fmt=style["marker"], color=style["color"],
                    ecolor=style["color"], elinewidth=1, capsize=3, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        texts.append(ax.annotate(row["config_id"], (tput, delta), fontsize=7))

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6, zorder=1)
    if texts and adjust_text is not None:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    ax.legend(handles=legend_handles(), fontsize=7, ncol=2, loc="best")
    ax.set_xlabel(f"output throughput @ highest concurrency ({workload}) [tok/s]")
    ax.set_ylabel("accuracy delta vs bf16 [pp]")
    ax.set_title("Accuracy Delta vs Throughput: Pareto View")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_throughput_vs_concurrency(df: pd.DataFrame, out_path: Path, workload: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    any_points = False
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
        style = style_of(row)
        ax.plot(xs, ys, marker=style["marker"], linestyle=style["linestyle"], color=style["color"],
                label=row["config_id"])
        any_points = True

    if not any_points:
        print(f"[plot] skip {out_path.name}: no bench data for workload={workload}")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output throughput [tok/s]")
    ax.set_title(f"Throughput vs Concurrency: {workload}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_speedup(df: pd.DataFrame, out_path: Path, workload: str) -> None:
    if "bf16" not in df["config_id"].values:
        print(f"[plot] skip {out_path.name}: no bf16 row to compute speedup against")
        return
    bf16_row = df[df["config_id"] == "bf16"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    any_points = False
    for _, row in df.iterrows():
        points = []
        for col in row.index:
            m = BENCH_COL_RE.match(col)
            if not (m and m.group("workload") == workload and m.group("metric") == "output_throughput"
                    and pd.notna(row[col])):
                continue
            bf16_val = bf16_row.get(col)
            if bf16_val is None or pd.isna(bf16_val) or bf16_val == 0:
                continue
            points.append((int(m.group("concurrency")), row[col] / bf16_val))
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        style = style_of(row)
        ax.plot(xs, ys, marker=style["marker"], linestyle=style["linestyle"], color=style["color"],
                label=row["config_id"])
        any_points = True

    if not any_points:
        print(f"[plot] skip {out_path.name}: no bench data for workload={workload}")
        plt.close(fig)
        return

    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("speedup vs bf16 [output throughput ratio]")
    ax.set_title(f"Speedup vs Concurrency: {workload}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_kv_capacity(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["kv_cache_tokens"].notna()].copy()
    if sub.empty:
        print(f"[plot] skip {out_path.name}: no kv_cache_tokens data")
        return
    sub = sub.sort_values("kv_cache_tokens")
    colors = [style_of(row)["color"] for _, row in sub.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sub["config_id"], sub["kv_cache_tokens"], color=colors, edgecolor="white")
    ax.set_xlabel("config")
    ax.set_ylabel("GPU KV cache capacity [tokens]")
    ax.set_title("KV Cache Capacity: longctx")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="decode", choices=WORKLOADS,
                     help="workload used for the Pareto plot x-axis")
    args = ap.parse_args()

    df = load_master(args.model)

    plot_pareto(df, RESULTS_DIR / "pareto.png", args.workload)

    for workload in WORKLOADS:
        plot_throughput_vs_concurrency(df, RESULTS_DIR / f"throughput_vs_conc_{workload}.png", workload)
        plot_speedup(df, RESULTS_DIR / f"speedup_{workload}.png", workload)

    plot_kv_capacity(df, RESULTS_DIR / "longctx_kv_capacity.png")


if __name__ == "__main__":
    main()
