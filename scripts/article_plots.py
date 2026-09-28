#!/usr/bin/env python3
"""Article-ready figures + summary table, built directly from raw per-config
JSONs under results/<model>/<config_id>/ (not master.csv, so it can run
mid-sweep while a GPU job is still pushing new configs).

Produces, under results/article/:
    fig1_speedup.png       decode + prefill throughput speedup vs bf16, by concurrency
    fig2_accuracy.png      GSM8K / MMLU accuracy delta vs bf16, in points
    fig3_kv_capacity.png   KV cache capacity + long-context TTFT vs bf16
    fig4_timing.png        GPU-hours breakdown (quantize / eval / bench) + cost
    summary.md             one table with every number behind the figures

Usage:
    python scripts/article_plots.py --model target --rate 1.09
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR, load_matrix, read_json
from plot import TARGET_COLORS

ARTICLE_DIR = RESULTS_DIR / "article"

LABELS = {
    "bf16": "BF16 (baseline)",
    "w8a16": "INT8 weights",
    "w4a16": "INT4 weights",
    "w8a16-fp8": "FP8 weights",
    "w8a8-fp8": "FP8 weights + act.",
    "w8a8-int8": "INT8 weights + act.",
    "kv-fp8-e4m3": "FP8 KV (e4m3)",
    "kv-fp8-e5m2": "FP8 KV (e5m2)",
}

DECODE_CONCURRENCIES = [1, 4, 16, 64]
PREFILL_CONCURRENCIES = [1, 4, 16, 64]
LONGCTX_CONCURRENCY = 64


def label_of(config_id: str) -> str:
    return LABELS.get(config_id, config_id)


def style_of(meta: dict) -> dict:
    color = TARGET_COLORS.get(meta.get("target"), "#333333")
    if meta.get("is_isolation_ref"):
        return dict(color=color, linestyle=":", marker="^")
    tier = meta.get("tier")
    if tier is None or tier <= 1:
        return dict(color=color, linestyle="-", marker="o")
    return dict(color=color, linestyle="--", marker="s")


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


class Config:
    def __init__(self, config_id: str, model_name: str):
        d = RESULTS_DIR / model_name / config_id
        self.id = config_id
        self.dir = d
        self.meta = read_json(d / "meta.json") or {}
        self.bench_meta = read_json(d / "bench_meta.json") or {}
        self.eval = read_json(d / "eval.json") or {}
        self.bench = {}
        for f in d.glob("bench_*.json"):
            if f.name == "bench_meta.json":
                continue
            self.bench[f.stem[len("bench_"):]] = read_json(f)

    def bench_at(self, workload: str, concurrency: int) -> dict | None:
        return self.bench.get(f"{workload}_c{concurrency}")

    @property
    def label(self) -> str:
        return label_of(self.id)

    @property
    def style(self) -> dict:
        return style_of(self.meta)


def load_configs(model_name: str) -> list[Config]:
    ordered_ids = [c["id"] for c in load_matrix()["configs"]]
    configs = []
    for cid in ordered_ids:
        d = RESULTS_DIR / model_name / cid
        if not (d / "DONE").exists():
            continue
        if (d / "SKIPPED").exists() or (d / "FAILED").exists():
            continue
        if not d.exists():
            continue
        configs.append(Config(cid, model_name))
    return configs


def gsm8k_acc(cfg: Config) -> tuple[float, float] | tuple[None, None]:
    r = cfg.eval.get("results", {}).get("gsm8k")
    if not r or "exact_match,strict-match" not in r:
        return None, None
    return r["exact_match,strict-match"], r.get("exact_match_stderr,strict-match", 0.0)


def mmlu_acc(cfg: Config) -> tuple[float, float] | tuple[None, None]:
    r = cfg.eval.get("results", {}).get("mmlu")
    if not r or "acc,none" not in r:
        return None, None
    return r["acc,none"], r.get("acc_stderr,none", 0.0)


# ---------------------------------------------------------------------------
# fig1: speedup vs bf16, decode (output_throughput) + prefill (input tok/s)
# ---------------------------------------------------------------------------

def fig1_speedup(configs: list[Config], bf16: Config, out_dir: Path) -> None:
    fig, (ax_decode, ax_prefill) = plt.subplots(1, 2, figsize=(12, 5))

    others = [c for c in configs if c.id != bf16.id]

    for ax, workload, concurrencies, metric_label in (
        (ax_decode, "decode", DECODE_CONCURRENCIES, "output_throughput"),
        (ax_prefill, "prefill", PREFILL_CONCURRENCIES, None),
    ):
        bf16_vals = {}
        for c in concurrencies:
            b = bf16.bench_at(workload, c)
            if b is None:
                continue
            bf16_vals[c] = (
                b["output_throughput"] if metric_label else b["total_input_tokens"] / b["duration"]
            )

        for cfg in others:
            xs, ys = [], []
            for c in concurrencies:
                b = cfg.bench_at(workload, c)
                if b is None or c not in bf16_vals or bf16_vals[c] == 0:
                    continue
                val = b["output_throughput"] if metric_label else b["total_input_tokens"] / b["duration"]
                xs.append(c)
                ys.append(val / bf16_vals[c])
            if xs:
                ax.plot(xs, ys, label=cfg.label, **cfg.style)

        ax.axhline(1.0, color="#999999", linewidth=1, linestyle="-", zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel("Speedup vs BF16 (x)")
        clean_axes(ax)

    ax_decode.set_title("Writing answers (decode)")
    ax_prefill.set_title("Reading long prompts (prefill)")

    handles, labels = ax_decode.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Where each version speeds things up", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / "fig1_speedup.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig2: accuracy delta vs bf16 (GSM8K + MMLU), in percentage points
# ---------------------------------------------------------------------------

def fig2_accuracy(configs: list[Config], bf16: Config, out_dir: Path) -> None:
    others = [c for c in configs if c.id != bf16.id]
    bf16_gsm8k, bf16_gsm8k_se = gsm8k_acc(bf16)
    bf16_mmlu, bf16_mmlu_se = mmlu_acc(bf16)

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(others)), 5.5))

    x = list(range(len(others)))
    width = 0.35
    label_extents = [0.0]

    for i, cfg in enumerate(others):
        g, gse = gsm8k_acc(cfg)
        m, mse = mmlu_acc(cfg)

        bars = []
        if g is not None and bf16_gsm8k is not None:
            delta = (g - bf16_gsm8k) * 100
            err = math.sqrt(gse ** 2 + bf16_gsm8k_se ** 2) * 100
            bars.append(("GSM8K", i - width / 2, delta, err, "#4C78A8"))
        if m is not None and bf16_mmlu is not None:
            delta = (m - bf16_mmlu) * 100
            err = math.sqrt(mse ** 2 + bf16_mmlu_se ** 2) * 100
            bars.append(("MMLU", i + width / 2, delta, err, "#F58518"))

        for name, xpos, delta, err, color in bars:
            ax.bar(xpos, delta, width=width, color=color, yerr=err, capsize=3, zorder=2,
                   label=name if i == 0 else None)
            offset = (err + abs(delta) * 0.02 + 0.15) * (1 if delta >= 0 else -1)
            va = "bottom" if delta >= 0 else "top"
            ax.text(xpos, delta + offset, f"{delta:+.2f}", ha="center", va=va, fontsize=8)
            label_extents.append(abs(delta + offset))

    ax.axhline(0.0, color="#999999", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.label for c in others], rotation=20, ha="right")
    ax.set_ylabel("Change vs BF16 (percentage points)")
    ymax = max(label_extents) * 1.35
    ax.set_ylim(-ymax, ymax)
    ax.legend(loc="best")
    clean_axes(ax)

    baseline_bits = []
    if bf16_gsm8k is not None:
        baseline_bits.append(f"GSM8K {bf16_gsm8k * 100:.1f}%")
    if bf16_mmlu is not None:
        baseline_bits.append(f"MMLU {bf16_mmlu * 100:.1f}%")
    baseline_str = ", ".join(baseline_bits)
    fig.suptitle("What each version costs in accuracy", fontsize=14, fontweight="bold")
    ax.set_title(f"BF16 baseline: {baseline_str}", fontsize=10, color="#555555")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig3: KV cache capacity + long-context TTFT
# ---------------------------------------------------------------------------

def fig3_kv_capacity(configs: list[Config], bf16: Config, out_dir: Path) -> None:
    bf16_kv = bf16.bench_meta.get("kv_cache_tokens")

    fig, (ax_kv, ax_ttft) = plt.subplots(1, 2, figsize=(12, max(4, 0.6 * len(configs))))

    y = list(range(len(configs)))
    labels = [c.label for c in configs]
    colors = [c.style["color"] for c in configs]

    kv_vals = [c.bench_meta.get("kv_cache_tokens") for c in configs]
    kv_thousands = [v / 1000 if v is not None else 0 for v in kv_vals]
    ax_kv.barh(y, kv_thousands, color=colors, zorder=2)
    for yi, v in zip(y, kv_vals):
        if v is None:
            continue
        pct = (v / bf16_kv - 1) * 100 if bf16_kv else 0
        ax_kv.text(v / 1000, yi, f"  {v / 1000:.0f}k ({pct:+.0f}%)", va="center", fontsize=8)
    ax_kv.set_yticks(y)
    ax_kv.set_yticklabels(labels)
    ax_kv.invert_yaxis()
    ax_kv.set_xlabel("KV cache capacity (thousands of tokens)")
    clean_axes(ax_kv)
    ax_kv.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax_kv.grid(axis="y", visible=False)

    ttft_vals = []
    for c in configs:
        b = c.bench_at("longctx", LONGCTX_CONCURRENCY)
        ttft_vals.append(b["median_ttft_ms"] / 1000 if b else None)
    ax_ttft.barh(y, [v if v is not None else 0 for v in ttft_vals], color=colors, zorder=2)
    for yi, v in zip(y, ttft_vals):
        if v is None:
            continue
        ax_ttft.text(v, yi, f"  {v:.1f}s", va="center", fontsize=8)
    ax_ttft.set_yticks(y)
    ax_ttft.set_yticklabels(labels)
    ax_ttft.invert_yaxis()
    ax_ttft.set_xlabel(f"Median TTFT at c={LONGCTX_CONCURRENCY}, long context (s)")
    clean_axes(ax_ttft)
    ax_ttft.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax_ttft.grid(axis="y", visible=False)

    fig.suptitle("More memory for conversations means less waiting", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "fig3_kv_capacity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig4: GPU-hours breakdown + cost
# ---------------------------------------------------------------------------

def bench_seconds(cfg: Config) -> float:
    return sum(b.get("duration", 0.0) for b in cfg.bench.values() if b)


def fig4_timing(configs: list[Config], rate: float, out_dir: Path) -> tuple[float, float]:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(configs))))

    y = list(range(len(configs)))
    labels = [c.label for c in configs]

    quant_h = [c.meta.get("quantize_seconds", 0.0) / 3600 for c in configs]
    eval_h = [c.eval.get("eval_seconds", 0.0) / 3600 for c in configs]
    bench_h = [bench_seconds(c) / 3600 for c in configs]

    ax.barh(y, quant_h, color="#4C78A8", label="Quantize", zorder=2)
    left = list(quant_h)
    ax.barh(y, eval_h, left=left, color="#F58518", label="Accuracy tests", zorder=2)
    left = [a + b for a, b in zip(left, eval_h)]
    ax.barh(y, bench_h, left=left, color="#54A24B", label="Speed tests", zorder=2)

    totals = [q + e + b for q, e, b in zip(quant_h, eval_h, bench_h)]
    ax.set_xlim(0, max(totals) * 1.28)
    for yi, total in zip(y, totals):
        ax.text(total, yi, f"  {total:.2f}h (\\${total * rate:.2f})", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("GPU hours")
    clean_axes(ax)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    handles, leg_labels = ax.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    grand_total_h = sum(totals)
    grand_total_cost = grand_total_h * rate
    fig.suptitle(
        f"Total: {grand_total_h:.2f} GPU-hours, \\${grand_total_cost:.2f} at \\${rate:.2f}/hr",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(out_dir / "fig4_timing.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return grand_total_h, grand_total_cost


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

def write_summary(configs: list[Config], bf16: Config, rate: float, out_dir: Path) -> None:
    bf16_gsm8k, _ = gsm8k_acc(bf16)
    bf16_mmlu, _ = mmlu_acc(bf16)
    bf16_kv = bf16.bench_meta.get("kv_cache_tokens")
    bf16_decode = {c: (bf16.bench_at("decode", c) or {}).get("output_throughput") for c in (1, 64)}
    bf16_prefill_b = bf16.bench_at("prefill", 64)
    bf16_prefill_64 = (
        bf16_prefill_b["total_input_tokens"] / bf16_prefill_b["duration"] if bf16_prefill_b else None
    )

    headers = [
        "Config", "GSM8K", "dGSM8K (pp)", "MMLU", "dMMLU (pp)",
        "KV tokens", "KV vs BF16", "Decode c1 (tok/s)", "Decode c64 (tok/s)",
        "Decode c1 speedup", "Decode c64 speedup", "Prefill c64 (tok/s)",
        "Prefill c64 speedup", "Longctx c64 TTFT (s)", "GPU hours",
    ]
    rows = []
    total_h = 0.0
    for cfg in configs:
        g, _ = gsm8k_acc(cfg)
        m, _ = mmlu_acc(cfg)
        kv = cfg.bench_meta.get("kv_cache_tokens")
        d1 = (cfg.bench_at("decode", 1) or {}).get("output_throughput")
        d64 = (cfg.bench_at("decode", 64) or {}).get("output_throughput")
        p64_b = cfg.bench_at("prefill", 64)
        p64 = p64_b["total_input_tokens"] / p64_b["duration"] if p64_b else None
        ttft_b = cfg.bench_at("longctx", 64)
        ttft = ttft_b["median_ttft_ms"] / 1000 if ttft_b else None
        hours = (cfg.meta.get("quantize_seconds", 0.0) + cfg.eval.get("eval_seconds", 0.0)
                 + bench_seconds(cfg)) / 3600
        total_h += hours

        def fmt_pct(v):
            return f"{v * 100:.2f}%" if v is not None else "-"

        def fmt_delta(v, base):
            return f"{(v - base) * 100:+.2f}" if (v is not None and base is not None) else "-"

        def fmt_speedup(v, base):
            return f"{v / base:.2f}x" if (v and base) else "-"

        rows.append([
            cfg.label,
            fmt_pct(g), fmt_delta(g, bf16_gsm8k),
            fmt_pct(m), fmt_delta(m, bf16_mmlu),
            f"{kv:,}" if kv is not None else "-",
            f"{(kv / bf16_kv - 1) * 100:+.0f}%" if (kv is not None and bf16_kv) else "-",
            f"{d1:.1f}" if d1 is not None else "-",
            f"{d64:.1f}" if d64 is not None else "-",
            fmt_speedup(d1, bf16_decode.get(1)),
            fmt_speedup(d64, bf16_decode.get(64)),
            f"{p64:.1f}" if p64 is not None else "-",
            fmt_speedup(p64, bf16_prefill_64),
            f"{ttft:.1f}" if ttft is not None else "-",
            f"{hours:.2f}",
        ])

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    lines.append(f"**Total: {total_h:.2f} GPU-hours, ${total_h * rate:.2f} at ${rate:.2f}/hr**")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="target")
    ap.add_argument("--rate", type=float, default=1.09, help="$/GPU-hour")
    args = ap.parse_args()

    configs = load_configs(args.model)
    if not configs:
        raise SystemExit(f"no usable configs found for model={args.model}")
    bf16 = next((c for c in configs if c.id == "bf16"), None)
    if bf16 is None:
        raise SystemExit("bf16 baseline config not found (required for all figures)")

    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

    fig1_speedup(configs, bf16, ARTICLE_DIR)
    fig2_accuracy(configs, bf16, ARTICLE_DIR)
    fig3_kv_capacity(configs, bf16, ARTICLE_DIR)
    fig4_timing(configs, args.rate, ARTICLE_DIR)
    write_summary(configs, bf16, args.rate, ARTICLE_DIR)

    print(f"Wrote figures + summary for {len(configs)} configs to {ARTICLE_DIR}")


if __name__ == "__main__":
    main()
