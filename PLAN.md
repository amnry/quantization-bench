# Quantization Target Benchmark — Plan

## Context
Goal: show *which inference phase* each quantization target speeds up, and what it costs in accuracy.
Hypotheses to expose:
- **Weights** → decode (memory-bound, low batch) → TPOT drops.
- **Activations** → prefill + large-batch (compute-bound) → TTFT / input tok/s improve.
- **KV cache** → long context + high concurrency → more KV tokens fit, fewer preemptions, higher throughput at high conc.
- **Attention projections** → sensitivity study: accuracy cost of weight-quantizing q/k/v/o only, vs whole-model weight quant.
Each target: control (BF16) + Tier 1 + Tier 2 (tier meaning defined below). Output: master table, accuracy-vs-throughput Pareto plot, throughput-vs-concurrency per config, short analysis.

Repo: https://github.com/amnry/quantization-bench (local `~/quantization-bench`, own `git init`, `origin` = this repo).

## Decisions (locked)
- Test model: `Qwen/Qwen2.5-0.5B-Instruct`. Target: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, same arch → recipes transfer).
- Real GPU: H100 80GB. Smoke GPU: L4 24GB (Ada, FP8 path works).
- Evals: `gsm8k` (5-shot), `mmlu` (5-shot), `ruler` (lengths 4096 + 16384; local profile 4096 only). `max_model_len` = 16384 for every config.

## Tier definition (read before interpreting any result)
Tier does **not** mean the same thing for every target:
- **Weights, attn-proj**: tier = **bit-width**. Tier 1 = 8-bit, Tier 2 = 4-bit. Tier 2 simply has fewer bits.
- **Activations, KV cache**: **both tiers are 8-bit**. Tier = **format difficulty** at equal bit-width. Tier 2 is a harder/lossier 8-bit format (INT8 activations = no exponent, outlier-sensitive, needs SmoothQuant; FP8 e5m2 KV = 2 mantissa bits vs 3). Memory/bandwidth savings are identical across tiers; only accuracy (and kernel speed) differs.
So "Tier 2 vs Tier 1" is a bits comparison for weights/attn-proj and a format comparison for activations/KV. Master table carries an explicit `tier_axis` column (`bits` | `format`) and `w_bits`/`a_bits`/`kv_bits` columns so no one reads tiers as comparable across targets.

## Config matrix (10 configs, all via llm-compressor → compressed-tensors → vLLM)
| id | target | tier | tier_axis | scheme | notes |
|---|---|---|---|---|---|
| `bf16` | control | 0 | – | none | baseline |
| `w8a16` | weights | 1 | bits | INT8 weight-only, GPTQ | |
| `w4a16` | weights | 2 | bits | INT4 weight-only, GPTQ g128 | |
| `w8a16-fp8` | isolation ref | – | – | FP8 weights (per-channel), BF16 activations | weight-only FP8 (Marlin FP8 path in vLLM); exists only to isolate activation effect |
| `w8a8-fp8` | activations | 1 | format | FP8 W + dynamic per-token FP8 A | needs Ada/Hopper |
| `w8a8-int8` | activations | 2 | format | SmoothQuant + GPTQ INT8 W8A8 | outlier-sensitive |
| `kv-fp8-e4m3` | KV | 1 | format | BF16 weights, FP8 e4m3 KV, calibrated scales | |
| `kv-fp8-e5m2` | KV | 2 | format | BF16 weights, FP8 e5m2 KV | check newer vLLM KV dtypes at smoke |
| `attn-proj-w8` | attn-proj | 1 | bits | W8A16 INT8 on q/k/v/o weights only, MLP BF16 | attention math stays BF16 |
| `attn-proj-w4` | attn-proj | 2 | bits | W4A16 INT4 on q/k/v/o weights only, MLP BF16 | compare vs `w4a16` normalized by param share |

**Isolation rules for analysis** (unconfounded: same weight format on both sides of each subtraction):
- Activation effect (Tier 1) = `w8a8-fp8` − `w8a16-fp8` (both FP8 weights; only activations differ).
- Activation effect (Tier 2) = `w8a8-int8` − `w8a16` (both INT8 weights; only activations differ).
- KV effect = `kv-*` − `bf16` (weights/activations identical).
- Weight effect = `w8a16` / `w4a16` − `bf16`.
- attn-proj sensitivity = accuracy drop of `attn-proj-wN` per quantized param vs `wNa16` per quantized param.
- `lm_head` + embeddings never quantized.

**Naming note:** `attn-proj-*` = *weight quantization applied only to attention projection layers*. Attention math (QKᵀ, softmax, ·V) runs in BF16. Nothing in this study quantizes attention computation; README states this explicitly.

## Bench workloads (vllm bench serve, random dataset, `--ignore-eos`, fixed seed)
| workload | in/out tokens | concurrency | primary metric | exposes |
|---|---|---|---|---|
| `decode` | 128 / 512 | 1,4,16,64 | TPOT, ITL | weights |
| `prefill` | 4096 / 16 | 1,4,16,64 | TTFT, input tok/s | activations |
| `longctx` | 8192 / 256 | 16,64,128,256 | output tok/s, preemptions | KV |
Server fixed across configs: same `--max-model-len`, `--gpu-memory-utilization 0.90`, `--max-num-seqs 256`, prefix caching OFF. Scrape vLLM startup log for `GPU KV cache size: N tokens` → direct KV-capacity metric per config.

## Repo layout
```
configs/
  models/{qwen2.5-0.5b.yaml, target.yaml}     # hf id, max_model_len, calib settings
  quant/<id>.yaml                               # llm-compressor recipe per config
  matrix.yaml                                   # config id → target, tier, tier_axis, bits, recipe, vllm extra args
  profiles/{local,smoke,full}.yaml              # limit, num_prompts, concurrency lists, eval backend
scripts/
  quantize.py   --model --config --out          # llm-compressor oneshot, calib = ultrachat 256/512 samples
  eval.py       --model --config --profile      # lm-eval; hf backend local, vllm backend on pod
  serve_bench.py --model --config --profile     # launch vllm serve, wait health, sweep, parse KV log, kill
  collect.py                                    # results → master.csv + master.md (+ tier_axis, bits, isolation deltas)
  plot.py                                       # table + pareto.png + throughput_vs_conc.png
  fake_results.py                               # synthetic results matching hypotheses (local plot test)
  run_all.sh    --model --profile               # per config: quantize→eval→bench→commit+push; skip if DONE
  setup_pod.sh                                  # install pinned deps, gh auth via PAT env
results/<model>/<config>/{eval.json, bench_<workload>_c<N>.json, meta.json, DONE}
requirements-local.txt / requirements-pod.txt → requirements-lock.txt (pinned after smoke)
README.md (incl. tier definition + attn-proj note), ANALYSIS.md
```
Quantized checkpoints stay on pod disk (gitignored); only JSON results go to GitHub.

## Phases
1. **Local (M2, 16GB)**: `uv` + Python 3.12 venv (3.13 risky for llm-compressor/torch). Write scripts. Quantize 0.5B on CPU (all recipes), lm-eval HF backend `--limit 5` on MPS/CPU. FP8 checkpoints may not run on MPS → eval.py falls back to CPU, or marks `skipped_local`. `fake_results.py` → `collect.py` → `plot.py` must emit table + both charts. `serve_bench.py` tested in `--dry-run` (prints commands, parses a canned vLLM log/json fixture).
2. **Push** to `amnry/quantization-bench`.
3. **Smoke pod** (L4 24GB): clone, `setup_pod.sh`, `run_all.sh --model qwen2.5-0.5b --profile smoke` (limit 20, 20 prompts/bench, all 10 configs). Fix breaks, `pip freeze` → `requirements-lock.txt`, push. Terminate + delete disk.
4. **Real pod** (H100): install lock file, `--model target --profile full`. Per-config commit+push (`git pull --rebase && git push`, retry). Resume = skip configs with `DONE`. Terminate + delete storage.
5. **Local**: pull, `collect.py`, `plot.py`, write ANALYSIS.md.

## Verification
- Local: `run_all.sh --profile local --skip-bench` completes for bf16 + ≥1 quantized config; plots render from fake data; `collect.py` master table has `tier_axis`, bit columns, and isolation deltas.
- Smoke: every config yields eval.json + all bench jsons + KV-token count; sanity: kv-fp8 KV tokens ≈ 2× bf16; w4a16 checkpoint ~¼ size; `w8a16-fp8` loads with weight-only FP8 kernel (check vLLM log), not W8A8.
- Real: same checks + hypothesis direction visible in plots.
