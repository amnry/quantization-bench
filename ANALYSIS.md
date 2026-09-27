# KV cache FP8 accuracy anomaly (0.5B smoke)

**Status:** unresolved, deprioritized — to be re-checked at 7B.

## Observation

On Qwen2.5-0.5B-Instruct, MMLU (limit=200/subtask, 57 subtasks, n=11,400) collapses
to ~chance level for both KV cache FP8 formats:

| config | MMLU acc,none | delta vs bf16 |
|---|---|---|
| bf16 | 0.474 | — |
| kv-fp8-e4m3 (32-sample calib) | 0.243 | −0.232 |
| kv-fp8-e4m3 (512-sample calib) | 0.247 | −0.227 |
| kv-fp8-e5m2 (no calibrated scales) | 0.243 | −0.231 |

~0.25 is chance level on 4-way multiple choice — the model is effectively guessing
under FP8 KV cache.

## What was ruled out

- **Calibration quantity:** 32 vs 512 samples produced nearly identical k_scale/v_scale
  values (layer 0 exactly: k_scale=0.291016, v_scale=0.000679 in both) and identical
  MMLU collapse. Not the cause.
- **Missing/degenerate scales:** checkpoint contains k_scale/v_scale per layer (24/24),
  values are non-degenerate (no NaN/Inf/default-1.0). vLLM log shows no "scaling factor
  1.0" fallback warning.
- **Wrong backend/config wiring:** vLLM correctly selects FLASHINFER when FP8 KV is
  requested (vs FLASH_ATTN otherwise); config.json / kv_cache_dtype flags all correct.
- e5m2 needs no calibrated scales at all, yet collapses identically to e4m3 — this is
  the strongest signal that the bug isn't in scale computation.

## Suspected root cause

Not yet confirmed. Leading hypothesis: a bug somewhere in the FP8 KV cache eval path
itself (not necessarily quantization fidelity) — e.g. an interaction between vLLM's
FP8 KV cache + FLASHINFER backend and lm-eval's loglikelihood scoring on this small
model/short-context combination, rather than genuine quantization-induced accuracy
loss. A ~50-point MMLU collapse to chance level for 8-bit KV cache (2-3 mantissa bits)
is far outside what's reported elsewhere for this technique (usually <1-2 points).

Deprioritized per decision to stop investigating at 0.5B and check behavior at 7B
instead — if the 7B run shows normal (<2pt) KV deltas, this confirms a 0.5B-specific
eval-path issue rather than a quantization defect.

# RULER: never ran (missing deps)

`wonderwords` and `nltk` are not in `requirements-lock.txt` / not installed on the
pod, so every RULER call in `scripts/eval.py:run_ruler()` fails immediately with:

    Please install the `wonderwords` and `nltk` packages to run this script.
    You can install them with `pip install lm_eval["ruler"]` or `pip install
    wonderwords nltk`.

This happened silently (caught by `run_ruler()`'s try/except, recorded as
`{"error": ...}` per length) in **both** the 0.5B smoke run and the 7B full run —
`ruler_4096`/`ruler_16384` have never contained real scores in any result so far.

Confirmed impact is negligible: the import fails before any model call, so it costs
no meaningful time per config (no "ruler" log lines appear at all — instant failure),
and `eval.py` exits 0 regardless, so it never marks a config `FAILED`.

**Decision:** RULER is dropped for this run — out of scope for budget and to keep
every config's eval settings identical to bf16 (installing the deps mid-run would
make later configs run RULER while bf16 didn't, breaking the comparison). Not fixed.
