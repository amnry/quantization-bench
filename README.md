# Quantization Bench: which part of an LLM should you quantize?

"Quantize your model, it runs faster." You hear this all the time. But faster at *what*?

A language model has several parts you can shrink to lower precision, and each one speeds up a different part of the work. This project measures which part helps where, and how much accuracy each one costs.

**Status:** smoke test complete ✅ · full run on Qwen2.5-7B in progress ⏳

---

## The idea in 60 seconds

When a model answers you, it does two kinds of work:

- **Reading your prompt** (called *prefill*). Lots of math at once. Speed here depends on how fast the GPU can compute.
- **Writing the answer, one word at a time** (called *decode*). Little math per step, but the whole model is read from memory every step. Speed here depends on how fast the GPU can move data.

There are three main things you can shrink:

| What you shrink | What it is | Where I expect it to help |
|---|---|---|
| **Weights** | The model's learned numbers | Writing answers, since there's less data to move each step |
| **Activations** | The intermediate numbers during the math | Reading long prompts and serving many users at once |
| **KV cache** | The model's short-term memory of the conversation | Long documents and many users at once, since more conversations fit in memory |

I also test shrinking only the **attention projection layers**, to see how sensitive that part of the model is compared to shrinking everything.

---

## What I test

10 versions of the same model:

| Version | What's shrunk |
|---|---|
| `bf16` | Nothing. This is the baseline everything is compared to. |
| `w8a16` | Weights to 8-bit |
| `w4a16` | Weights to 4-bit |
| `w8a16-fp8` | Weights to 8-bit floating point. A reference point used to separate the activation effect. |
| `w8a8-fp8` | Weights and activations to 8-bit floating point |
| `w8a8-int8` | Weights and activations to 8-bit integers (harder to get right) |
| `kv-fp8-e4m3` | KV cache to 8-bit |
| `kv-fp8-e5m2` | KV cache to a different 8-bit format with less precision |
| `attn-proj-w8` | Attention projection weights only, to 8-bit |
| `attn-proj-w4` | Attention projection weights only, to 4-bit |

---

## How I measure

**Speed.** Each part of the model gets a test designed to stress it:

- **Decode test:** short prompt, long answer. Tests weights.
- **Prefill test:** very long prompt, short answer. Tests activations.
- **Long context test:** long prompts from many users at once. Tests the KV cache.

Each test runs at several load levels, from 1 user up to 256 at the same time.

**Accuracy.** Three standard tests:

- **GSM8K:** grade-school math word problems. The most sensitive to quantization damage.
- **MMLU:** general knowledge across 57 subjects.
- **RULER:** finding information in very long documents. The only one that really stresses the KV cache.

---

## Smoke test results (Qwen2.5-0.5B)

Before spending money on a big GPU, I ran all 10 versions on a tiny 0.5B model to make sure the pipeline works end to end. All 10 completed.

| Version | Model size | vs baseline | Memory capacity (tokens) | vs baseline |
|---|---|---|---|---|
| `bf16` | 958 MB | baseline | 3,347,584 | baseline |
| `w8a16` | 617 MB | −36% | 3,375,488 | +0.8% |
| `w4a16` | 451 MB | −53% | 3,391,184 | +1.3% |
| `w8a16-fp8` | 617 MB | −36% | 3,354,448 | +0.2% |
| `w8a8-fp8` | 617 MB | −36% | 3,354,448 | +0.2% |
| `w8a8-int8` | 617 MB | −36% | 3,354,336 | +0.2% |
| `kv-fp8-e4m3` | 958 MB | same | 6,695,856 | **+100%** |
| `kv-fp8-e5m2` | 958 MB | same | 6,695,856 | **+100%** |
| `attn-proj-w8` | 916 MB | −4% | 3,351,280 | +0.1% |
| `attn-proj-w4` | 896 MB | −6% | 3,353,184 | +0.2% |

*Memory capacity is how many tokens of conversation fit in GPU memory at once. More capacity means more users or longer documents before the server has to slow down.*

### Why most of these numbers look small

**This is expected, and it's a limitation of the tiny model, not of quantization.**

- **Shrinking KV cache doubled memory capacity.** This works at any model size, and it's the clearest result so far.
- **Shrinking weights barely helped capacity (+1%).** The whole 0.5B model is under 1 GB on a 24 GB GPU, so saving half a gigabyte barely matters. On the 7B model, weights are about 15 GB on a 48 GB GPU, so 4-bit weights should free around 10 GB. That's a big jump in capacity.
- **4-bit weights only cut the size by 53%, not 75%.** Some parts of the model (the embedding layers) are never shrunk, and in a tiny model they make up a large share of its size. In the 7B model they're a much smaller share, so the savings will be closer to the full amount.
- **Speed and accuracy aren't reported for the smoke test.** It used only 20 test questions and 20 requests per load level, and a model this small is limited by overhead rather than by memory or compute. Those numbers can't tell the versions apart, so I don't draw conclusions from them.

---

## What to expect from the 7B run

The full run uses **Qwen2.5-7B-Instruct on an NVIDIA L40S (48 GB)**, with full test sets. What I expect to see:

- **4-bit weights** give the fastest answers for a single user.
- **8-bit weights and activations** catch up or pull ahead when many users are served at once.
- **8-bit KV cache** fits about twice as many conversations and keeps throughput high on long documents.
- **Accuracy** drops very little for most versions. 8-bit integer activations are the most likely to show damage.

Results, charts, and a full writeup will be added here when the run finishes.

---

## Run it yourself

Everything is open source and runs on a single rented GPU.

```bash
git clone https://github.com/amnry/quantization-bench
cd quantization-bench
bash scripts/setup_pod.sh
bash scripts/run_all.sh --model target --profile full
```

Use `--model qwen2.5-0.5b --profile smoke` for a quick test run first (about an hour on an L40S).

**Built with:** [llm-compressor](https://github.com/vllm-project/llm-compressor) for quantization, [vLLM](https://github.com/vllm-project/vllm) for serving, and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for accuracy tests.

---

Questions or ideas? Open an issue.
