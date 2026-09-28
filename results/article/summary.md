| Config | GSM8K | dGSM8K (pp) | MMLU | dMMLU (pp) | KV tokens | KV vs BF16 | Decode c1 (tok/s) | Decode c64 (tok/s) | Decode c1 speedup | Decode c64 speedup | Prefill c64 (tok/s) | Prefill c64 speedup | Longctx c64 TTFT (s) | GPU hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BF16 (baseline) | 80.59% | +0.00 | 74.26% | +0.00 | 467,600 | +0% | 49.1 | 1864.0 | 1.00x | 1.00x | 11615.1 | 1.00x | 14.5 | 2.65 |
| INT4 weights | 76.80% | -3.79 | 73.27% | -0.98 | 629,360 | +35% | 133.9 | 4388.6 | 2.73x | 2.35x | 12054.2 | 1.04x | 3.7 | 2.34 |
| FP8 weights + act. | 80.14% | -0.45 | 73.80% | -0.46 | 568,960 | +22% | 73.4 | 2992.7 | 1.50x | 1.61x | 15218.5 | 1.31x | 3.5 | 2.20 |

**Total: 7.19 GPU-hours, $7.98 at $1.11/hr**
