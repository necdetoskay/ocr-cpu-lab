# TEST RESULTS — OvisOCR2 CPU

Test family: `OCR-CPU-OVIS2`

Status: **IN PROGRESS**

This file is the canonical human-readable result log. Measured values are kept separate from estimates.

## Confirmed runs

### OCR-CPU-OVIS2-001 — Transformers / Torch CPU baseline

- Backend: Hugging Face Transformers + PyTorch CPU fallback
- Input: 1-page PDF, rendered at 1489x2105
- Model load: ~6.5 s
- System CPU during inference: ~89–93%
- Process RAM during inference: ~1.0–1.1 GB observed
- Result: inference exceeded 200 seconds for one page and was stopped before a useful completed result
- Classification: **FAIL — interactive CPU path not worthwhile**
- Important note: the runtime reported that the optimized fast path was unavailable and it fell back to the Torch implementation.

### OCR-CPU-OVIS2-002 — llama.cpp GGUF Q4_K_M, single page

- Backend: llama.cpp
- Quantization: Q4_K_M
- GPU layers: 0
- Multimodal projector GPU offload: disabled
- Input: 1-page PDF, 1489x2105
- Total request: **64.90 s**
- Prompt tokens: **3171**
- Completion tokens: **1393**
- Max output tokens: **2048**
- Finish reason: `stop`
- Token limit hit: **NO**
- Output characters: **4267**
- User-observed extraction quality: **excellent**
- Classification: **PASS — technically viable CPU-only path, currently too slow for interactive multi-page use**

### OCR-CPU-OVIS2-003 — llama.cpp GGUF Q4_K_M, 3-page sequential workstation run

- Backend: llama.cpp / Q4_K_M
- Server: local `127.0.0.1:8080`
- Pages completed: **3/3**
- Total OCR time: **253.63 s**
- Average/page: **84.54 s**
- Adaptive retries (2048→4096): **0**

| Page | Seconds | Prompt tok | Completion tok | Max tok | Finish | Retry | Chars |
|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 63.15 | 3171 | 1393 | 2048 | stop | NO | 4267 |
| 2 | 109.85 | 3171 | 1543 | 2048 | stop | NO | 4567 |
| 3 | 80.62 | 3171 | 832 | 2048 | stop | NO | 2126 |

Classification: **PASS — multi-page pipeline is functionally stable; performance too slow for interactive use.**

### OCR-CPU-OVIS2-004 — Docker server baseline before memory fix

Server profile observed during the initial Docker benchmark:

- CPU: **8 vCPU**, Intel Xeon Cascade Lake under KVM
- RAM: **11 GiB**
- Available RAM during test: approximately **382 MiB**
- Swap: **2 GiB**, effectively exhausted during the test
- Initial llama.cpp configuration exposed 4 slots with extremely large context capacity

Observed partial page timings:

| Page | Seconds | Prompt tok | Completion tok | Max tok | Finish | Retry | Chars |
|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 155.27 | 3171 | 1393 | 2048 | stop | NO | 4260 |
| 2 | 216.05 | 3171 | 1528 | 2048 | stop | NO | 4570 |

The run also showed generation throughput around **0.53 token/s** on a later page. This run is classified as **INVALID FOR PERFORMANCE DECISION** because the host was under severe memory/swap pressure.

### OCR-CPU-OVIS2-005 — Docker server, 32 GiB RAM + constrained llama.cpp

Controlled server configuration:

- CPU: **8 vCPU**, Intel Xeon Cascade Lake under KVM
- RAM: **32 GiB**
- llama.cpp: Q4_K_M, CPU only
- GPU layers: **0**
- multimodal projector GPU offload: disabled
- `--parallel 1`
- `--ctx-size 8192`
- `--threads 8`
- `--threads-batch 8`
- Pages completed: **3/3**
- Total OCR time: **254.30 s**
- Average/page: **84.77 s**
- Adaptive retries (2048→4096): **0**

| Page | Seconds | Prompt tok | Completion tok | Max tok | Finish | Retry | Chars |
|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 90.88 | 3171 | 1393 | 2048 | stop | NO | 4260 |
| 2 | 99.75 | 3171 | 1528 | 2048 | stop | NO | 4570 |
| 3 | 63.67 | 3171 | 836 | 2048 | stop | NO | 2128 |

Observed facts:

- Increasing RAM from 11 GiB to 32 GiB and removing the oversized multi-slot context eliminated the catastrophic 155–216 s/page behavior.
- The optimized server average (**84.77 s/page**) is almost identical to the workstation 3-page average (**84.54 s/page**).
- All pages completed naturally with `finish_reason=stop`; no adaptive retry was required.
- The result strongly suggests the remaining bottleneck is compute/vision-generation throughput rather than RAM capacity.

Classification: **PASS — stable CPU-only batch path; FAIL — interactive performance target.**

## Current interpretation

The CPU-only OvisOCR2 GGUF path is technically stable, produces excellent output on the tested clean documents, supports multi-page page-scoped processing, and does not require GPU acceleration. However, on both the workstation and the intended 8-vCPU Cascade Lake server, sustained performance is approximately **85 seconds per page** for the current 1489x2105 render setting.

The initial server slowdown was caused largely by memory/swap pressure and overly large llama.cpp context/parallel defaults. After correcting those issues and increasing RAM, the server converged to the same performance class as the workstation. More RAM alone therefore does not solve the remaining throughput problem.

## Next tests

1. `OCR-CPU-OVIS2-006` — resolution sweep using the same known page at lower render resolutions/DPI.
2. Compare extraction quality and speed across those resolution settings.
3. Complexity suite at the best resolution: clean text, dense text, table, multi-column, formula, handwriting/degraded scan.
4. Add a CPU-oriented competing OCR/document parser and run the same source pages for speed/quality comparison.

## Decision state

**OvisOCR2 is currently classified as a viable CPU-only batch/offline parser, but not an interactive CPU OCR engine on the tested hardware.**

Final decision may change if the resolution sweep preserves quality while reducing runtime substantially.
