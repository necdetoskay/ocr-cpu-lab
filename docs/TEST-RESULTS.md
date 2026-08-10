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

### OCR-CPU-OVIS2-003 — llama.cpp GGUF Q4_K_M, 3-page sequential

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

Observed facts:

- All three pages completed naturally with `finish_reason=stop`.
- No page required the 4096-token retry path.
- Runtime varied substantially by page: **63.15–109.85 s**.
- Prompt-token count stayed fixed at **3171** for all three pages, while completion length varied.
- Page 2 was the slowest and also had the largest completion-token count, but the relationship between completion length and runtime is not yet sufficient to establish causality.
- This run was performed on a workstation under concurrent/heavy usage, so it is retained as a development baseline rather than the production-server performance result.

Classification: **PASS — multi-page pipeline is functionally stable; performance decision deferred to controlled server benchmark.**

## Current interpretation

The GGUF path is substantially more practical than the generic Transformers/Torch CPU fallback and completes pages without truncation. The 3-page test proves that page-scoped token ceilings and sequential multi-page processing work as intended.

The measured workstation average of **84.54 s/page** is too slow for interactive multi-page OCR, but the machine was under significant concurrent load. The next authoritative performance gate is therefore the Docker Compose CPU-only benchmark on the intended server hardware.

## Next tests

1. `OCR-CPU-OVIS2-004` — controlled Docker/server benchmark using the same known 1-page source.
2. `OCR-CPU-OVIS2-005` — controlled Docker/server 3-page sequential benchmark using the same document.
3. Resolution sweep on the server to measure quality/speed at lower rendered resolutions.
4. Complexity suite: clean text, dense text, table, multi-column, formula, handwriting/degraded scan.
5. Add a CPU-oriented competing OCR/document parser and run the same source pages for speed/quality comparison.

## Decision state

**OvisOCR2 remains under evaluation for CPU-only batch/offline use.**

Final classifications remain:

- PASS — interactive CPU candidate
- PASS — batch/offline CPU candidate
- FAIL — CPU path not worthwhile
