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

### OCR-CPU-OVIS2-002 — llama.cpp GGUF Q4_K_M

- Backend: llama.cpp
- Quantization: Q4_K_M
- GPU layers: 0
- Multimodal projector GPU offload: disabled
- Input: same 1-page PDF, 1489x2105
- Total request: **64.90 s**
- Prompt tokens: **3171**
- Completion tokens: **1393**
- Max output tokens: **2048**
- Finish reason: `stop`
- Token limit hit: **NO**
- Output characters: **4267**
- User-observed extraction quality: **excellent**
- Classification: **PASS — technically viable CPU-only path, currently too slow for interactive multi-page use**

## Current interpretation

The GGUF path is substantially more practical than the generic Transformers/Torch CPU fallback and completes the page without truncation. However, approximately 65 seconds per page remains the primary production concern.

The next gate is multi-page behavior. PDFs will be processed page-by-page with a default 2048-token ceiling per page. A page is retried at 4096 only when llama.cpp reports `finish_reason=length` or the completion reaches the configured ceiling. This prevents a 10- or 50-page document from sharing one global output-token budget.

## Next tests

1. `OCR-CPU-OVIS2-003` — 3-page sequential document test.
2. `OCR-CPU-OVIS2-004` — 10-page sequential throughput test if the 3-page run is stable.
3. Add a CPU-oriented competing OCR/document parser and run the same source pages for speed/quality comparison.

## Decision state

**OvisOCR2 remains under evaluation for CPU-only batch/offline use.**

Final classifications remain:

- PASS — interactive CPU candidate
- PASS — batch/offline CPU candidate
- FAIL — CPU path not worthwhile
