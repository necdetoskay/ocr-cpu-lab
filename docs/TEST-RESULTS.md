# TEST RESULTS — OCR CPU Lab

Status: **IN PROGRESS**

This file is the canonical human-readable result log. Measured values are kept separate from estimates.

## OvisOCR2 CPU

### OCR-CPU-OVIS2-001 — Transformers / Torch CPU baseline

- Backend: Hugging Face Transformers + PyTorch CPU fallback
- Input: 1-page PDF, rendered at 1489x2105
- Model load: ~6.5 s
- System CPU during inference: ~89–93%
- Process RAM during inference: ~1.0–1.1 GB observed
- Result: inference exceeded 200 seconds for one page and was stopped before a useful completed result
- Classification: **FAIL — interactive CPU path not worthwhile**

### OCR-CPU-OVIS2-002 — llama.cpp GGUF Q4_K_M, single page

- Total request: **64.90 s**
- Prompt tokens: **3171**
- Completion tokens: **1393**
- Finish reason: `stop`
- Output characters: **4267**
- User-observed extraction quality: **excellent**

### OCR-CPU-OVIS2-003 — workstation 3-page sequential

- Pages completed: **3/3**
- Total OCR time: **253.63 s**
- Average/page: **84.54 s**

| Page | Seconds | Prompt tok | Completion tok | Chars |
|---:|---:|---:|---:|---:|
| 1 | 63.15 | 3171 | 1393 | 4267 |
| 2 | 109.85 | 3171 | 1543 | 4567 |
| 3 | 80.62 | 3171 | 832 | 2126 |

### OCR-CPU-OVIS2-004 — Docker server before memory fix

- CPU: **8 vCPU**, Intel Xeon Cascade Lake under KVM
- RAM: **11 GiB**
- Host was under severe memory/swap pressure
- Partial page timings: **155.27 s**, **216.05 s**
- Classification: **INVALID FOR PERFORMANCE DECISION**

### OCR-CPU-OVIS2-005 — Docker server, 32 GiB RAM + constrained llama.cpp

- CPU: **8 vCPU**, Intel Xeon Cascade Lake under KVM
- RAM: **32 GiB**
- `--parallel 1`
- `--ctx-size 8192`
- `--threads 8`
- `--threads-batch 8`
- Pages completed: **3/3**
- Total OCR time: **254.30 s**
- Average/page: **84.77 s**

| Page | Seconds | Prompt tok | Completion tok | Chars |
|---:|---:|---:|---:|---:|
| 1 | 90.88 | 3171 | 1393 | 4260 |
| 2 | 99.75 | 3171 | 1528 | 4570 |
| 3 | 63.67 | 3171 | 836 | 2128 |

Classification: **PASS — stable CPU-only batch path; FAIL — interactive performance target.**

## PaddleOCR / PP-StructureV3 CPU

### OCR-CPU-PADDLE-001 — default recognition model

- Backend: PP-StructureV3 / PaddlePaddle CPU
- CPU threads: **8**
- Pages completed: **3/3**
- Total OCR time: **115.79 s**
- Average/page: **38.60 s**
- Model load: **14.16 s**
- Page timings: **95.97 s**, **11.98 s**, **7.84 s**
- User-observed Turkish quality: **FAIL — Turkish characters/recognition quality poor**
- Interpretation: speed was promising after warm-up, but the default recognition model was not acceptable for Turkish.

### OCR-CPU-PADDLE-002 — Turkish PP-OCRv5 multilingual recognition

Configuration:

- Backend: **PP-StructureV3 / PaddlePaddle CPU**
- OCR language: **tr**
- Recognition family: **PP-OCRv5 multilingual (Turkish/Latin)**
- CPU threads: **8**
- Pages completed: **3/3**
- Model load: **65.70 s** cold-start
- Total OCR time: **27.68 s**
- Average/page: **9.23 s**

| Page | Seconds | Results | Chars |
|---:|---:|---:|---:|
| 1 | 12.30 | 1 | 3178 |
| 2 | 9.11 | 1 | 4164 |
| 3 | 6.28 | 1 | 1790 |

User-observed extraction quality: **excellent / effectively flawless on the tested Turkish document**.

Observed facts:

- Explicit Turkish recognition fixed the character/recognition problems seen with the default model.
- All three pages produced valid structured output.
- Sustained page processing was **6.28–12.30 s/page**.
- Average OCR time was **9.23 s/page**, approximately **9.2× faster** than the controlled OvisOCR2 server average of **84.77 s/page** on the same 3-page document.
- The **65.70 s model-load time is cold-start cost**, not per-page inference time. Once the container/model remains warm, the relevant steady-state metric is the per-page OCR time.

Classification: **PASS — primary CPU-only candidate for Turkish document parsing.**

## Current interpretation

For the tested clean Turkish documents on the intended 8-vCPU / 32-GiB server, PaddleOCR PP-StructureV3 with explicit Turkish PP-OCRv5 multilingual recognition currently provides the best balance of speed and quality.

OvisOCR2 remains a useful quality reference and possible fallback for document types where generative end-to-end parsing materially outperforms PaddleOCR, but its approximately **85 s/page** CPU runtime is not competitive for the primary path on this hardware.

PaddleOCR should therefore become the current **primary candidate**, subject to complexity/regression testing.

## Next tests

1. `OCR-CPU-PADDLE-003` — warm repeat of the same 3-page PDF to confirm steady-state reproducibility without cold-start effects.
2. `OCR-CPU-PADDLE-004` — 10-page throughput test.
3. Complexity suite: table-heavy, multi-column, formula, handwriting, low-quality/degraded scan, rotated/photographed page.
4. Compare PaddleOCR vs OvisOCR2 on only the difficult cases where Paddle quality degrades.
5. Define routing/fallback policy if Ovis materially wins any difficult-document category.

## Decision state

**Current leader: PaddleOCR PP-StructureV3 + Turkish PP-OCRv5 multilingual recognition.**

OvisOCR2 remains available as a fallback/research candidate rather than the default CPU OCR engine.
