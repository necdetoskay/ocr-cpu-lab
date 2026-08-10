# TEST RESULTS — OCR CPU Lab

Status: **IN PROGRESS**

This file is the canonical human-readable result log. Measured values are kept separate from estimates.

## OvisOCR2 CPU

### OCR-CPU-OVIS2-001 — Transformers / Torch CPU baseline

- Backend: Hugging Face Transformers + PyTorch CPU fallback
- Input: 1-page PDF, rendered at 1489x2105
- Result: inference exceeded 200 seconds before a useful completed result
- Classification: **FAIL — interactive CPU path not worthwhile**

### OCR-CPU-OVIS2-002 — llama.cpp GGUF Q4_K_M, single page

- Total request: **64.90 s**
- Prompt tokens: **3171**
- Completion tokens: **1393**
- Output characters: **4267**
- User-observed extraction quality: **excellent**

### OCR-CPU-OVIS2-003 — workstation 3-page sequential

- Pages completed: **3/3**
- Total OCR time: **253.63 s**
- Average/page: **84.54 s**

### OCR-CPU-OVIS2-004 — Docker server before memory fix

- CPU: **8 vCPU**, Intel Xeon Cascade Lake under KVM
- RAM: **11 GiB**
- Host was under severe memory/swap pressure
- Partial page timings: **155.27 s**, **216.05 s**
- Classification: **INVALID FOR PERFORMANCE DECISION**

### OCR-CPU-OVIS2-005 — Docker server, 32 GiB RAM + constrained llama.cpp

- CPU: **8 vCPU**, Intel Xeon Cascade Lake under KVM
- RAM: **32 GiB**
- `--parallel 1`, `--ctx-size 8192`, `--threads 8`, `--threads-batch 8`
- Pages completed: **3/3**
- Total OCR time: **254.30 s**
- Average/page: **84.77 s**
- User-observed extraction quality: **excellent**

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

### OCR-CPU-PADDLE-002 — Turkish PP-OCRv5 multilingual recognition

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

Classification: **PASS — primary CPU-only candidate for Turkish document parsing.**

### OCR-CPU-PADDLE-003 — 15-page sustained warm throughput

Configuration remained unchanged from the successful Turkish run:

- Backend: **PP-StructureV3 / PaddlePaddle CPU**
- OCR language: **tr**
- Recognition family: **PP-OCRv5 multilingual (Turkish/Latin)**
- CPU threads: **8**
- Pages completed: **15/15**
- Total OCR time: **94.12 s**
- Average/page: **6.27 s**
- Previously measured model load: **65.70 s** (cold-start; not part of the 94.12 s OCR total)

| Page | Seconds | Results | Chars |
|---:|---:|---:|---:|
| 1 | 9.14 | 1 | 3382 |
| 2 | 8.06 | 1 | 4165 |
| 3 | 5.68 | 1 | 1794 |
| 4 | 8.16 | 1 | 1247 |
| 5 | 5.38 | 1 | 2966 |
| 6 | 5.95 | 1 | 2202 |
| 7 | 5.53 | 1 | 1075 |
| 8 | 5.58 | 1 | 2539 |
| 9 | 4.95 | 1 | 1307 |
| 10 | 7.51 | 1 | 3071 |
| 11 | 5.44 | 1 | 1372 |
| 12 | 6.39 | 1 | 3015 |
| 13 | 5.11 | 1 | 835 |
| 14 | 6.43 | 1 | 2844 |
| 15 | 4.81 | 1 | 388 |

Observed facts:

- All **15 pages completed successfully** and every page returned one result object.
- No progressive slowdown is visible across the document; later pages remain in the same or faster performance band.
- Per-page runtime ranged from **4.81 to 9.14 seconds**.
- Sustained average improved to **6.27 s/page**, demonstrating that the earlier 9.23 s/page 3-page run was conservative for warm sustained throughput.
- Against the controlled OvisOCR2 average of **84.77 s/page**, Paddle's sustained 6.27 s/page is approximately **13.5× faster** on the tested server.
- User had already verified the Turkish recognition output as effectively flawless on the preceding Turkish-model run; document-complexity-specific quality validation remains outstanding.

Classification: **PASS — sustained CPU throughput target; current primary engine candidate.**

## Current interpretation

For clean Turkish documents on the intended 8-vCPU / 32-GiB server, PaddleOCR PP-StructureV3 with Turkish PP-OCRv5 multilingual recognition is now clearly ahead of OvisOCR2 for the primary CPU path.

The 15-page run demonstrates both throughput and stability: **94.12 seconds total / 6.27 seconds per page**, with no failed pages or throughput collapse. OvisOCR2 remains valuable as a quality reference and possible difficult-document fallback, but its approximately **84.77 s/page** runtime is about **13.5× slower** than Paddle's sustained result on this workload.

## Next tests

1. `OCR-CPU-PADDLE-004` — complexity suite rather than another clean-text volume test.
2. Include table-heavy, multi-column, formula, handwriting, low-quality/degraded scan, rotated page and photographed page cases.
3. Record both speed and human-observed extraction quality for every category.
4. Run OvisOCR2 only on cases where Paddle quality is materially degraded, to measure whether a fallback path is justified.
5. If complexity tests pass, define the production-oriented Paddle service profile: healthcheck, persistent model cache, startup readiness, resource limits and API boundary.

## Decision state

**Current leader: PaddleOCR PP-StructureV3 + Turkish PP-OCRv5 multilingual recognition.**

Paddle has passed clean-document Turkish quality, 3-page functional testing, and 15-page sustained CPU throughput. The remaining gate is difficult-document quality.
