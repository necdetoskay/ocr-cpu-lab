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

- Backend: **PP-StructureV3 / PaddlePaddle CPU**
- OCR language: **tr**
- Recognition family: **PP-OCRv5 multilingual (Turkish/Latin)**
- CPU threads: **8**
- Pages completed: **15/15**
- Total OCR time: **94.12 s**
- Average/page: **6.27 s**
- Previously measured model load: **65.70 s** cold-start

Observed facts:

- All 15 pages completed successfully.
- Per-page runtime ranged from **4.81 to 9.14 seconds**.
- Sustained average was **6.27 s/page**.
- Against OvisOCR2's controlled **84.77 s/page**, Paddle was approximately **13.5× faster** on clean Turkish documents.

Classification: **PASS — sustained CPU throughput target; current primary engine candidate for standard documents.**

### OCR-CPU-PADDLE-LABEL-001 — Philips device label / PP-StructureV3

Input type: photographed equipment label containing brand, model number, EAN, model ID, serial number and manufacturing text.

Critical expected fields:

- `27E2N25`
- `8721038004472`
- `27E2N2500/01`
- `UK02601033689`
- `HF4BRT2BFGPHDNE`

PP-StructureV3 output retained only a subset of the visible label text and omitted key inventory fields including EAN, Model ID and Serial Number.

Classification: **FAIL — PP-StructureV3 document mode is not suitable as the only path for dense device/equipment labels.**

### OCR-CPU-PADDLE-LABEL-002 — Philips device label / General OCR Label Mode

Configuration:

- Backend: **Paddle General OCR / PP-OCRv5 CPU**
- OCR language: **tr**
- CPU threads: **8**
- Upscale: **2.0×**
- Detection side limit: **1920**
- Recognition threshold: **0.35**
- Model load: **1.94 s**
- OCR time: **3.65 s**
- Detected text lines: **23**
- Average confidence: **0.903**

Critical field recall:

| Expected field | Status |
|---|---|
| `27E2N25` | PASS |
| `8721038004472` | PASS |
| `27E2N2500/01` | PASS |
| `UK02601033689` | PASS |
| `HF4BRT2BFGPHDNE` | PASS |

- Correct: **5/5**
- Field recall: **100.0%**
- Regression result: **PASS**

Additional observed text included brand, product type, color, EAN label, certification text and Made in China. Some non-critical labels had OCR spelling noise, but all required inventory identifiers were recovered correctly.

Classification: **PASS — preferred CPU path for device labels and dense small-text inventory images.**

## Current interpretation

A single Paddle pipeline is not sufficient for all document types, but the Paddle family is currently the strongest CPU-only solution when routed by input class:

- **Standard documents / PDFs / structured pages:** PP-StructureV3 + Turkish PP-OCRv5 multilingual recognition.
- **Device labels / serial-number photos / dense small text:** Paddle General OCR Label Mode with Turkish PP-OCRv5, 2× upscale and tuned detection.
- **Difficult cases where Paddle quality materially degrades:** OvisOCR2 remains the quality-reference/fallback candidate, accepting its much higher CPU cost.

The device-label regression is especially important because PP-StructureV3 silently omitted business-critical fields that were visually clear, while General OCR recovered **5/5 critical identifiers in 3.65 seconds**.

## Next tests

1. Add more label regressions: laptop/desktop service tag, switch/router label, printer label, UPS/monitor label and a low-light/angled photograph.
2. Add structured-document complexity cases: table-heavy, multi-column, formula, low-quality scan and rotated/photographed page.
3. Define an automatic routing heuristic between Document Mode and Label Mode; initially this can also be user-selectable in the test UI.
4. Add field-recall assertions to every inventory-label regression rather than relying only on visual quality review.
5. If the regression suite remains strong, consolidate the two Paddle modes behind one production API and keep Ovis as an optional fallback.

## Decision state

**Current CPU-only architecture leader: routed PaddleOCR.**

PP-StructureV3 is the leading standard-document parser; Paddle General OCR is the leading device-label parser. OvisOCR2 remains a fallback/research path for difficult cases rather than the default engine.
