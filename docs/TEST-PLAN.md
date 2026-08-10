# TEST PLAN — OvisOCR2 CPU

Test ID: `OCR-CPU-OVIS2-001`

## Objective

Measure whether OvisOCR2 is practically useful for local CPU-only document ingestion, with special attention to Turkish documents, structured layouts, tables, degraded scans, and reading order.

## Hard constraints

- GPU must not be used.
- No remote OCR API.
- No database.
- No manual cleanup before recording the raw OCR result.
- First benchmark uses the official Transformers model without quantization.

## Phase A — Smoke test

Start with 5 pages:

1. clean Turkish digital document
2. scanned Turkish document
3. table-heavy document
4. multi-column/complex layout
5. low-quality or photographed page

For every run record:

- file/test-case identifier
- page type
- image dimensions
- cold model-load time
- inference seconds
- total seconds
- process RAM before/after inference
- output character count
- obvious missing text
- duplicate output
- hallucinated text
- Turkish character problems
- reading-order problems
- table-structure problems

## Phase B — 20-page evaluation set

Target distribution:

| Category | Pages |
|---|---:|
| Clean Turkish | 5 |
| Scanned Turkish | 3 |
| Tables | 3 |
| Multi-column | 3 |
| Low quality / photographed | 2 |
| Handwriting | 2 |
| Mixed layout | 2 |
| **Total** | **20** |

Do not commit private or confidential source documents. `samples/` is ignored by default. Store only sanitized/public samples when redistribution rights are clear.

## Quality grading

For V0.1 use both measurable runtime data and a simple human quality grade:

- `A` — essentially correct; immediately usable
- `B` — minor errors; usable for RAG with normal validation
- `C` — meaningful omissions/order/table errors; risky without correction
- `D` — unusable result

Also explicitly flag:

- missing content
- duplicated content
- hallucinated content
- wrong reading order
- malformed tables/formulas

## Initial decision gate

After 20 pages classify the model as one of:

### PASS — interactive CPU candidate

Typical page latency is acceptable for an upload-and-wait user workflow and quality is predominantly A/B.

### PASS — batch/offline CPU candidate

Quality is predominantly A/B but latency is too high for an interactive workflow. Background/batch ingestion remains practical.

### FAIL — CPU path not worthwhile

Latency, memory usage, or quality makes the official CPU model impractical.

No fixed seconds/page threshold is imposed before the first measurement; the first run establishes empirical baseline data rather than validating a guessed target.

## Phase C — only if Phase B is promising

Compare the official baseline against CPU-friendly quantized runtimes/models. Record quality regressions as well as speed/RAM gains. Do not promote a quantized option based on speed alone.
