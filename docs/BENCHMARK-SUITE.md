# OCR CPU Benchmark Suite

This suite turns ad-hoc OCR trials into repeatable acceptance gates.

## Routing under test

- **Text OCR** — Paddle General OCR / PP-OCRv5 (`lang=tr`) for ordinary text PDFs and completeness-sensitive documents.
- **Structured** — PP-StructureV3 for tables, multi-column layouts and structure-aware Markdown.
- **Label OCR** — Paddle General OCR with 2x upscale and tuned detection for device labels, serial/model numbers and dense small text.
- **OvisOCR2** — quality reference / fallback when Paddle quality is materially insufficient.

## Canonical regression cases

### DOC-TEXT-001 — Turkish procurement text completeness

Purpose: detect silent line loss on a clean, single-column Turkish legal/procurement page.

Expected markers:

- `37.1`
- `37.2`
- `Madde 38`
- `38.1`
- `38.2`
- `38.3`

Acceptance: **100% marker recall**.

Known finding: PP-StructureV3 previously omitted `Madde 38` and `38.2`; Text OCR recovered the missing content. Therefore Text OCR is the recommended mode for this case.

### DEVICE-LABEL-001 — Philips monitor label

Expected critical fields:

- `27E2N25`
- `8721038004472`
- `27E2N2500/01`
- `UK02601033689`
- `HF4BRT2BFGPHDNE`

Acceptance: **100% field recall**.

Measured Label OCR result: **5/5 fields, 100% recall, 3.65 s OCR time** on the test server.

## Planned cases

- `DOC-TABLE-001` — dense table with merged/empty cells.
- `DOC-MULTICOLUMN-001` — two- or three-column reading order.
- `DOC-DEGRADED-001` — low-resolution / compressed scan.
- `DOC-ROTATED-001` — rotated or photographed page.
- `LABEL-SWITCH-001` — model, serial and MAC address.
- `LABEL-LAPTOP-001` — service tag / serial / model.
- `LABEL-PRINTER-001` — serial / product number.
- `HANDWRITE-001` — handwritten note quality reference.

## Metrics

Every benchmark should record at minimum:

- backend / mode,
- CPU-only confirmation,
- model load or cold-start time,
- OCR/inference time,
- pages processed,
- expected-field or expected-marker recall where a ground truth exists,
- human-observed quality for layout/table/handwriting cases,
- PASS / FAIL classification.

Performance alone cannot pass a test if required information is silently lost.
