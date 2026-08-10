# OCR Routing and Regression Strategy

## Current CPU routing

The primary CPU OCR stack uses two PaddleOCR pipelines behind one UI/service.

### Document Mode

Use `PPStructureV3` for:

- PDFs
- reports
- normal office documents
- multi-page documents
- layouts/tables where document structure matters

### Label Mode

Use Paddle General OCR / PP-OCRv5 for:

- device labels
- serial numbers
- model numbers
- MAC/service-tag style identifiers
- barcode-adjacent text
- dense small-print photos

The label path uses Turkish/Latin recognition, 2x image upscale and a larger detection-side limit.

### Auto Mode

The first deterministic router is intentionally simple:

- PDF -> Document Mode
- image/photo -> Label Mode

This is conservative and observable. Users can override the selected mode. A future router may use OCR confidence, text density and document-layout signals, but should be added only after regression coverage exists.

## Regression cases

Canonical label cases live in:

`regressions/label_cases.json`

Each case defines expected critical fields and an acceptance threshold.

### DEVICE-LABEL-001

Philips monitor label. Critical fields:

- `27E2N25`
- `8721038004472`
- `27E2N2500/01`
- `UK02601033689`
- `HF4BRT2BFGPHDNE`

Observed result with General OCR Label Mode:

- OCR time: 3.65 s
- detected lines: 23
- average confidence: 0.903
- field recall: 5/5 = 100%
- classification: PASS

PP-StructureV3 missed important fields on the same image, proving that document parsing and dense-label OCR need separate routes.

## Next regression targets

Add representative cases for:

1. Laptop service tag / serial number.
2. Network switch/router label with serial and MAC address.
3. Printer label with model and serial number.
4. UPS/power-device label.
5. Low-light or angled device-label photograph.
6. Small-font label with glare/noise.

For each case, record exact expected identifiers rather than relying only on subjective OCR quality.

## Fallback policy under evaluation

OvisOCR2 remains available as a high-quality but slow CPU fallback. It should only be invoked when Paddle output fails a measurable quality gate or a difficult-document category is proven to benefit from Ovis.
