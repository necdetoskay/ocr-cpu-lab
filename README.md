# OCR CPU Lab

Minimal, reproducible CPU-only laboratory for testing modern OCR/document-parsing models on local hardware.

## First target: OvisOCR2

The initial experiment evaluates `ATH-MaaS/OvisOCR2` on CPU only. The model is a compact end-to-end page-level document parser that produces Markdown from document-page images.

This repository intentionally avoids production architecture. There is no database, authentication layer, API service, Docker requirement, or persistent backend. The goal is to answer one question with evidence:

> Is OvisOCR2 accurate and fast enough to be useful on CPU-only hardware for local document ingestion?

## V0.1 scope

- Gradio local UI
- JPG / JPEG / PNG upload
- PDF upload, first page only
- Explicit CPU-only execution
- Markdown OCR result
- model-load time
- OCR inference time
- total request time
- process RAM before/after inference
- input dimensions and output character count
- repeatable test-plan and result log

## Quick start

Recommended: Python 3.11 or 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

The first run downloads the model weights from Hugging Face and therefore requires internet access. Later runs use the local Hugging Face cache.

## CPU guarantee

The application sets `CUDA_VISIBLE_DEVICES=-1`, loads the model onto `cpu`, and displays the selected device in the UI. GPU acceleration is intentionally excluded from V0.1 so the results represent CPU-only behavior.

## Repository layout

```text
ocr-cpu-lab/
├── app.py
├── requirements.txt
├── src/
│   ├── ocr.py
│   ├── pdf.py
│   └── metrics.py
├── samples/
├── results/
└── docs/
    ├── TEST-PLAN.md
    └── TEST-RESULTS.md
```

## Current status

**V0.1 — CPU smoke-test harness: implementation started.**

See `docs/TEST-PLAN.md` before recording benchmark results.

## Model/license note

OvisOCR2 is published as `ATH-MaaS/OvisOCR2` under Apache-2.0. This repository does not redistribute model weights.
