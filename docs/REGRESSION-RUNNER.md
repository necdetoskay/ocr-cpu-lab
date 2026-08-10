# Automated OCR regression runner

The regression runner turns the manual OCR experiments into a repeatable gate. Canonical expectations live in Git; source documents/photos can remain private on the test server.

## Private sample storage

On the server, copy regression inputs into:

```text
ocr-cpu-lab/regression-samples/
```

Current filenames expected by `regressions/suite.json`:

```text
doc-text-001.pdf
device-label-001.jpg
```

The directory is gitignored and mounted read-only into the Paddle container at `/data/regression-samples`.

## Run the suite

After rebuilding `paddle-ui`:

```bash
docker compose -p ocr-cpu-lab -f docker-compose.cpu.yml exec paddle-ui \
  python scripts/run_regression_suite.py
```

To make missing samples fail the gate instead of being reported as SKIP:

```bash
docker compose -p ocr-cpu-lab -f docker-compose.cpu.yml exec paddle-ui \
  python scripts/run_regression_suite.py --strict-missing
```

## Reports

The runner writes:

```text
artifacts/regression/latest.json
artifacts/regression/latest.md
```

The `artifacts` directory is mounted back to the host and ignored by Git.

Each case reports:

- pipeline
- PASS / FAIL / SKIP
- expected-field or marker recall
- runtime
- recognized item count
- per-marker/per-field result
- raw OCR output in JSON

## Current gates

### DOC-TEXT-001

Pipeline: `document_text` (General OCR / PP-OCRv5 Turkish)

Acceptance: 100% recall for the expected section markers (`37.1`, `37.2`, `Madde 38`, `38.1`, `38.2`, `38.3`).

### DEVICE-LABEL-001

Pipeline: `label` (General OCR / PP-OCRv5 Turkish, 2x upscale)

Acceptance: 100% recall for all five critical identifiers. Critical fields must all match.

## Expansion plan

Add one real sample and one expectation block per new category. Planned categories:

- `DOC-TABLE-001` — table structure/content
- `DOC-MULTICOL-001` — multi-column reading order
- `DOC-DEGRADED-001` — low-quality scan
- `DOC-ROTATED-001` — rotated/photographed page
- `DEVICE-LABEL-002+` — laptop, switch/router, printer, UPS/device labels
- `HANDWRITING-001` — handwriting quality reference
- `FORMULA-001` — formula/document parser comparison

Do not commit sensitive customer/device files. Keep them in `regression-samples/` on the controlled test server.
