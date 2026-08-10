from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image

from src.pdf import document_page_count, load_document_page

PORT = int(os.getenv("PADDLE_GRADIO_PORT", "7862"))
CPU_THREADS = int(os.getenv("PADDLE_CPU_THREADS", "8"))
OCR_LANG = os.getenv("PADDLE_OCR_LANG", "tr")
UPSCALE = float(os.getenv("PADDLE_LABEL_UPSCALE", "2.0"))
DET_SIDE_LEN = int(os.getenv("PADDLE_LABEL_DET_SIDE_LEN", "1920"))
REC_SCORE_THRESH = float(os.getenv("PADDLE_LABEL_REC_SCORE_THRESH", "0.35"))
REGRESSION_PATH = Path(os.getenv("OCR_REGRESSION_PATH", "/app/regressions/label_cases.json"))

_doc_pipeline: PPStructureV3 | None = None
_label_pipeline: PaddleOCR | None = None
_doc_load_seconds: float | None = None
_label_load_seconds: float | None = None


def _load_regressions() -> dict:
    if not REGRESSION_PATH.exists():
        return {}
    return json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))


REGRESSIONS = _load_regressions()


def get_doc_pipeline() -> PPStructureV3:
    global _doc_pipeline, _doc_load_seconds
    if _doc_pipeline is None:
        print(f"[ocr-cpu-lab] Loading PP-StructureV3 lang={OCR_LANG} on CPU...", flush=True)
        started = time.perf_counter()
        _doc_pipeline = PPStructureV3(
            device="cpu",
            cpu_threads=CPU_THREADS,
            lang=OCR_LANG,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _doc_load_seconds = time.perf_counter() - started
        print(f"[ocr-cpu-lab] PP-StructureV3 ready in {_doc_load_seconds:.2f}s", flush=True)
    return _doc_pipeline


def get_label_pipeline() -> PaddleOCR:
    global _label_pipeline, _label_load_seconds
    if _label_pipeline is None:
        print(f"[ocr-cpu-lab] Loading General OCR lang={OCR_LANG} on CPU...", flush=True)
        started = time.perf_counter()
        _label_pipeline = PaddleOCR(
            lang=OCR_LANG,
            ocr_version="PP-OCRv5",
            device="cpu",
            cpu_threads=CPU_THREADS,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=DET_SIDE_LEN,
            text_det_limit_type="max",
            text_rec_score_thresh=REC_SCORE_THRESH,
        )
        _label_load_seconds = time.perf_counter() - started
        print(f"[ocr-cpu-lab] General OCR ready in {_label_load_seconds:.2f}s", flush=True)
    return _label_pipeline


def _extract_markdown(result) -> str:
    md = getattr(result, "markdown", None)
    if not isinstance(md, dict):
        return ""
    text = md.get("markdown_texts", "")
    return text if isinstance(text, str) else ""


def _result_payload(result) -> dict:
    for attr in ("json", "res"):
        value = getattr(result, attr, None)
        if isinstance(value, dict):
            if "res" in value and isinstance(value["res"], dict):
                return value["res"]
            return value
    try:
        value = result.to_dict()
        if isinstance(value, dict):
            return value.get("res", value)
    except Exception:
        pass
    return {}


def _normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _regression_markdown(texts: list[str], case_id: str) -> str:
    case = REGRESSIONS.get(case_id)
    if not case:
        return ""
    expected = case.get("expected_fields", [])
    haystack = _normalize(" ".join(texts))
    rows = []
    correct = 0
    for field in expected:
        found = _normalize(field) in haystack
        correct += int(found)
        rows.append(f"| `{field}` | {'PASS' if found else 'MISS'} |")
    recall = (correct / len(expected) * 100) if expected else 0.0
    required = float(case.get("acceptance", {}).get("field_recall", 1.0)) * 100
    status = "PASS" if recall >= required else "FAIL"
    return (
        f"### {case_id} field recall\n\n"
        f"- Correct: **{correct}/{len(expected)}**\n"
        f"- Field recall: **{recall:.1f}%**\n"
        f"- Required: **{required:.1f}%**\n"
        f"- Result: **{status}**\n\n"
        "| Expected field | Status |\n"
        "|---|---|\n" + "\n".join(rows)
    )


def _resolve_mode(file_path: str, requested: str) -> str:
    if requested != "Auto":
        return requested
    return "Document" if Path(file_path).suffix.lower() == ".pdf" else "Label"


def _run_label(file_path: str, regression_case: str):
    with Image.open(file_path) as source:
        image = source.convert("RGB")
    if UPSCALE > 1.0:
        image = image.resize(
            (round(image.width * UPSCALE), round(image.height * UPSCALE)),
            Image.Resampling.LANCZOS,
        )
    pipeline = get_label_pipeline()
    started = time.perf_counter()
    results = list(pipeline.predict(input=np.asarray(image)))
    elapsed = time.perf_counter() - started
    if not results:
        raise RuntimeError("General OCR returned no result objects")

    texts: list[str] = []
    scores: list[float] = []
    for result in results:
        payload = _result_payload(result)
        texts.extend(str(text) for text in (payload.get("rec_texts", []) or []))
        scores.extend(float(score) for score in (payload.get("rec_scores", []) or []))
    if not texts:
        raise RuntimeError("General OCR returned no recognized text")

    lines = []
    for idx, text in enumerate(texts):
        score = scores[idx] if idx < len(scores) else None
        lines.append(text if score is None else f"{text}  _(conf: {score:.3f})_")
    avg_score = sum(scores) / len(scores) if scores else 0.0
    metrics = (
        "### Paddle General OCR — Label Mode\n\n"
        f"- OCR time: **{elapsed:.2f} s**\n"
        f"- Model load: {_label_load_seconds:.2f} s\n" if _label_load_seconds is not None else "- Model load: n/a\n"
    )
    metrics += (
        f"- Detected text lines: **{len(texts)}**\n"
        f"- Average confidence: **{avg_score:.3f}**\n"
        f"- Upscale: {UPSCALE:.1f}x\n"
        f"- Detection side limit: {DET_SIDE_LEN}\n"
    )
    regression_md = _regression_markdown(texts, regression_case)
    if regression_md:
        metrics += "\n" + regression_md
    return image, "\n\n".join(lines), metrics


def _run_document(file_path: str, progress):
    pipeline = get_doc_pipeline()
    count = document_page_count(file_path)
    rows: list[dict] = []
    outputs: list[str] = []
    preview = None
    for page_index in range(count):
        page_no = page_index + 1
        progress(page_index / count, desc=f"Document page {page_no}/{count}")
        image = load_document_page(file_path, page_index)
        preview = image
        started = time.perf_counter()
        results = list(pipeline.predict(input=np.asarray(image.convert("RGB"))))
        elapsed = time.perf_counter() - started
        if not results:
            raise RuntimeError(f"PP-StructureV3 returned no result for page {page_no}")
        parts = [_extract_markdown(result).strip() for result in results]
        page_md = "\n\n".join(part for part in parts if part).strip()
        if not page_md:
            raise RuntimeError(f"PP-StructureV3 returned no Markdown for page {page_no}")
        outputs.append(f"# Page {page_no}\n\n{page_md}")
        rows.append({"page": page_no, "seconds": elapsed, "chars": len(page_md)})
        yield preview, "\n\n---\n\n".join(outputs), _document_metrics(rows)
    progress(1.0, desc="Document complete")


def _document_metrics(rows: list[dict]) -> str:
    total = sum(row["seconds"] for row in rows)
    lines = [
        "### PP-StructureV3 — Document Mode",
        "",
        f"- Pages completed: {len(rows)}",
        f"- Total OCR time: {total:.2f} s",
        f"- Average/page: {total / len(rows):.2f} s" if rows else "- Average/page: n/a",
        f"- Model load: {_doc_load_seconds:.2f} s" if _doc_load_seconds is not None else "- Model load: n/a",
        "",
        "| Page | Seconds | Chars |",
        "|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['page']} | {row['seconds']:.2f} | {row['chars']} |")
    return "\n".join(lines)


def run_ocr(file_path: str | None, mode: str, regression_case: str, progress=gr.Progress()):
    if not file_path:
        yield None, "Please select a PDF or image.", ""
        return
    try:
        resolved = _resolve_mode(file_path, mode)
        if resolved == "Label":
            if Path(file_path).suffix.lower() == ".pdf":
                raise ValueError("Label Mode currently accepts image files; use Document Mode for PDFs.")
            preview, output, metrics = _run_label(file_path, regression_case)
            metrics = f"**Resolved mode:** Label\n\n{metrics}"
            yield preview, output, metrics
            return
        for preview, output, metrics in _run_document(file_path, progress):
            yield preview, output, f"**Resolved mode:** Document\n\n{metrics}"
    except Exception as exc:
        print(f"[ocr-cpu-lab] Unified Paddle ERROR {type(exc).__name__}: {exc}", flush=True)
        yield None, f"OCR failed: `{type(exc).__name__}: {exc}`", ""


regression_choices = ["None"] + sorted(REGRESSIONS.keys())

with gr.Blocks(title="OCR CPU Lab — Paddle Unified") as demo:
    gr.Markdown(
        "# OCR CPU Lab — Paddle Unified\n"
        "**Document Mode:** PP-StructureV3 for PDFs/reports/tables.  \n"
        "**Label Mode:** General OCR for device labels, serial/model numbers and dense small text.  \n"
        "**Auto:** PDF → Document, image → Label (initial deterministic router)."
    )
    source = gr.File(
        label="PDF / image",
        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
        type="filepath",
    )
    with gr.Row():
        mode = gr.Radio(["Auto", "Document", "Label"], value="Auto", label="OCR mode")
        regression_case = gr.Dropdown(regression_choices, value="None", label="Regression case")
    run_button = gr.Button("Run OCR", variant="primary")
    with gr.Row():
        preview = gr.Image(label="Current input/page", type="pil")
        output = gr.Markdown(label="OCR output")
    metrics = gr.Markdown(label="Runtime / quality metrics")
    run_button.click(run_ocr, [source, mode, regression_case], [preview, output, metrics])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT)
