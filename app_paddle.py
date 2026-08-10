from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3

from src.pdf import document_page_count, load_document_page

PORT = int(os.getenv("PADDLE_GRADIO_PORT", "7862"))
CPU_THREADS = int(os.getenv("PADDLE_CPU_THREADS", "8"))
OCR_LANG = os.getenv("PADDLE_OCR_LANG", "tr")
TEXT_REC_SCORE_THRESH = float(os.getenv("PADDLE_DOC_REC_SCORE_THRESH", "0.30"))
TEXT_DET_SIDE_LEN = int(os.getenv("PADDLE_DOC_DET_SIDE_LEN", "1920"))

_structure_pipeline: PPStructureV3 | None = None
_text_pipeline: PaddleOCR | None = None
_structure_load_seconds: float | None = None
_text_load_seconds: float | None = None


def get_structure_pipeline() -> PPStructureV3:
    global _structure_pipeline, _structure_load_seconds
    if _structure_pipeline is None:
        print(f"[ocr-cpu-lab] Loading PP-StructureV3 lang={OCR_LANG} on CPU...", flush=True)
        started = time.perf_counter()
        _structure_pipeline = PPStructureV3(
            device="cpu",
            cpu_threads=CPU_THREADS,
            lang=OCR_LANG,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _structure_load_seconds = time.perf_counter() - started
        print(f"[ocr-cpu-lab] PP-StructureV3 ready in {_structure_load_seconds:.2f}s", flush=True)
    return _structure_pipeline


def get_text_pipeline() -> PaddleOCR:
    global _text_pipeline, _text_load_seconds
    if _text_pipeline is None:
        print(f"[ocr-cpu-lab] Loading General OCR document text pipeline lang={OCR_LANG} on CPU...", flush=True)
        started = time.perf_counter()
        _text_pipeline = PaddleOCR(
            lang=OCR_LANG,
            ocr_version="PP-OCRv5",
            device="cpu",
            cpu_threads=CPU_THREADS,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=TEXT_DET_SIDE_LEN,
            text_det_limit_type="max",
            text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        )
        _text_load_seconds = time.perf_counter() - started
        print(f"[ocr-cpu-lab] General OCR document pipeline ready in {_text_load_seconds:.2f}s", flush=True)
    return _text_pipeline


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


def _run_text_page(image) -> tuple[str, int, float]:
    pipeline = get_text_pipeline()
    started = time.perf_counter()
    results = list(pipeline.predict(input=np.asarray(image.convert("RGB"))))
    elapsed = time.perf_counter() - started
    texts: list[str] = []
    scores: list[float] = []
    for result in results:
        payload = _result_payload(result)
        texts.extend(str(text) for text in (payload.get("rec_texts", []) or []))
        scores.extend(float(score) for score in (payload.get("rec_scores", []) or []))
    if not texts:
        raise RuntimeError("General OCR returned no recognized text")
    return "\n\n".join(texts), len(texts), elapsed


def _run_structured_page(image) -> tuple[str, int, float]:
    pipeline = get_structure_pipeline()
    started = time.perf_counter()
    results = list(pipeline.predict(input=np.asarray(image.convert("RGB"))))
    elapsed = time.perf_counter() - started
    if not results:
        raise RuntimeError("PP-StructureV3 returned no result objects")
    parts = [_extract_markdown(result).strip() for result in results]
    parts = [part for part in parts if part]
    page_markdown = "\n\n".join(parts).strip()
    if not page_markdown:
        raise RuntimeError("PP-StructureV3 returned result objects but no Markdown")
    return page_markdown, len(results), elapsed


def _metrics(rows: list[dict], mode: str) -> str:
    total = sum(row["seconds"] for row in rows)
    load_seconds = _text_load_seconds if mode == "Text OCR" else _structure_load_seconds
    backend = "General OCR / PP-OCRv5" if mode == "Text OCR" else "PP-StructureV3"
    lines = [
        "### PaddleOCR CPU document metrics",
        "",
        f"- Mode: **{mode}**",
        f"- Backend: {backend} / PaddlePaddle CPU",
        f"- OCR language: {OCR_LANG}",
        f"- CPU threads: {CPU_THREADS}",
        f"- Pages completed: {len(rows)}",
        f"- Total OCR time: {total:.2f} s",
        f"- Average/page: {total / len(rows):.2f} s" if rows else "- Average/page: n/a",
        f"- Model load: {load_seconds:.2f} s" if load_seconds is not None else "- Model load: n/a",
        "",
        "| Page | Seconds | Items | Chars |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['page']} | {row['seconds']:.2f} | {row['items']} | {row['chars']} |")
    return "\n".join(lines)


def run_ocr(file_path: str | None, mode: str, progress=gr.Progress()):
    if not file_path:
        yield None, "Please select a PDF or image.", ""
        return

    try:
        count = document_page_count(file_path)
        rows: list[dict] = []
        outputs: list[str] = []
        preview = None

        for page_index in range(count):
            page_no = page_index + 1
            progress(page_index / count, desc=f"Processing page {page_no}/{count}")
            image = load_document_page(file_path, page_index)
            preview = image
            yield preview, "\n\n---\n\n".join(outputs), _metrics(rows, mode) if rows else f"Processing page {page_no}/{count}..."

            if mode == "Text OCR":
                page_text, items, elapsed = _run_text_page(image)
            else:
                page_text, items, elapsed = _run_structured_page(image)

            outputs.append(f"# Page {page_no}\n\n{page_text}")
            rows.append({"page": page_no, "seconds": elapsed, "items": items, "chars": len(page_text)})
            print(
                f"[ocr-cpu-lab] paddle document mode={mode} page={page_no} seconds={elapsed:.2f} items={items} chars={len(page_text)}",
                flush=True,
            )
            progress((page_index + 1) / count, desc=f"Completed page {page_no}/{count}")
            yield preview, "\n\n---\n\n".join(outputs), _metrics(rows, mode)

        progress(1.0, desc="Document complete")
    except Exception as exc:
        print(f"[ocr-cpu-lab] Paddle document ERROR {type(exc).__name__}: {exc}", flush=True)
        yield None, f"PaddleOCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab — PaddleOCR Documents") as demo:
    gr.Markdown(
        "# OCR CPU Lab — PaddleOCR Documents\n"
        "**Text OCR (recommended):** raw PP-OCRv5 detection + recognition for normal text PDFs; prioritizes completeness.  \n"
        "**Structured:** PP-StructureV3 for tables, complex multi-column layouts and structure-aware Markdown."
    )
    source = gr.File(
        label="PDF / image",
        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
        type="filepath",
    )
    mode = gr.Radio(["Text OCR", "Structured"], value="Text OCR", label="Document mode")
    run_button = gr.Button("Run PaddleOCR", variant="primary")
    with gr.Row():
        preview = gr.Image(label="Current rendered page", type="pil")
        output = gr.Markdown(label="OCR output")
    metrics = gr.Markdown(label="Runtime metrics")
    run_button.click(run_ocr, [source, mode], [preview, output, metrics])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT)
