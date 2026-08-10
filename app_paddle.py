from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np
from paddleocr import PPStructureV3

from src.pdf import document_page_count, load_document_page

PORT = int(os.getenv("PADDLE_GRADIO_PORT", "7862"))
CPU_THREADS = int(os.getenv("PADDLE_CPU_THREADS", "8"))
OCR_LANG = os.getenv("PADDLE_OCR_LANG", "tr")

_pipeline: PPStructureV3 | None = None
_model_load_seconds: float | None = None


def get_pipeline() -> PPStructureV3:
    global _pipeline, _model_load_seconds
    if _pipeline is None:
        print(f"[ocr-cpu-lab] Loading PP-StructureV3 on CPU with lang={OCR_LANG}...", flush=True)
        started = time.perf_counter()
        _pipeline = PPStructureV3(
            device="cpu",
            cpu_threads=CPU_THREADS,
            lang=OCR_LANG,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _model_load_seconds = time.perf_counter() - started
        print(f"[ocr-cpu-lab] PP-StructureV3 ready in {_model_load_seconds:.2f}s", flush=True)
    return _pipeline


def _extract_markdown(result) -> str:
    md = getattr(result, "markdown", None)
    if not isinstance(md, dict):
        return ""
    text = md.get("markdown_texts", "")
    return text if isinstance(text, str) else ""


def _metrics(rows: list[dict]) -> str:
    total = sum(row["seconds"] for row in rows)
    lines = [
        "### PaddleOCR CPU document metrics",
        "",
        "- Backend: PP-StructureV3 / PaddlePaddle CPU",
        f"- OCR language: {OCR_LANG}",
        "- Recognition family: PP-OCRv5 multilingual (Turkish/Latin)",
        f"- CPU threads: {CPU_THREADS}",
        f"- Pages completed: {len(rows)}",
        f"- Total OCR time: {total:.2f} s",
        f"- Average/page: {total / len(rows):.2f} s" if rows else "- Average/page: n/a",
        f"- Model load: {_model_load_seconds:.2f} s" if _model_load_seconds is not None else "- Model load: n/a",
        "",
        "| Page | Seconds | Results | Chars |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['page']} | {row['seconds']:.2f} | {row['result_count']} | {row['chars']} |"
        )
    return "\n".join(lines)


def run_ocr(file_path: str | None, progress=gr.Progress()):
    if not file_path:
        yield None, "Please select a PDF or image.", ""
        return

    try:
        pipeline = get_pipeline()
        count = document_page_count(file_path)
        rows: list[dict] = []
        outputs: list[str] = []
        preview = None

        for page_index in range(count):
            page_no = page_index + 1
            progress(page_index / count, desc=f"Rendering page {page_no}/{count}")
            image = load_document_page(file_path, page_index)
            preview = image
            yield preview, "\n\n---\n\n".join(outputs), _metrics(rows) if rows else f"Processing page {page_no}/{count}..."

            image_np = np.asarray(image.convert("RGB"))

            started = time.perf_counter()
            results = list(pipeline.predict(input=image_np))
            elapsed = time.perf_counter() - started

            if not results:
                raise RuntimeError(
                    f"PP-StructureV3 returned no result objects for page {page_no}; benchmark aborted instead of recording a false zero-second success."
                )

            parts = [_extract_markdown(result).strip() for result in results]
            parts = [part for part in parts if part]
            page_markdown = "\n\n".join(parts).strip()

            if not page_markdown:
                result_types = ", ".join(type(result).__name__ for result in results)
                raise RuntimeError(
                    f"PP-StructureV3 returned {len(results)} result object(s) but no Markdown for page {page_no}. Result types: {result_types}"
                )

            outputs.append(f"# Page {page_no}\n\n{page_markdown}")
            rows.append(
                {
                    "page": page_no,
                    "seconds": elapsed,
                    "result_count": len(results),
                    "chars": len(page_markdown),
                }
            )
            print(
                f"[ocr-cpu-lab] paddle lang={OCR_LANG} page={page_no} seconds={elapsed:.2f} results={len(results)} chars={len(page_markdown)}",
                flush=True,
            )
            progress((page_index + 1) / count, desc=f"Completed page {page_no}/{count}")
            yield preview, "\n\n---\n\n".join(outputs), _metrics(rows)

        progress(1.0, desc="Document complete")
    except Exception as exc:
        print(f"[ocr-cpu-lab] PaddleOCR ERROR {type(exc).__name__}: {exc}", flush=True)
        yield None, f"PaddleOCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab — PaddleOCR") as demo:
    gr.Markdown(
        "# OCR CPU Lab — PP-StructureV3 / PaddleOCR\n"
        f"CPU-only document parsing benchmark using **Turkish OCR (`lang={OCR_LANG}`)**. "
        "Orientation classification, unwarping and text-line orientation are disabled for the baseline."
    )
    source = gr.File(
        label="PDF / image",
        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
        type="filepath",
    )
    run_button = gr.Button("Run PaddleOCR", variant="primary")
    with gr.Row():
        preview = gr.Image(label="Current rendered page", type="pil")
        output = gr.Markdown(label="PaddleOCR Markdown")
    metrics = gr.Markdown(label="Runtime metrics")
    run_button.click(run_ocr, [source], [preview, output, metrics])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT)
