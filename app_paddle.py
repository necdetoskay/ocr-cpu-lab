from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gradio as gr
from paddleocr import PPStructureV3

from src.pdf import document_page_count, load_document_page

PORT = int(os.getenv("PADDLE_GRADIO_PORT", "7862"))

_pipeline: PPStructureV3 | None = None


def get_pipeline() -> PPStructureV3:
    global _pipeline
    if _pipeline is None:
        print("[ocr-cpu-lab] Loading PP-StructureV3 on CPU...", flush=True)
        started = time.perf_counter()
        _pipeline = PPStructureV3(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print(f"[ocr-cpu-lab] PP-StructureV3 ready in {time.perf_counter() - started:.2f}s", flush=True)
    return _pipeline


def _extract_markdown(result) -> str:
    with tempfile.TemporaryDirectory(prefix="paddle-md-") as tmpdir:
        result.save_to_markdown(save_path=tmpdir)
        markdown_files = sorted(Path(tmpdir).rglob("*.md"))
        if not markdown_files:
            return ""
        return "\n\n".join(path.read_text(encoding="utf-8") for path in markdown_files)


def _metrics(rows: list[dict], model_load_seconds: float | None = None) -> str:
    total = sum(row["seconds"] for row in rows)
    lines = [
        "### PaddleOCR CPU document metrics",
        "",
        "- Backend: PP-StructureV3 / PaddlePaddle CPU",
        f"- Pages completed: {len(rows)}",
        f"- Total OCR time: {total:.2f} s",
        f"- Average/page: {total / len(rows):.2f} s" if rows else "- Average/page: n/a",
    ]
    if model_load_seconds is not None:
        lines.append(f"- Model load: {model_load_seconds:.2f} s")
    lines += [
        "",
        "| Page | Seconds | Chars |",
        "|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['page']} | {row['seconds']:.2f} | {row['chars']} |")
    return "\n".join(lines)


def run_ocr(file_path: str | None, progress=gr.Progress()):
    if not file_path:
        yield None, "Please select a PDF or image.", ""
        return

    try:
        load_started = time.perf_counter()
        pipeline = get_pipeline()
        load_seconds = time.perf_counter() - load_started
        count = document_page_count(file_path)
        rows: list[dict] = []
        outputs: list[str] = []
        preview = None

        for page_index in range(count):
            page_no = page_index + 1
            progress(page_index / count, desc=f"Rendering page {page_no}/{count}")
            image = load_document_page(file_path, page_index)
            preview = image
            yield preview, "\n\n---\n\n".join(outputs), _metrics(rows, load_seconds) if rows else f"Processing page {page_no}/{count}..."

            started = time.perf_counter()
            results = list(pipeline.predict(image))
            elapsed = time.perf_counter() - started
            page_markdown = "\n\n".join(_extract_markdown(result).strip() for result in results).strip()
            outputs.append(f"# Page {page_no}\n\n{page_markdown}")
            rows.append({"page": page_no, "seconds": elapsed, "chars": len(page_markdown)})
            print(
                f"[ocr-cpu-lab] paddle page={page_no} seconds={elapsed:.2f} chars={len(page_markdown)}",
                flush=True,
            )
            progress((page_index + 1) / count, desc=f"Completed page {page_no}/{count}")
            yield preview, "\n\n---\n\n".join(outputs), _metrics(rows, load_seconds)

        progress(1.0, desc="Document complete")
    except Exception as exc:
        print(f"[ocr-cpu-lab] PaddleOCR ERROR {type(exc).__name__}: {exc}", flush=True)
        yield None, f"PaddleOCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab — PaddleOCR") as demo:
    gr.Markdown(
        "# OCR CPU Lab — PP-StructureV3 / PaddleOCR\n"
        "CPU-only document parsing benchmark. Orientation classification, unwarping and text-line orientation are disabled for the baseline."
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
