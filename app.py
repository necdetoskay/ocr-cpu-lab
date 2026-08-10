from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gradio as gr

from src.ocr import OvisOCR2CPU
from src.pdf import load_document_image

_runner: OvisOCR2CPU | None = None


def get_runner() -> OvisOCR2CPU:
    global _runner
    if _runner is None:
        print("[ocr-cpu-lab] Loading OvisOCR2 on CPU...", flush=True)
        _runner = OvisOCR2CPU()
        print(
            f"[ocr-cpu-lab] Model ready in {_runner.model_load_seconds:.2f}s. Starting inference when requested.",
            flush=True,
        )
    return _runner


def run_ocr(file_path: str | None, progress=gr.Progress()):
    if not file_path:
        return None, "Please select a PDF or image.", ""

    try:
        progress(0.05, desc="Rendering document...")
        print(f"[ocr-cpu-lab] Input: {file_path}", flush=True)
        image = load_document_image(file_path)
        print(f"[ocr-cpu-lab] Rendered image: {image.width}x{image.height}", flush=True)

        progress(0.15, desc="Loading model on CPU...")
        runner = get_runner()

        progress(0.25, desc="CPU inference started — this can be slow...")
        print("[ocr-cpu-lab] CPU inference started...", flush=True)
        result = runner.run(image)
        print(
            f"[ocr-cpu-lab] CPU inference finished in {result.metrics.inference_seconds:.2f}s.",
            flush=True,
        )
        progress(1.0, desc="Done")
        return image, result.markdown, result.metrics.to_markdown()
    except Exception as exc:
        print(f"[ocr-cpu-lab] ERROR {type(exc).__name__}: {exc}", flush=True)
        return None, f"OCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab") as demo:
    gr.Markdown(
        "# OCR CPU Lab — OvisOCR2\n"
        "Minimal **CPU-only** document parsing smoke test. PDFs use the first page in V0.1."
    )

    with gr.Row():
        source = gr.File(
            label="PDF / image",
            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
            type="filepath",
        )
        run_button = gr.Button("Run OCR", variant="primary")

    gr.Markdown("**Execution device: CPU only — CUDA hidden from process**")

    with gr.Row():
        preview = gr.Image(label="Rendered input", type="pil")
        output = gr.Markdown(label="OCR Markdown")

    metrics = gr.Markdown(label="Runtime metrics")

    run_button.click(
        fn=run_ocr,
        inputs=[source],
        outputs=[preview, output, metrics],
    )


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
