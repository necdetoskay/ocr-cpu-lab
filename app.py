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
        _runner = OvisOCR2CPU()
    return _runner


def run_ocr(file_path: str | None):
    if not file_path:
        return None, "Please select a PDF or image.", ""

    try:
        image = load_document_image(file_path)
        result = get_runner().run(image)
        return image, result.markdown, result.metrics.to_markdown()
    except Exception as exc:
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
    demo.launch(inbrowser=True)
