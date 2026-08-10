from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gradio as gr
import psutil

from src.ocr import OvisOCR2CPU
from src.pdf import load_document_image

_runner: OvisOCR2CPU | None = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovisocr2-cpu")
_process = psutil.Process(os.getpid())
_process.cpu_percent(interval=None)  # Prime psutil's process CPU measurement.


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


def _heartbeat_markdown(elapsed: float) -> str:
    ram_mb = _process.memory_info().rss / (1024 * 1024)
    cpu_percent = _process.cpu_percent(interval=None)
    return (
        "### Live status\n"
        "**State:** CPU inference running  \n"
        f"**Elapsed:** {elapsed:.1f} s  \n"
        f"**Process RAM:** {ram_mb:,.0f} MB  \n"
        f"**Process CPU:** {cpu_percent:.1f}%  \n"
        "\nThe elapsed time and resource values update every 2 seconds. "
        "This is a heartbeat, not token-level generation progress."
    )


def run_ocr(file_path: str | None, progress=gr.Progress()):
    if not file_path:
        yield None, "Please select a PDF or image.", "", "**State:** idle"
        return

    try:
        progress(0.05, desc="Rendering document...")
        print(f"[ocr-cpu-lab] Input: {file_path}", flush=True)
        image = load_document_image(file_path)
        print(f"[ocr-cpu-lab] Rendered image: {image.width}x{image.height}", flush=True)
        yield image, "", "", "**State:** document rendered"

        progress(0.15, desc="Loading model on CPU...")
        runner = get_runner()
        yield image, "", "", f"**State:** model ready ({runner.model_load_seconds:.2f}s load time)"

        progress(0.25, desc="CPU inference running — watch Live status below")
        print("[ocr-cpu-lab] CPU inference started...", flush=True)
        inference_started = time.perf_counter()
        future = _executor.submit(runner.run, image)

        while not future.done():
            elapsed = time.perf_counter() - inference_started
            status = _heartbeat_markdown(elapsed)
            print(
                "[ocr-cpu-lab] heartbeat "
                f"elapsed={elapsed:.1f}s "
                f"ram={_process.memory_info().rss / (1024 * 1024):.0f}MB "
                f"cpu={_process.cpu_percent(interval=None):.1f}%",
                flush=True,
            )
            yield image, "", "", status
            time.sleep(2.0)

        result = future.result()
        print(
            f"[ocr-cpu-lab] CPU inference finished in {result.metrics.inference_seconds:.2f}s.",
            flush=True,
        )
        progress(1.0, desc="Done")
        yield (
            image,
            result.markdown,
            result.metrics.to_markdown(),
            f"### Live status\n**State:** completed  \n**Inference:** {result.metrics.inference_seconds:.2f} s",
        )
    except Exception as exc:
        print(f"[ocr-cpu-lab] ERROR {type(exc).__name__}: {exc}", flush=True)
        yield None, f"OCR failed: `{type(exc).__name__}: {exc}`", "", "**State:** failed"


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
    live_status = gr.Markdown("**State:** idle")

    with gr.Row():
        preview = gr.Image(label="Rendered input", type="pil")
        output = gr.Markdown(label="OCR Markdown")

    metrics = gr.Markdown(label="Runtime metrics")

    run_button.click(
        fn=run_ocr,
        inputs=[source],
        outputs=[preview, output, metrics, live_status],
    )


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
