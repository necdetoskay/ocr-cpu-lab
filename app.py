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


def _heartbeat_markdown(elapsed: float, ram_mb: float, process_cpu: float, system_cpu: float) -> str:
    return (
        "### Live status\n"
        "**State:** CPU inference running  \n"
        f"**Elapsed:** {elapsed:.1f} s  \n"
        f"**Process RAM:** {ram_mb:,.0f} MB  \n"
        f"**Process CPU:** {process_cpu:.1f}%  \n"
        f"**System CPU:** {system_cpu:.1f}%  \n"
        "\nValues update every ~2 seconds. This is a liveness/resource heartbeat, "
        "not token-level generation progress."
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

        # Re-prime immediately before inference so the first heartbeat has a clean baseline.
        _process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        future = _executor.submit(runner.run, image)

        while not future.done():
            time.sleep(2.0)
            elapsed = time.perf_counter() - inference_started
            ram_mb = _process.memory_info().rss / (1024 * 1024)
            process_cpu = _process.cpu_percent(interval=None)
            system_cpu = psutil.cpu_percent(interval=None)
            status = _heartbeat_markdown(elapsed, ram_mb, process_cpu, system_cpu)
            print(
                "[ocr-cpu-lab] heartbeat "
                f"elapsed={elapsed:.1f}s "
                f"ram={ram_mb:.0f}MB "
                f"process_cpu={process_cpu:.1f}% "
                f"system_cpu={system_cpu:.1f}%",
                flush=True,
            )
            yield image, "", "", status

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
