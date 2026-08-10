from __future__ import annotations

import base64
import io
import os
import time

import gradio as gr
import requests
from PIL import Image

from src.pdf import document_page_count, load_document_page

SERVER = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
DEFAULT_MAX_TOKENS = int(os.getenv("OCR_DEFAULT_MAX_TOKENS", "2048"))
RETRY_MAX_TOKENS = int(os.getenv("OCR_RETRY_MAX_TOKENS", "4096"))
GRADIO_HOST = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
PROMPT = (
    "Extract all readable content from the image in natural human reading order and output the result as a single Markdown document. "
    "For charts or images, represent them using an HTML image tag with bounding-box coordinates. "
    "Format formulas as LaTeX and tables as HTML. Transcribe all other text as standard Markdown."
)


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _model_id() -> str:
    response = requests.get(f"{SERVER}/v1/models", timeout=10)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise RuntimeError("llama.cpp server returned no models")
    return data[0]["id"]


def _ocr_page(image: Image.Image, model: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{SERVER}/v1/chat/completions",
        json=payload,
        timeout=3600,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    body = response.json()
    choice = body["choices"][0]
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    finish_reason = choice.get("finish_reason", "unknown")
    hit_limit = finish_reason == "length" or completion_tokens == max_tokens
    return {
        "text": choice["message"]["content"],
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "max_tokens": max_tokens,
        "hit_limit": hit_limit,
    }


def _metrics_markdown(rows: list[dict], model: str) -> str:
    total = sum(row["elapsed"] for row in rows)
    retries = sum(1 for row in rows if row["retried"])
    lines = [
        "### GGUF CPU document metrics",
        "",
        "- Backend: llama.cpp / Q4_K_M",
        f"- Server: {SERVER}",
        f"- Model: `{model}`",
        f"- Pages completed: {len(rows)}",
        f"- Total OCR time: {total:.2f} s",
        f"- Average/page: {total / len(rows):.2f} s" if rows else "- Average/page: n/a",
        f"- Adaptive retries ({DEFAULT_MAX_TOKENS}→{RETRY_MAX_TOKENS}): {retries}",
        "",
        "| Page | Seconds | Prompt tok | Completion tok | Max tok | Finish | Retry | Chars |",
        "|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['page']} | {row['elapsed']:.2f} | {row['prompt_tokens'] or 'n/a'} | "
            f"{row['completion_tokens'] or 'n/a'} | {row['max_tokens']} | {row['finish_reason']} | "
            f"{'YES' if row['retried'] else 'NO'} | {row['chars']} |"
        )
    return "\n".join(lines)


def run_ocr(file_path: str | None, progress=gr.Progress()):
    if not file_path:
        yield None, "Please select a PDF or image.", ""
        return

    try:
        page_count = document_page_count(file_path)
        model = _model_id()
        outputs: list[str] = []
        rows: list[dict] = []
        preview = None

        for page_index in range(page_count):
            page_no = page_index + 1
            progress(page_index / page_count, desc=f"Rendering page {page_no}/{page_count}")
            image = load_document_page(file_path, page_index)
            preview = image
            yield preview, "\n\n".join(outputs), _metrics_markdown(rows, model) if rows else f"Processing page {page_no}/{page_count}..."

            progress((page_index + 0.15) / page_count, desc=f"OCR page {page_no}/{page_count} — {DEFAULT_MAX_TOKENS} token ceiling")
            first = _ocr_page(image, model, DEFAULT_MAX_TOKENS)
            result = first
            retried = False

            if first["hit_limit"]:
                retried = True
                progress((page_index + 0.55) / page_count, desc=f"Page {page_no} hit token ceiling — retrying with {RETRY_MAX_TOKENS}")
                result = _ocr_page(image, model, RETRY_MAX_TOKENS)

            outputs.append(f"# Page {page_no}\n\n{result['text'].strip()}")
            rows.append(
                {
                    "page": page_no,
                    "elapsed": first["elapsed"] + (result["elapsed"] if retried else 0.0),
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "max_tokens": result["max_tokens"],
                    "finish_reason": result["finish_reason"],
                    "retried": retried,
                    "chars": len(result["text"]),
                }
            )
            progress((page_index + 1) / page_count, desc=f"Completed page {page_no}/{page_count}")
            yield preview, "\n\n---\n\n".join(outputs), _metrics_markdown(rows, model)

        progress(1.0, desc="Document complete")
    except Exception as exc:
        yield None, f"GGUF OCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab — OvisOCR2 GGUF") as demo:
    gr.Markdown(
        "# OCR CPU Lab — OvisOCR2 GGUF / llama.cpp\n"
        "CPU-only Q4_K_M document test. PDFs are processed **page by page**. "
        f"Each page starts at {DEFAULT_MAX_TOKENS} output tokens; only pages that hit the limit are retried at {RETRY_MAX_TOKENS}."
    )
    source = gr.File(
        label="PDF / image",
        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
        type="filepath",
    )
    run_button = gr.Button("Run GGUF OCR", variant="primary")
    with gr.Row():
        preview = gr.Image(label="Current rendered page", type="pil")
        output = gr.Markdown(label="OCR Markdown")
    metrics = gr.Markdown(label="Runtime metrics")
    run_button.click(run_ocr, [source], [preview, output, metrics])


if __name__ == "__main__":
    demo.queue().launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, inbrowser=False)
