from __future__ import annotations

import base64
import io
import time

import gradio as gr
import requests
from PIL import Image

from src.pdf import load_document_image

SERVER = "http://127.0.0.1:8080"
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


def run_ocr(file_path: str | None):
    if not file_path:
        return None, "Please select a PDF or image.", ""

    try:
        image = load_document_image(file_path)
        started = time.perf_counter()
        model = _model_id()

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
            "max_tokens": 512,
            "stream": False,
        }

        response = requests.post(
            f"{SERVER}/v1/chat/completions",
            json=payload,
            timeout=900,
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        elapsed = time.perf_counter() - started
        usage = body.get("usage", {})

        metrics = (
            "### GGUF CPU metrics\n"
            f"- Backend: llama.cpp / Q4_K_M\n"
            f"- Server: {SERVER}\n"
            f"- Model: `{model}`\n"
            f"- Total request: {elapsed:.2f} s\n"
            f"- Input: {image.width}x{image.height}\n"
            f"- Prompt tokens: {usage.get('prompt_tokens', 'n/a')}\n"
            f"- Completion tokens: {usage.get('completion_tokens', 'n/a')}\n"
            f"- Output chars: {len(text)}"
        )
        return image, text, metrics
    except Exception as exc:
        return None, f"GGUF OCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab — OvisOCR2 GGUF") as demo:
    gr.Markdown(
        "# OCR CPU Lab — OvisOCR2 GGUF / llama.cpp\n"
        "CPU-only comparison path using Q4_K_M. Start `scripts/run-ovis-gguf-cpu.ps1` first."
    )
    source = gr.File(
        label="PDF / image",
        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
        type="filepath",
    )
    run_button = gr.Button("Run GGUF OCR", variant="primary")
    with gr.Row():
        preview = gr.Image(label="Rendered input", type="pil")
        output = gr.Markdown(label="OCR Markdown")
    metrics = gr.Markdown(label="Runtime metrics")
    run_button.click(run_ocr, [source], [preview, output, metrics])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
