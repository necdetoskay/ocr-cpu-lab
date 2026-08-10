from __future__ import annotations

import os
import re
import time

import gradio as gr
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

PORT = int(os.getenv("PADDLE_LABEL_GRADIO_PORT", "7863"))
CPU_THREADS = int(os.getenv("PADDLE_CPU_THREADS", "8"))
UPSCALE = float(os.getenv("PADDLE_LABEL_UPSCALE", "2.0"))
DET_SIDE_LEN = int(os.getenv("PADDLE_LABEL_DET_SIDE_LEN", "1920"))
REC_SCORE_THRESH = float(os.getenv("PADDLE_LABEL_REC_SCORE_THRESH", "0.35"))

DEVICE_LABEL_001 = [
    "27E2N25",
    "8721038004472",
    "27E2N2500/01",
    "UK02601033689",
    "HF4BRT2BFGPHDNE",
]

_pipeline: PaddleOCR | None = None
_model_load_seconds: float | None = None


def get_pipeline() -> PaddleOCR:
    global _pipeline, _model_load_seconds
    if _pipeline is None:
        print("[ocr-cpu-lab] Loading Paddle General OCR label pipeline on CPU...", flush=True)
        started = time.perf_counter()
        _pipeline = PaddleOCR(
            lang="tr",
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
        _model_load_seconds = time.perf_counter() - started
        print(f"[ocr-cpu-lab] General OCR label pipeline ready in {_model_load_seconds:.2f}s", flush=True)
    return _pipeline


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


def _regression_markdown(texts: list[str], enabled: bool) -> str:
    if not enabled:
        return ""
    haystack = _normalize(" ".join(texts))
    rows = []
    correct = 0
    for expected in DEVICE_LABEL_001:
        found = _normalize(expected) in haystack
        correct += int(found)
        rows.append(f"| `{expected}` | {'PASS' if found else 'MISS'} |")
    recall = correct / len(DEVICE_LABEL_001) * 100
    status = "PASS" if correct == len(DEVICE_LABEL_001) else "FAIL"
    return (
        "### DEVICE-LABEL-001 field recall\n\n"
        f"- Correct: **{correct}/{len(DEVICE_LABEL_001)}**\n"
        f"- Field recall: **{recall:.1f}%**\n"
        f"- Result: **{status}**\n\n"
        "| Expected field | Status |\n"
        "|---|---|\n" + "\n".join(rows)
    )


def run_ocr(file_path: str | None, regression: bool):
    if not file_path:
        return None, "Please select an image.", ""

    try:
        with Image.open(file_path) as source:
            image = source.convert("RGB")
        if UPSCALE > 1.0:
            image = image.resize(
                (round(image.width * UPSCALE), round(image.height * UPSCALE)),
                Image.Resampling.LANCZOS,
            )

        image_np = np.asarray(image)
        pipeline = get_pipeline()
        started = time.perf_counter()
        results = list(pipeline.predict(input=image_np))
        elapsed = time.perf_counter() - started
        if not results:
            raise RuntimeError("General OCR returned no result objects")

        texts: list[str] = []
        scores: list[float] = []
        for result in results:
            payload = _result_payload(result)
            rec_texts = payload.get("rec_texts", []) or []
            rec_scores = payload.get("rec_scores", []) or []
            texts.extend(str(text) for text in rec_texts)
            scores.extend(float(score) for score in rec_scores)

        if not texts:
            raise RuntimeError("General OCR returned result objects but no recognized text")

        lines = []
        for idx, text in enumerate(texts):
            score = scores[idx] if idx < len(scores) else None
            if score is None:
                lines.append(text)
            else:
                lines.append(f"{text}  _(conf: {score:.3f})_")
        output = "\n\n".join(lines)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        regression_md = _regression_markdown(texts, regression)
        metrics = (
            "### Paddle General OCR — Label Mode metrics\n\n"
            "- Backend: Paddle General OCR / PP-OCRv5 CPU\n"
            "- OCR language: tr\n"
            f"- CPU threads: {CPU_THREADS}\n"
            f"- Upscale: {UPSCALE:.1f}x\n"
            f"- Detection side limit: {DET_SIDE_LEN}\n"
            f"- Recognition threshold: {REC_SCORE_THRESH:.2f}\n"
            f"- Model load: {_model_load_seconds:.2f} s\n" if _model_load_seconds is not None else "- Model load: n/a\n"
        )
        metrics += (
            f"- OCR time: **{elapsed:.2f} s**\n"
            f"- Detected text lines: **{len(texts)}**\n"
            f"- Average confidence: **{avg_score:.3f}**\n"
        )
        if regression_md:
            metrics += "\n" + regression_md

        return image, output, metrics
    except Exception as exc:
        print(f"[ocr-cpu-lab] Paddle label ERROR {type(exc).__name__}: {exc}", flush=True)
        return None, f"Label OCR failed: `{type(exc).__name__}: {exc}`", ""


with gr.Blocks(title="OCR CPU Lab — Paddle Label Mode") as demo:
    gr.Markdown(
        "# OCR CPU Lab — Paddle General OCR / Label Mode\n"
        "For device labels, serial/model numbers, barcodes-with-text and dense small-print images. "
        "This bypasses PP-StructureV3 layout parsing and returns raw detected text lines."
    )
    source = gr.File(
        label="Label / photo",
        file_types=[".png", ".jpg", ".jpeg", ".webp"],
        type="filepath",
    )
    regression = gr.Checkbox(
        value=True,
        label="Run DEVICE-LABEL-001 field recall (Philips test label)",
    )
    run_button = gr.Button("Run Label OCR", variant="primary")
    with gr.Row():
        preview = gr.Image(label="Upscaled input", type="pil")
        output = gr.Markdown(label="Raw OCR lines")
    metrics = gr.Markdown(label="Runtime / regression metrics")
    run_button.click(run_ocr, [source, regression], [preview, output, metrics])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT)
