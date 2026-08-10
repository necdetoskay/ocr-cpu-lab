from __future__ import annotations

import os
from dataclasses import dataclass

# Must be set before importing torch/transformers so CUDA is hidden from this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .metrics import RunMetrics, cpu_name, process_ram_mb, timer

MODEL_ID = "ATH-MaaS/OvisOCR2"
PROMPT = (
    "Parse this document page into Markdown. Preserve natural reading order, "
    "headings, paragraphs, tables, formulas, and visible document structure. "
    "Do not summarize or explain the document; transcribe/parse it faithfully."
)


@dataclass
class OCRResult:
    markdown: str
    metrics: RunMetrics


class OvisOCR2CPU:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.device = torch.device("cpu")
        started = timer()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_load_seconds = timer() - started

    @torch.inference_mode()
    def run(self, image: Image.Image, max_new_tokens: int = 512) -> OCRResult:
        total_started = timer()
        ram_before = process_ram_mb()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        inference_started = timer()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        inference_seconds = timer() - inference_started

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_tokens:]
        markdown = self.processor.decode(generated, skip_special_tokens=True).strip()

        ram_after = process_ram_mb()
        total_seconds = timer() - total_started
        width, height = image.size

        metrics = RunMetrics(
            device="CPU",
            model=self.model_id,
            model_load_seconds=self.model_load_seconds,
            inference_seconds=inference_seconds,
            total_seconds=total_seconds,
            process_ram_before_mb=ram_before,
            process_ram_after_mb=ram_after,
            process_ram_delta_mb=ram_after - ram_before,
            input_width=width,
            input_height=height,
            output_characters=len(markdown),
            cpu=cpu_name(),
            logical_cores=os.cpu_count() or 0,
        )
        return OCRResult(markdown=markdown, metrics=metrics)
