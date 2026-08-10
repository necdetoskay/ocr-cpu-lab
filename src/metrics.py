from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict, dataclass

import psutil


@dataclass
class RunMetrics:
    device: str
    model: str
    model_load_seconds: float
    inference_seconds: float
    total_seconds: float
    process_ram_before_mb: float
    process_ram_after_mb: float
    process_ram_delta_mb: float
    input_width: int
    input_height: int
    output_characters: int
    cpu: str
    logical_cores: int

    def to_markdown(self) -> str:
        data = asdict(self)
        rows = ["| Metric | Value |", "|---|---:|"]
        for key, value in data.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                rendered = f"{value:.2f}"
            else:
                rendered = str(value)
            rows.append(f"| {label} | {rendered} |")
        return "\n".join(rows)


def process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def cpu_name() -> str:
    return platform.processor() or platform.machine() or "unknown"


def timer() -> float:
    return time.perf_counter()
