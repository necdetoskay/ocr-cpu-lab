from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from src.pdf import document_page_count, load_document_page

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "regressions" / "suite.json"
DOCUMENT_CASES_PATH = ROOT / "regressions" / "document_cases.json"
LABEL_CASES_PATH = ROOT / "regressions" / "label_cases.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_payload(result) -> dict:
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


def normalize_document(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def normalize_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def evaluate_document(output: str, case: dict) -> dict:
    markers = case.get("expected_markers", [])
    haystack = normalize_document(output)
    checks = []
    for marker in markers:
        ok = normalize_document(marker) in haystack
        checks.append({"expected": marker, "ok": ok})
    correct = sum(1 for item in checks if item["ok"])
    recall = correct / len(checks) if checks else 0.0
    required = float(case.get("acceptance", {}).get("marker_recall", 1.0))
    return {
        "correct": correct,
        "total": len(checks),
        "recall": recall,
        "required": required,
        "passed": recall >= required,
        "checks": checks,
    }


def evaluate_label(output: str, case: dict) -> dict:
    fields = case.get("expected_fields", [])
    haystack = normalize_label(output)
    checks = []
    for field in fields:
        ok = normalize_label(field) in haystack
        checks.append({"expected": field, "ok": ok})
    correct = sum(1 for item in checks if item["ok"])
    recall = correct / len(checks) if checks else 0.0
    required = float(case.get("acceptance", {}).get("field_recall", 1.0))
    must_all = bool(case.get("acceptance", {}).get("critical_fields_must_all_match", False))
    passed = recall >= required and (not must_all or correct == len(checks))
    return {
        "correct": correct,
        "total": len(checks),
        "recall": recall,
        "required": required,
        "passed": passed,
        "checks": checks,
    }


def make_pipeline(cpu_threads: int) -> PaddleOCR:
    return PaddleOCR(
        lang="tr",
        ocr_version="PP-OCRv5",
        device="cpu",
        cpu_threads=cpu_threads,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1920,
        text_det_limit_type="max",
        text_rec_score_thresh=0.30,
    )


def run_general_ocr(pipeline: PaddleOCR, image: Image.Image) -> tuple[str, float, int]:
    started = time.perf_counter()
    results = list(pipeline.predict(input=np.asarray(image.convert("RGB"))))
    elapsed = time.perf_counter() - started
    texts: list[str] = []
    for result in results:
        payload = result_payload(result)
        texts.extend(str(text) for text in (payload.get("rec_texts", []) or []))
    if not texts:
        raise RuntimeError("General OCR returned no recognized text")
    return "\n".join(texts), elapsed, len(texts)


def run_document_case(pipeline: PaddleOCR, sample: Path) -> tuple[str, float, int]:
    outputs: list[str] = []
    total = 0.0
    items = 0
    for page_index in range(document_page_count(str(sample))):
        image = load_document_page(str(sample), page_index)
        text, elapsed, count = run_general_ocr(pipeline, image)
        outputs.append(f"# Page {page_index + 1}\n{text}")
        total += elapsed
        items += count
    return "\n\n".join(outputs), total, items


def run_label_case(pipeline: PaddleOCR, sample: Path) -> tuple[str, float, int]:
    with Image.open(sample) as source:
        image = source.convert("RGB")
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    return run_general_ocr(pipeline, image)


def render_markdown(report: dict) -> str:
    lines = [
        "# OCR Regression Suite",
        "",
        f"- Passed: **{report['passed']}**",
        f"- Failed: **{report['failed']}**",
        f"- Skipped: **{report['skipped']}**",
        f"- Total runtime: **{report['total_seconds']:.2f} s**",
        "",
        "| Case | Pipeline | Status | Recall | Seconds | Items |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in report["cases"]:
        recall = item.get("evaluation", {}).get("recall")
        recall_text = "n/a" if recall is None else f"{recall * 100:.1f}%"
        lines.append(
            f"| {item['id']} | {item['pipeline']} | {item['status']} | {recall_text} | {item.get('seconds', 0):.2f} | {item.get('items', 0)} |"
        )
    for item in report["cases"]:
        if item["status"] == "SKIP":
            continue
        lines += ["", f"## {item['id']} — {item['status']}", ""]
        for check in item.get("evaluation", {}).get("checks", []):
            lines.append(f"- {'✅' if check['ok'] else '❌'} `{check['expected']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR CPU Lab regression suite")
    parser.add_argument("--sample-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "regression")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--strict-missing", action="store_true")
    args = parser.parse_args()

    suite = load_json(SUITE_PATH)
    document_cases = load_json(DOCUMENT_CASES_PATH)
    label_cases = load_json(LABEL_CASES_PATH)
    sample_root = args.sample_root or Path(suite.get("sample_root", "/data/regression-samples"))

    print(f"[regression] sample_root={sample_root}")
    pipeline = make_pipeline(args.cpu_threads)
    report_cases = []
    total_seconds = 0.0

    for spec in suite.get("cases", []):
        case_id = spec["id"]
        sample = sample_root / spec["sample"]
        if not sample.exists():
            print(f"[regression] SKIP {case_id}: missing {sample}")
            report_cases.append({"id": case_id, "pipeline": spec["pipeline"], "status": "SKIP", "sample": str(sample)})
            continue

        print(f"[regression] RUN {case_id}: {sample}")
        if spec["pipeline"] == "document_text":
            output, seconds, items = run_document_case(pipeline, sample)
            evaluation = evaluate_document(output, document_cases[case_id])
        elif spec["pipeline"] == "label":
            output, seconds, items = run_label_case(pipeline, sample)
            evaluation = evaluate_label(output, label_cases[case_id])
        else:
            raise ValueError(f"Unsupported pipeline: {spec['pipeline']}")

        status = "PASS" if evaluation["passed"] else "FAIL"
        total_seconds += seconds
        print(f"[regression] {status} {case_id}: recall={evaluation['recall']:.3f} seconds={seconds:.2f}")
        report_cases.append({
            "id": case_id,
            "pipeline": spec["pipeline"],
            "status": status,
            "sample": str(sample),
            "seconds": seconds,
            "items": items,
            "evaluation": evaluation,
            "output": output,
        })

    passed = sum(1 for item in report_cases if item["status"] == "PASS")
    failed = sum(1 for item in report_cases if item["status"] == "FAIL")
    skipped = sum(1 for item in report_cases if item["status"] == "SKIP")
    report = {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total_seconds": total_seconds,
        "sample_root": str(sample_root),
        "cases": report_cases,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"[regression] report={args.output_dir / 'latest.md'}")

    if failed:
        return 1
    if args.strict_missing and skipped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
