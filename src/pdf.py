from pathlib import Path

import pymupdf
from PIL import Image


def _render_pdf_page(page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def load_document_image(file_path: str, dpi: int = 180) -> Image.Image:
    """Return an RGB PIL image from an image file or the first page of a PDF."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        document = pymupdf.open(path)
        try:
            if document.page_count == 0:
                raise ValueError("PDF contains no pages.")
            return _render_pdf_page(document.load_page(0), dpi)
        finally:
            document.close()

    with Image.open(path) as image:
        return image.convert("RGB")


def load_document_pages(file_path: str, dpi: int = 180) -> list[Image.Image]:
    """Return all PDF pages as RGB PIL images, or a one-item list for an image file."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        with Image.open(path) as image:
            return [image.convert("RGB")]

    document = pymupdf.open(path)
    try:
        if document.page_count == 0:
            raise ValueError("PDF contains no pages.")
        return [_render_pdf_page(document.load_page(index), dpi) for index in range(document.page_count)]
    finally:
        document.close()


def document_page_count(file_path: str) -> int:
    """Return PDF page count or 1 for supported image files."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return 1
    document = pymupdf.open(path)
    try:
        return document.page_count
    finally:
        document.close()
