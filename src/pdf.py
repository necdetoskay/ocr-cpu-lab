from pathlib import Path

import pymupdf
from PIL import Image


def _render_pdf_page(page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


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


def load_document_page(file_path: str, page_index: int = 0, dpi: int = 180) -> Image.Image:
    """Render one zero-based PDF page, or return an image file as RGB."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        if page_index != 0:
            raise IndexError("Image inputs only contain one page.")
        with Image.open(path) as image:
            return image.convert("RGB")

    document = pymupdf.open(path)
    try:
        if document.page_count == 0:
            raise ValueError("PDF contains no pages.")
        if page_index < 0 or page_index >= document.page_count:
            raise IndexError(f"Page index {page_index} is out of range for {document.page_count} pages.")
        return _render_pdf_page(document.load_page(page_index), dpi)
    finally:
        document.close()


def load_document_image(file_path: str, dpi: int = 180) -> Image.Image:
    """Backward-compatible helper returning the first page/image."""
    return load_document_page(file_path, page_index=0, dpi=dpi)
