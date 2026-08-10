from pathlib import Path

import pymupdf
from PIL import Image


def load_document_image(file_path: str, dpi: int = 180) -> Image.Image:
    """Return an RGB PIL image from an image file or the first page of a PDF."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        document = pymupdf.open(path)
        try:
            if document.page_count == 0:
                raise ValueError("PDF contains no pages.")
            page = document.load_page(0)
            scale = dpi / 72.0
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        finally:
            document.close()

    with Image.open(path) as image:
        return image.convert("RGB")
