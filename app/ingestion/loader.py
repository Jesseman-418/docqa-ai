"""Document loaders for PDF, TXT, and Markdown files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.utils.text import clean_text

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A loaded document with its text content and metadata."""

    text: str
    metadata: dict

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def num_characters(self) -> int:
        return len(self.text)


def load_pdf(file_path: Path | str, raw_bytes: bytes | None = None) -> Document:
    """Load a PDF file and extract text from all pages.

    Args:
        file_path: Path to the PDF (used for metadata even when raw_bytes given).
        raw_bytes: If provided, read the PDF from memory instead of disk.
    """
    from PyPDF2 import PdfReader
    import io

    path = Path(file_path)

    if raw_bytes is not None:
        reader = PdfReader(io.BytesIO(raw_bytes))
    else:
        reader = PdfReader(str(path))

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    full_text = clean_text("\n\n".join(pages))
    logger.info("Loaded PDF %s — %d pages, %d chars", path.name, len(reader.pages), len(full_text))

    return Document(
        text=full_text,
        metadata={
            "source": path.name,
            "file_type": "pdf",
            "num_pages": len(reader.pages),
        },
    )


def load_text(file_path: Path | str, raw_bytes: bytes | None = None) -> Document:
    """Load a plain-text or Markdown file.

    Args:
        file_path: Path to the file (used for metadata even when raw_bytes given).
        raw_bytes: If provided, decode from memory instead of reading disk.
    """
    path = Path(file_path)

    if raw_bytes is not None:
        content = raw_bytes.decode("utf-8", errors="replace")
    else:
        content = path.read_text(encoding="utf-8", errors="replace")

    full_text = clean_text(content)
    ext = path.suffix.lower().lstrip(".")
    logger.info("Loaded %s file %s — %d chars", ext.upper(), path.name, len(full_text))

    return Document(
        text=full_text,
        metadata={
            "source": path.name,
            "file_type": ext,
        },
    )


def load_document(file_path: Path | str, raw_bytes: bytes | None = None) -> Document:
    """Auto-detect file type and load accordingly.

    Args:
        file_path: Path to the document.
        raw_bytes: Optional in-memory bytes (e.g. from Streamlit upload).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(path, raw_bytes=raw_bytes)
    if ext in {".txt", ".md", ".markdown"}:
        return load_text(path, raw_bytes=raw_bytes)

    raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md")
