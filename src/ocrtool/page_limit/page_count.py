from typing import Any, List
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter


def is_pdf(data: bytes) -> bool:
    """
    Check if the given bytes represent a PDF file.

    Args:
        data: The file data as bytes.

    Returns:
        bool: True if the data is a PDF, False otherwise.
    """
    return data.startswith(b'%PDF')


def count_pdf_pages(pdf_bytes: bytes) -> int:
    """
    Count the number of pages in a PDF file.

    Args:
        pdf_bytes: The PDF file data as bytes.

    Returns:
        int: The number of pages in the PDF.
    """
    try:
        import PyPDF2
    except ImportError as e:
        raise ImportError("PyPDF2 is required for counting PDF pages.") from e
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    return len(reader.pages)


def split_pdf_to_segments(pdf_bytes: bytes, page_limit: int) -> List[bytes]:
    """
    Split a PDF into segments, each with at most page_limit pages.

    Args:
        pdf_bytes: The PDF file data as bytes.
        page_limit: The maximum number of pages per segment.

    Returns:
        List[bytes]: List of PDF segments as bytes.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    segments = []
    for i in range(0, len(reader.pages), page_limit):
        writer = PdfWriter()
        for page in reader.pages[i:i+page_limit]:
            writer.add_page(page)
        buf = BytesIO()
        writer.write(buf)
        segments.append(buf.getvalue())
    return segments 