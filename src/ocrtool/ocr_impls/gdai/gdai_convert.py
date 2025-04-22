from pathlib import Path
from typing import Dict, Any, List

from ocrtool.canonical_ocr.ocr_schema import (
    BoundingBox, Symbol, Word, Line, Paragraph, Block, Page, Document, OcrResult,
    Properties
)


def vertices_to_bounding_box(vertices: List[Dict[str, float]]) -> BoundingBox:
    """
    Convert vertices format from DocumentAI to BoundingBox format.

    Args:
        vertices: List of vertices (x, y coordinates)

    Returns:
        BoundingBox with left, top, width, height
    """
    if not vertices:
        return BoundingBox()

    # Extract x and y values, ignoring None values
    x_values = [v.get("x") for v in vertices if v.get("x") is not None]
    y_values = [v.get("y") for v in vertices if v.get("y") is not None]

    if not x_values or not y_values:
        return BoundingBox()

    # Calculate bounding box properties
    left = min(x_values)
    top = min(y_values)
    right = max(x_values)
    bottom = max(y_values)

    width = right - left
    height = bottom - top

    return BoundingBox(
        left=left,
        top=top,
        width=width,
        height=height
    )


def extract_text_from_anchor(document_text: str, text_anchor: Dict[str, Any]) -> str:
    """
    Extract text from document using text anchor information.

    Args:
        document_text: Full document text
        text_anchor: Text anchor information with text segments

    Returns:
        Extracted text
    """
    text_segments = text_anchor.get("textSegments", [])
    if not text_segments:
        return ""

    text_parts = []
    for segment in text_segments:
        start_index = segment.get("startIndex", 0)
        end_index = segment.get("endIndex", 0)
        if start_index is not None and end_index is not None:
            text_parts.append(document_text[int(start_index):int(end_index)])

    return "".join(text_parts)


def convert_documentai_to_ocr(documentai_result: Dict[str, Any]) -> OcrResult:
    """
    Convert Google DocumentAI OCR result to OcrResult format.

    Args:
        documentai_result: JSON result from Google DocumentAI

    Returns:
        OcrResult object with converted data
    """
    # Extract the document from the result
    document_data = documentai_result.get("document", {})
    document_text = document_data.get("text", "")

    # Create root document path
    document_path = Path("document")

    # Process all pages from DocumentAI
    ocr_pages = []

    # Calculate document dimensions by combining all pages
    all_page_widths = []
    all_page_heights = []

    for page_idx, page_data in enumerate(document_data.get("pages", [])):
        page_number = page_data.get("pageNumber", page_idx + 1)
        page_path = document_path / f"page_{page_number}"

        # Extract page dimensions
        dimension = page_data.get("dimension", {})
        width = int(dimension.get("width", 0))
        height = int(dimension.get("height", 0))

        all_page_widths.append(width)
        all_page_heights.append(height)

        # Extract page confidence
        page_layout = page_data.get("layout", {})
        page_confidence = page_layout.get("confidence", 0.0)

        # Create page bounding box
        page_bounding_box = BoundingBox(left=0, top=0, width=width, height=height)

        # Store page properties and extra information
        page_extra: Properties = {
            "detectedLanguages": [
                {
                    "languageCode": lang.get("languageCode", ""),
                    "confidence": lang.get("confidence", 0.0)
                }
                for lang in page_data.get("detectedLanguages", [])
            ],
            "orientation": page_layout.get("orientation", ""),
            "mimeType": document_data.get("mimeType", ""),
            "layout": {
                "confidence": page_layout.get("confidence", 0.0),
                "textAnchor": page_layout.get("textAnchor", {})
            }
        }

        # Process blocks for this page using text anchors for matching
        # We'll use dictionaries to track processed elements by their text anchor indices
        processed_block_indices = set()
        processed_para_indices = set()
        processed_line_indices = set()
        processed_token_indices = set()

        # Get blocks
        blocks = []

        # Process blocks
        for block_idx, block_data in enumerate(page_data.get("blocks", [])):
            if block_idx in processed_block_indices:
                continue

            processed_block_indices.add(block_idx)

            block_path = page_path / f"block_{block_idx}"
            block_layout = block_data.get("layout", {})
            block_confidence = block_layout.get("confidence", 0.0)

            # Create bounding box for the block
            bounding_poly = block_layout.get("boundingPoly", {})
            vertices = bounding_poly.get("vertices", [])
            block_box = vertices_to_bounding_box(vertices)

            # Get block text anchor
            block_text_anchor = block_layout.get("textAnchor", {})

            # Store block extra information
            block_extra: Properties = {
                "orientation": block_layout.get("orientation", ""),
                "textAnchor": block_text_anchor,
                # "normalized_vertices": bounding_poly.get("normalizedVertices", [])
            }

            # Process paragraphs for this block - match by spatial containment
            paragraphs = []

            # Find paragraphs that are contained within this block's bounding box
            for para_idx, para_data in enumerate(page_data.get("paragraphs", [])):
                if para_idx in processed_para_indices:
                    continue

                para_layout = para_data.get("layout", {})
                para_bounding_poly = para_layout.get("boundingPoly", {})
                para_vertices = para_bounding_poly.get("vertices", [])
                para_box = vertices_to_bounding_box(para_vertices)

                # Check if paragraph is within block (simple containment check)
                # A more robust implementation would check for overlap percentage
                if (para_box.left is not None and para_box.top is not None and
                        block_box.left is not None and block_box.top is not None and
                        para_box.left >= block_box.left and
                        para_box.top >= block_box.top and
                        para_box.left + para_box.width <= block_box.left + block_box.width and
                        para_box.top + para_box.height <= block_box.top + block_box.height):

                    processed_para_indices.add(para_idx)

                    para_path = block_path / f"paragraph_{para_idx}"
                    para_confidence = para_layout.get("confidence", 0.0)

                    # Get paragraph text anchor
                    para_text_anchor = para_layout.get("textAnchor", {})

                    # Store paragraph extra information
                    para_extra: Properties = {
                        "orientation": para_layout.get("orientation", ""),
                        "textAnchor": para_text_anchor,
                        # "normalized_vertices": para_bounding_poly.get("normalizedVertices", [])
                    }

                    # Process lines for this paragraph
                    lines = []

                    # Find lines that are contained within this paragraph's bounding box
                    for line_idx, line_data in enumerate(page_data.get("lines", [])):
                        if line_idx in processed_line_indices:
                            continue

                        line_layout = line_data.get("layout", {})
                        line_bounding_poly = line_layout.get("boundingPoly", {})
                        line_vertices = line_bounding_poly.get("vertices", [])
                        line_box = vertices_to_bounding_box(line_vertices)

                        # Check if line is within paragraph (simple containment check)
                        if (line_box.left is not None and line_box.top is not None and
                                para_box.left is not None and para_box.top is not None and
                                line_box.left >= para_box.left and
                                line_box.top >= para_box.top and
                                line_box.left + line_box.width <= para_box.left + para_box.width and
                                line_box.top + line_box.height <= para_box.top + para_box.height):

                            processed_line_indices.add(line_idx)

                            line_path = para_path / f"line_{line_idx}"
                            line_confidence = line_layout.get("confidence", 0.0)

                            # Get line text anchor
                            line_text_anchor = line_layout.get("textAnchor", {})

                            # Store line extra information
                            line_extra: Properties = {
                                "orientation": line_layout.get("orientation", ""),
                                "textAnchor": line_text_anchor,
                                "detectedLanguages": [
                                    {
                                        "languageCode": lang.get("languageCode", ""),
                                        "confidence": lang.get("confidence", 0.0)
                                    }
                                    for lang in line_data.get("detectedLanguages", [])
                                ],
                                # "normalized_vertices": line_bounding_poly.get("normalizedVertices", [])
                            }

                            # Process words for this line
                            words = []

                            # Find tokens that are contained within this line's bounding box
                            for token_idx, token_data in enumerate(page_data.get("tokens", [])):
                                if token_idx in processed_token_indices:
                                    continue

                                token_layout = token_data.get("layout", {})
                                token_bounding_poly = token_layout.get("boundingPoly", {})
                                token_vertices = token_bounding_poly.get("vertices", [])
                                token_box = vertices_to_bounding_box(token_vertices)

                                # Check if token is within line (simple containment check)
                                if (token_box.left is not None and token_box.top is not None and
                                        line_box.left is not None and line_box.top is not None and
                                        token_box.left >= line_box.left and
                                        token_box.top >= line_box.top and
                                        token_box.left + token_box.width <= line_box.left + line_box.width and
                                        token_box.top + token_box.height <= line_box.top + line_box.height):

                                    processed_token_indices.add(token_idx)

                                    word_path = line_path / f"word_{token_idx}"
                                    token_confidence = token_layout.get("confidence", 0.0)

                                    # Get token text anchor
                                    text_anchor = token_layout.get("textAnchor", {})

                                    # Store word extra information
                                    token_extra: Properties = {
                                        "detectedBreak": token_data.get("detectedBreak", {}),
                                        "textAnchor": text_anchor,
                                        # "normalized_vertices": token_bounding_poly.get("normalizedVertices", [])
                                    }

                                    # Extract token text using text anchors
                                    token_text = extract_text_from_anchor(document_text, text_anchor)

                                    # Create a symbol for each character in the token text
                                    symbols = []
                                    for i, char in enumerate(token_text):
                                        symbol_path = word_path / f"symbol_{i}"

                                        # Create character-level symbol without bounding box
                                        symbol = Symbol(
                                            element_path=symbol_path,
                                            text_value=char,
                                            confidence=token_confidence
                                        )
                                        symbols.append(symbol)

                                    # Create a word with these symbols
                                    if symbols:  # Only add non-empty words
                                        word = Word(
                                            boundingBox=token_box,
                                            element_path=word_path,
                                            extra=token_extra,
                                            symbols=symbols,
                                            confidence=token_confidence
                                        )
                                        words.append(word)

                            # Add the line if we found any words
                            if words:
                                line = Line(
                                    boundingBox=line_box,
                                    element_path=line_path,
                                    extra=line_extra,
                                    words=words,
                                    confidence=line_confidence
                                )
                                lines.append(line)

                    # Add the paragraph if we found any lines
                    if lines:
                        paragraph = Paragraph(
                            boundingBox=para_box,
                            element_path=para_path,
                            extra=para_extra,
                            lines=lines,
                            confidence=para_confidence
                        )
                        paragraphs.append(paragraph)

            # Add the block if we found any paragraphs
            if paragraphs:
                block = Block(
                    boundingBox=block_box,
                    element_path=block_path,
                    extra=block_extra,
                    elements=paragraphs,
                    blockType="TEXT",  # Default type, DocumentAI doesn't specify block types as clearly
                    confidence=block_confidence,
                    block_no=block_idx,
                    page_span=(page_number, page_number)
                )
                blocks.append(block)

        # Create the page object
        page = Page(
            boundingBox=page_bounding_box,
            element_path=page_path,
            extra=page_extra,
            width=width,
            height=height,
            blocks=blocks,
            confidence=page_confidence,
            page_no=page_number
        )

        ocr_pages.append(page)

    # Calculate document dimensions
    doc_width = max(all_page_widths) if all_page_widths else 0
    doc_height = sum(all_page_heights) if all_page_heights else 0

    # Create document bounding box
    doc_bounding_box = BoundingBox(
        left=0,
        top=0,
        width=doc_width,
        height=doc_height
    )

    # Create document extra properties
    doc_extra: Properties = {
        "mimeType": document_data.get("mimeType", ""),
        "uri": document_data.get("uri", ""),
        "humanReviewStatus": documentai_result.get("humanReviewStatus", {}),
        "text": document_text
    }

    # Create Document object
    document = Document(
        boundingBox=doc_bounding_box,
        element_path=document_path,
        extra=doc_extra,
        pages=ocr_pages
    )

    # Create the final OcrResult
    return OcrResult(document=document)


def process_documentai_result(documentai_result: dict) -> OcrResult:
    """
    Process DocumentAI JSON result and convert it to OcrResult.

    Args:
        json_data: JSON string with DocumentAI result

    Returns:
        OcrResult object
    """
    # documentai_result = json.loads(json_data)
    return convert_documentai_to_ocr(documentai_result)
