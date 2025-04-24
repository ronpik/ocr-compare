from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ocrtool.canonical_ocr.ocr_schema import (
    BoundingBox, Symbol, Word, Line, Paragraph, Block, Page, Document, OcrResult,
    Table, HeaderRow, BodyRow, Cell, Properties, LayoutElement
)

def process_layout_result(layout_result: Dict[str, Any]) -> OcrResult:
    """
    Convert a layout processor result JSON to the canonical OCR schema (OcrResult).

    Args:
        layout_result: The layout processor result as a dictionary.

    Returns:
        OcrResult: The canonical OCR result object.
    """
    document = layout_result.get("document", {})
    document_layout = document.get("documentLayout", {})
    blocks = document_layout.get("blocks", [])
    document_path = Path("document")
    page_no = 1  # Assume single page for now
    page_path = document_path / f"page_{page_no}"
    page_elements: List[LayoutElement[Any]] = []
    table_count = 0

    def make_symbol(text: str, idx: int, word_path: Path) -> Symbol:
        return Symbol(
            element_path=word_path / f"symbol_{idx}",
            text_value=text,
            confidence=None
        )

    def make_word(text: str, word_path: Path) -> Word:
        symbols = [make_symbol(c, i, word_path) for i, c in enumerate(text)]
        return Word(
            element_path=word_path,
            symbols=symbols,
            confidence=1.0
        )

    def make_line(text: str, line_path: Path) -> Line:
        words = [make_word(w, line_path / f"word_{i}") for i, w in enumerate(text.split())]
        return Line(
            element_path=line_path,
            words=words,
            confidence=1.0
        )

    def make_paragraph(text: str, para_path: Path) -> Paragraph:
        lines = [make_line(text, para_path / "line_0")]
        return Paragraph(
            element_path=para_path,
            lines=lines,
            confidence=1.0
        )

    def parse_block(block: Dict[str, Any], block_path: Path, block_idx: int) -> Block:
        """
        Recursively parse a block dict into a Block object, supporting paragraphs, tables, and nested blocks.
        """
        elements: List[LayoutElement[Any]] = []
        block_type = block.get("textBlock", {}).get("type", "TEXT") if "textBlock" in block else (
            "TABLE" if "tableBlock" in block else "BLOCK"
        )
        
        # Handle textBlock with nested blocks
        if "textBlock" in block:
            text_block = block["textBlock"]
            text = text_block.get("text", "")
            
            # Create a paragraph for the direct text
            if text:
                para = make_paragraph(text, block_path / "paragraph_0")
                elements.append(para)
            
            # Process nested blocks within textBlock if they exist
            if "blocks" in text_block and isinstance(text_block["blocks"], list):
                for i, nested_block in enumerate(text_block["blocks"]):
                    nested_path = block_path / f"nested_{i}"
                    nested_obj = parse_block(nested_block, nested_path, i)
                    elements.append(nested_obj)
        
        # Table
        if "tableBlock" in block:
            nonlocal table_count
            table_count += 1
            table_path = block_path / f"table_{table_count}"
            table_block = block["tableBlock"]
            body_rows = table_block.get("bodyRows", [])
            header_row: Optional[HeaderRow] = None
            body_row_objs: List[BodyRow] = []
            for row_idx, row in enumerate(body_rows):
                cells = row.get("cells", [])
                cell_objs = []
                for col_idx, cell in enumerate(cells):
                    # Each cell can have its own blocks (layout structure)
                    cell_blocks = cell.get("blocks", [])
                    parsed_blocks = [parse_block(cb, table_path / f"row_{row_idx}" / f"cell_{col_idx}" / f"block_{i}", i)
                                    for i, cb in enumerate(cell_blocks)]
                    # text_value is the concatenation of all leaf text in the cell
                    cell_text = " ".join(
                        b.get("textBlock", {}).get("text", "") for b in cell_blocks if "textBlock" in b
                    )
                    cell_obj = Cell(
                        element_path=table_path / f"row_{row_idx}" / f"cell_{col_idx}",
                        text_value=cell_text,
                        confidence=1.0,
                        column_no=col_idx,
                        blocks=parsed_blocks
                    )
                    cell_objs.append(cell_obj)
                if row_idx == 0:
                    header_row = HeaderRow(
                        element_path=table_path / f"row_{row_idx}",
                        cells=cell_objs,
                        confidence=1.0,
                        row_no=row_idx
                    )
                else:
                    body_row_objs.append(BodyRow(
                        element_path=table_path / f"row_{row_idx}",
                        cells=cell_objs,
                        confidence=1.0,
                        row_no=row_idx
                    ))
            table_obj = Table(
                element_path=table_path,
                header=header_row,
                body=body_row_objs,
                confidence=1.0,
                table_no=table_count
            )
            elements.append(table_obj)
            
        # Top-level nested blocks (not within textBlock or tableBlock)
        if "blocks" in block and not ("textBlock" in block or "tableBlock" in block):
            for i, subblock in enumerate(block["blocks"]):
                subblock_obj = parse_block(subblock, block_path / f"block_{i}", i)
                elements.append(subblock_obj)
                
        # Extract page_span from block if present
        page_span_dict = block.get('pageSpan', {})
        page_start = page_span_dict.get('pageStart', page_no)
        page_end = page_span_dict.get('pageEnd', page_no)
        page_span = (page_start, page_end)
        
        return Block(
            element_path=block_path,
            elements=elements,
            blockType=block_type,
            confidence=1.0,
            block_no=block_idx,
            page_span=page_span
        )

    for block_idx, block in enumerate(blocks):
        block_path = page_path / f"block_{block_idx}"
        block_obj = parse_block(block, block_path, block_idx)
        page_elements.append(block_obj)

    page = Page(
        element_path=page_path,
        width=0,
        height=0,
        blocks=page_elements,  # blocks is now a list of Block, each with elements
        confidence=1.0,
        page_no=page_no
    )
    document_obj = Document(
        element_path=document_path,
        pages=[page]
    )
    return OcrResult(document=document_obj)
