from pathlib import Path
from typing import Any, Dict, List, Optional

from ocrtool.canonical_ocr.ocr_schema import (
    BoundingBox, Symbol, Word, Line, Paragraph, Block, Page, Document, OcrResult,
    Table, HeaderRow, BodyRow, Cell, Properties
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
    page_blocks: List[Block] = []
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

    for block_idx, block in enumerate(blocks):
        block_path = page_path / f"block_{block_idx}"
        if "textBlock" in block:
            text = block["textBlock"].get("text", "")
            para = make_paragraph(text, block_path / "paragraph_0")
            blk = Block(
                element_path=block_path,
                paragraphs=[para],
                blockType=block["textBlock"].get("type", "TEXT"),
                confidence=1.0,
                block_no=block_idx
            )
            page_blocks.append(blk)
        elif "tableBlock" in block:
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
                    cell_blocks = cell.get("blocks", [])
                    cell_text = " ".join(
                        b.get("textBlock", {}).get("text", "") for b in cell_blocks if "textBlock" in b
                    )
                    cell_obj = Cell(
                        element_path=table_path / f"row_{row_idx}" / f"cell_{col_idx}",
                        text_value=cell_text,
                        confidence=1.0,
                        column_no=col_idx
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
            # Wrap table in a paragraph and block for schema compatibility
            para = Paragraph(
                element_path=table_path / "paragraph_0",
                lines=[],
                confidence=1.0
            )
            blk = Block(
                element_path=block_path,
                paragraphs=[para],
                blockType="TABLE",
                confidence=1.0,
                block_no=block_idx,
                extra={"table": table_obj}
            )
            page_blocks.append(blk)

    page = Page(
        element_path=page_path,
        width=0,
        height=0,
        blocks=page_blocks,
        confidence=1.0,
        page_no=page_no
    )
    document = Document(
        element_path=document_path,
        pages=[page]
    )
    return OcrResult(document=document)
