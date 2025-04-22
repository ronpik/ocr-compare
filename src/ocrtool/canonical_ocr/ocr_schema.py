from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterable, Dict, TypeVar, Generic, Any, TYPE_CHECKING, Union, Tuple
from enum import Enum, auto
import pandas as pd


@dataclass(frozen=True)
class BoundingBox:
    left: Optional[float] = None
    top: Optional[float] = None
    height: Optional[float] = None
    width: Optional[float] = None


Properties = Dict[str, Any]

E = TypeVar('E', bound='LayoutElement')


@dataclass(kw_only=True)
class LayoutElement(Generic[E], ABC):
    element_path: Optional[Path] = None
    boundingBox: Optional[BoundingBox] = None
    extra: Properties = field(default_factory=dict)

    @abstractmethod
    def children(self) -> Iterable[E]:
        raise NotImplementedError

    @abstractmethod
    def text(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def type(cls) -> str:
        raise NotImplementedError


class LayoutElementType(str, Enum):
    """Enumeration of all canonical OCR layout element types."""
    SYMBOL = "SYMBOL"
    WORD = "WORD" 
    LINE = "LINE"
    PARAGRAPH = "PARAGRAPH"
    BLOCK = "BLOCK"
    PAGE = "PAGE"
    DOCUMENT = "DOCUMENT"
    CELL = "CELL"
    ROW = "ROW"
    HEADER_ROW = "HEADER_ROW"
    BODY_ROW = "BODY_ROW"
    TABLE = "TABLE"


@dataclass
class Symbol(LayoutElement[None]):
    text_value: str
    confidence: Optional[float]

    def children(self) -> Iterable[None]:
        return []  # Symbols have no children

    def text(self) -> str:
        return self.text_value

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.SYMBOL


@dataclass
class Word(LayoutElement[Symbol]):
    symbols: List[Symbol]
    confidence: float

    def children(self) -> Iterable[Symbol]:
        return self.symbols

    def text(self) -> str:
        return ''.join(symbol.text() for symbol in self.symbols)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.WORD


@dataclass
class Line(LayoutElement[Word]):
    words: List[Word]
    confidence: float

    def children(self) -> Iterable[Word]:
        return self.words

    def text(self) -> str:
        return ' '.join(word.text() for word in self.words)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.LINE


@dataclass
class Paragraph(LayoutElement[Line]):
    lines: List[Line]
    confidence: float

    def children(self) -> Iterable[Line]:
        return self.lines

    def text(self) -> str:
        return ' '.join(line.text() for line in self.lines)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.PARAGRAPH


if TYPE_CHECKING:
    from ocrtool.canonical_ocr.ocr_schema import Table, Paragraph, Block

@dataclass
class Block(LayoutElement[Union['Table', 'Paragraph', 'Block']]):
    """
    Represents a block element, which can contain paragraphs, tables, or other blocks as children.
    """
    elements: List['Table | Paragraph | Block']
    blockType: str
    confidence: float
    block_no: Optional[int] = None
    page_span: Tuple[int, int] = (-1, -1)

    def children(self) -> Iterable[Union['Table', 'Paragraph', 'Block']]:
        """Yield all child elements (Paragraph, Table, or Block)."""
        return self.elements

    def text(self) -> str:
        """Return the concatenated text of all child elements."""
        return '\n'.join(element.text() for element in self.elements)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.BLOCK

    @property
    def start_page(self) -> int:
        """Return the starting page number for this block."""
        return self.page_span[0]

    @property
    def end_page(self) -> int:
        """Return the ending page number for this block."""
        return self.page_span[1]

    @property
    def number_of_pages(self) -> int:
        """Return the number of pages spanned by this block."""
        return self.page_span[1] - self.page_span[0] + 1 if self.page_span[0] >= 0 and self.page_span[1] >= 0 else 0


@dataclass
class Page(LayoutElement[Block]):
    width: int
    height: int
    blocks: List[Block]
    confidence: float
    page_no: Optional[int] = -1

    def children(self) -> Iterable[Block]:
        return self.blocks

    def text(self) -> str:
        return '\n\n'.join(block.text() for block in self.blocks)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.PAGE


@dataclass
class Document(LayoutElement[Page]):
    pages: List[Page]

    def children(self) -> Iterable[Page]:
        return self.pages

    def text(self) -> str:
        return '\n=====\n'.join(page.text() for page in self.pages)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.DOCUMENT


@dataclass
class OcrResult:
    document: Document

    def blocks(self) -> Iterable[Block]:
        for page in self.document.children():
            yield from page.blocks


@dataclass
class Cell(LayoutElement[Block]):
    """
    Represents a single cell in a table row, which can contain blocks as children.
    """
    text_value: str
    confidence: Optional[float]
    column_no: Optional[int] = None
    blocks: List[Block] = field(default_factory=list)

    def children(self) -> Iterable[Block]:
        """Yield all block children of this cell."""
        return self.blocks

    def text(self) -> str:
        """Return the concatenated text of all blocks, or text_value if no blocks."""
        if self.blocks:
            return '\n'.join(block.text() for block in self.blocks)
        return self.text_value

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.CELL


@dataclass
class Row(LayoutElement[Cell]):
    """Represents a row in a table, consisting of cells."""
    cells: List[Cell]
    confidence: Optional[float] = None
    row_no: Optional[int] = None

    def children(self) -> Iterable[Cell]:
        """Return the cells in this row."""
        return self.cells

    def text(self) -> str:
        """Return the concatenated text of all cells in the row, separated by tabs."""
        return '\t'.join(cell.text() for cell in self.cells)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.ROW


@dataclass
class HeaderRow(Row):
    """Represents the header row of a table."""
    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.HEADER_ROW


@dataclass
class BodyRow(Row):
    """Represents a body row of a table."""
    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.BODY_ROW


@dataclass
class Table(LayoutElement[Row]):
    """Represents a table layout element, with header and body rows."""
    header: Optional[HeaderRow] = None
    body: List[BodyRow] = field(default_factory=list)
    confidence: Optional[float] = None
    table_no: Optional[int] = None

    def children(self) -> Iterable[Row]:
        """Return the header and body rows as children."""
        if self.header:
            yield self.header
        yield from self.body

    def as_dataframe(self) -> pd.DataFrame:
        """
        Convert the table to a pandas DataFrame, using the header row (if present) as columns.
        Returns:
            pd.DataFrame: The table as a DataFrame.
        """
        # Get header values
        if self.header:
            columns = [cell.text() for cell in self.header.cells]
        else:
            # If no header, use default column names
            if self.body and self.body[0].cells:
                columns = [f"column_{i}" for i in range(len(self.body[0].cells))]
            else:
                columns = []
        # Get body values
        data = []
        for row in self.body:
            data.append([cell.text() for cell in row.cells])
        return pd.DataFrame(data, columns=columns)

    def text(self) -> str:
        """
        Return the table as markdown using pandas DataFrame rendering.
        Returns:
            str: The table as a markdown string.
        """
        df = self.as_dataframe()
        return df.to_markdown(index=False)

    def raw_text(self) -> str:
        """
        Return the table as TSV using the inner text elements (legacy behavior).
        Returns:
            str: The table as TSV.
        """
        rows = []
        if self.header:
            rows.append('\t'.join(cell.text() for cell in self.header.cells))
        rows.extend('\t'.join(cell.text() for cell in row.cells) for row in self.body)
        return '\n'.join(rows)

    @classmethod
    def type(cls) -> LayoutElementType:
        """Return the type enum for this element."""
        return LayoutElementType.TABLE



@dataclass
class OcrResultSummary:
    """
    Summary statistics for an OcrResult, for logging and reporting.
    """
    num_pages: int
    num_blocks: int
    num_tables: int
    num_words: int
    total_length: int

    @classmethod
    def from_ocr_result(cls, result: OcrResult) -> "OcrResultSummary":
        """
        Create a summary from an OcrResult, using both the number of Page layout elements and the page spans in blocks.
        The number of pages is the maximum of the number of Page elements and the highest end_page in all blocks.
        """
        num_blocks = 0
        num_tables = 0
        num_words = 0
        max_end_page = 0
        total_length = len(result.document.text())
        for page in result.document.pages:
            for block in (page.blocks or []):
                num_blocks += 1
                if isinstance(block, Table):
                    num_tables += 1
                
                if hasattr(block, 'page_span') and block.page_span[1] is not None:
                    max_end_page = max(max_end_page, block.page_span[1])

                for element in block.elements or []:
                    # Handle different element types appropriately
                    if isinstance(element, Paragraph):
                        # Process Paragraph objects with lines
                        for line in element.lines or []:
                            for word in line.words or []:
                                num_words += 1
                    elif isinstance(element, Table):
                        # Process Table objects by iterating through rows and cells
                        if element.header:
                            for cell in element.header.cells or []:
                                # Count words in header cells
                                for b in cell.blocks or []:
                                    for p in b.elements or []:
                                        if isinstance(p, Paragraph):
                                            for line in p.lines or []:
                                                for word in line.words or []:
                                                    num_words += 1
                        
                        # Process body rows
                        for row in element.body or []:
                            for cell in row.cells or []:
                                # Count words in body cells
                                for b in cell.blocks or []:
                                    for p in b.elements or []:
                                        if isinstance(p, Paragraph):
                                            for line in p.lines or []:
                                                for word in line.words or []:
                                                    num_words += 1
                    elif isinstance(element, Block):
                        # Process nested Block objects recursively
                        # This is a simplified approach; a complete solution might need a recursive function
                        for sub_element in element.elements or []:
                            if isinstance(sub_element, Paragraph):
                                for line in sub_element.lines or []:
                                    for word in line.words or []:
                                        num_words += 1
                            
        num_layout_pages = len(result.document.pages)
        num_pages = max(num_layout_pages, max_end_page)
        return cls(
            num_pages=num_pages,
            num_blocks=num_blocks,
            num_tables=num_tables,
            num_words=num_words,
            total_length=total_length,
        )