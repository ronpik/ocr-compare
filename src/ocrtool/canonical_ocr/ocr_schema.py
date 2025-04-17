from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterable, Dict, TypeVar, Generic, Any


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


@dataclass
class Symbol(LayoutElement[None]):
    text_value: str
    confidence: Optional[float]

    def children(self) -> Iterable[None]:
        return []  # Symbols have no children

    def text(self) -> str:
        return self.text_value

    @classmethod
    def type(cls) -> str:
        return "symbol"


@dataclass
class Word(LayoutElement[Symbol]):
    symbols: List[Symbol]
    confidence: float

    def children(self) -> Iterable[Symbol]:
        return self.symbols

    def text(self) -> str:
        return ''.join(symbol.text() for symbol in self.symbols)

    @classmethod
    def type(cls) -> str:
        return "word"


@dataclass
class Line(LayoutElement[Word]):
    words: List[Word]
    confidence: float

    def children(self) -> Iterable[Word]:
        return self.words

    def text(self) -> str:
        return ' '.join(word.text() for word in self.words)

    @classmethod
    def type(cls) -> str:
        return "line"


@dataclass
class Paragraph(LayoutElement[Line]):
    lines: List[Line]
    confidence: float

    def children(self) -> Iterable[Line]:
        return self.lines

    def text(self) -> str:
        return ' '.join(line.text() for line in self.lines)

    @classmethod
    def type(cls) -> str:
        return "paragraph"


@dataclass
class Block(LayoutElement[Paragraph]):
    paragraphs: List[Paragraph]
    blockType: str
    confidence: float
    block_no: Optional[int] = None

    def children(self) -> Iterable[Paragraph]:
        return self.paragraphs

    def text(self) -> str:
        return '\n'.join(paragraph.text() for paragraph in self.paragraphs)

    @classmethod
    def type(cls) -> str:
        return "block"


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
    def type(cls) -> str:
        return "page"


@dataclass
class Document(LayoutElement[Page]):
    pages: List[Page]

    def children(self) -> Iterable[Page]:
        return self.pages

    def text(self) -> str:
        return '\n=====\n'.join(page.text() for page in self.pages)

    @classmethod
    def type(cls) -> str:
        return 'document'


@dataclass
class OcrResult:
    document: Document

    def blocks(self) -> Iterable[Block]:
        for page in self.document.children():
            yield from page.blocks;


@dataclass
class Cell(LayoutElement[None]):
    """Represents a single cell in a table row."""
    text_value: str
    confidence: Optional[float]
    column_no: Optional[int] = None

    def children(self) -> Iterable[None]:
        """Cells have no children."""
        return []

    def text(self) -> str:
        """Return the text value of the cell."""
        return self.text_value

    @classmethod
    def type(cls) -> str:
        """Return the type name for this element."""
        return "cell"


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
    def type(cls) -> str:
        """Return the type name for this element."""
        return "row"


@dataclass
class HeaderRow(Row):
    """Represents the header row of a table."""
    def type(cls) -> str:
        """Return the type name for this element."""
        return "header_row"


@dataclass
class BodyRow(Row):
    """Represents a body row of a table."""
    def type(cls) -> str:
        """Return the type name for this element."""
        return "body_row"


@dataclass
class Table(LayoutElement[Row]):
    """Represents a table layout element, with header and body rows."""
    header: Optional[HeaderRow] = None
    body: List[BodyRow] = field(default_factory=list)
    confidence: Optional[float] = None
    table_no: Optional[int] = None

    def children(self) -> Iterable[Row]:
        """Return the header and body rows as children."""
        children: List[Row] = []
        if self.header:
            children.append(self.header)
        children.extend(self.body)
        return children

    def text(self) -> str:
        """Return the text of the table as TSV (header + body)."""
        rows = []
        if self.header:
            rows.append(self.header.text())
        rows.extend(row.text() for row in self.body)
        return '\n'.join(rows)

    @classmethod
    def type(cls) -> str:
        """Return the type name for this element."""
        return "table"
