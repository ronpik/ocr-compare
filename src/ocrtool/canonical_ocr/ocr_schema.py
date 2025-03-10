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
            yield from page.blocks
