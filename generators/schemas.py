from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field

class Meta(BaseModel):
    title: str
    page_size: str
    columns: int
    margins_pt: List[int]

class HeadingSizes(BaseModel):
    h1: int
    h2: int

class Styles(BaseModel):
    base_font: str
    base_size: int
    heading_font: str
    heading_sizes: HeadingSizes

class HeadingElement(BaseModel):
    type: Literal["heading"]
    level: str
    text: str

class ParagraphElement(BaseModel):
    type: Literal["paragraph"]
    text: str

class TableElement(BaseModel):
    type: Literal["table"]
    rows: List[List[str]]
    col_widths: str

class FigureElement(BaseModel):
    type: Literal["figure"]
    caption: str
    height_pt: int

Element = Union[HeadingElement, ParagraphElement, TableElement, FigureElement]

class Page(BaseModel):
    elements: List[Element]

class Document(BaseModel):
    meta: Meta
    styles: Styles
    pages: List[Page]
