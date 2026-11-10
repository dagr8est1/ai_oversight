from typing import Any, Literal

from pydantic import BaseModel

ElementType = Literal["text", "key_value", "table", "chart"]


class Element(BaseModel):
    element_id: str
    type: ElementType
    page_or_slide: int
    bbox: list[float] | None = None
    method: str
    title: str | None = None
    content: Any | None = None
    raw_text: str | None = None


class ParsedDocument(BaseModel):
    document_id: str
    filename: str
    format: Literal["pdf", "pptx"]
    elements: list[Element] = []
    warnings: list[str] = []
