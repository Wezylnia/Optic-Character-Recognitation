"""
OCR Pipeline temel veri tipleri — TextBox ve OCRResult
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TextBox:
    """Tespit edilen metin kutusu"""
    box: np.ndarray  # [4, 2] polygon koordinatlari
    text: str = ""
    confidence: float = 0.0

    @property
    def x1(self) -> int:
        return int(np.min(self.box[:, 0]))

    @property
    def y1(self) -> int:
        return int(np.min(self.box[:, 1]))

    @property
    def x2(self) -> int:
        return int(np.max(self.box[:, 0]))

    @property
    def y2(self) -> int:
        return int(np.max(self.box[:, 1]))

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            'box': self.box.tolist(),
            'text': self.text,
            'confidence': self.confidence,
            'bbox': [self.x1, self.y1, self.x2, self.y2],
        }


@dataclass
class OCRResult:
    """OCR sonucu"""
    text_boxes: List[TextBox] = field(default_factory=list)
    full_text: str = ""
    processing_time: float = 0.0
    layout: Optional[Any] = None  # DocumentLayout (layout analizi yapildiysa)

    @property
    def text(self) -> str:
        """Tum metni birlestir. Layout varsa yapilandirilmis metin doner."""
        if self.full_text:
            return self.full_text
        if self.layout is not None:
            structured = self.layout.to_structured_text()
            if structured.strip():
                return structured
        return '\n'.join([tb.text for tb in self.text_boxes if tb.text])

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'boxes': [tb.to_dict() for tb in self.text_boxes],
            'processing_time': self.processing_time,
        }
