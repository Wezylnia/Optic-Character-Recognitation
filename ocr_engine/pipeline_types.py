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


def _group_boxes_to_text(text_boxes) -> str:
    """
    TextBox listesini satirlara gore gruplar ve metni birlestirir.
    Ayni satirda (y-merkezi yakin) olan kutular boslukla, farkli satirlar
    yeni satir ile ayrilir. Kelime duzeyinde TextBox'larda dogru calisir.
    """
    if not text_boxes:
        return ''

    # y merkezine gore sirala
    items = sorted(text_boxes, key=lambda tb: (tb.y1 + tb.y2) / 2)

    lines = []
    current_line = [items[0]]

    for tb in items[1:]:
        prev        = current_line[-1]
        avg_h       = ((prev.y2 - prev.y1) + (tb.y2 - tb.y1)) / 2
        y_diff      = abs((tb.y1 + tb.y2) / 2 - (prev.y1 + prev.y2) / 2)
        if y_diff < avg_h * 0.6:          # ayni satirda
            current_line.append(tb)
        else:
            lines.append(current_line)
            current_line = [tb]
    lines.append(current_line)

    result = []
    for line in lines:
        line_sorted = sorted(line, key=lambda tb: tb.x1)
        result.append(' '.join(tb.text for tb in line_sorted if tb.text))
    return '\n'.join(r for r in result if r)


@dataclass
class OCRResult:
    """OCR sonucu"""
    text_boxes: List[TextBox] = field(default_factory=list)
    full_text: str = ""
    processing_time: float = 0.0
    layout: Optional[Any] = None  # DocumentLayout (layout analizi yapildiysa)
    source_image: Optional[Any] = None  # preprocessed goruntu (numpy array)

    @property
    def text(self) -> str:
        """Tum metni birlestir. Layout varsa yapilandirilmis metin doner."""
        if self.full_text:
            return self.full_text
        if self.layout is not None:
            structured = self.layout.to_structured_text()
            if structured.strip():
                return structured
        return _group_boxes_to_text(self.text_boxes)

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'boxes': [tb.to_dict() for tb in self.text_boxes],
            'processing_time': self.processing_time,
        }
