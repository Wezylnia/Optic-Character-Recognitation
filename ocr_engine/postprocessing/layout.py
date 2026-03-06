"""
Layout analiz modulu

Belge yapısını tespit eder:
  - Sütun sayısı ve sınırları  (projeksiyon profili)
  - Başlık / paragraf / gövde metin sınıflandırması  (font yüksekliği)
  - Okuma sırası (Z-düzeni: soldan sağa, yukarıdan aşağıya, sütun sıralı)
  - Mantıksal blokları (başlık + altındaki paragraflar) gruplama
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------

class BlockType(Enum):
    HEADING  = "heading"     # Büyük/kalın başlık
    SUBHEAD  = "subheading"  # Alt başlık
    BODY     = "body"        # Gövde metni
    CAPTION  = "caption"     # Küçük açıklama
    UNKNOWN  = "unknown"


@dataclass
class LayoutBox:
    """Pozisyon + metin + tip bilgisi olan tek bir metin kutusu."""
    text: str
    box: np.ndarray          # [4, 2]  polygon
    confidence: float = 0.0
    block_type: BlockType = BlockType.UNKNOWN
    column: int = 0          # hangi sütunda

    # --- konum özellikleri ---
    @property
    def x1(self) -> float: return float(np.min(self.box[:, 0]))
    @property
    def y1(self) -> float: return float(np.min(self.box[:, 1]))
    @property
    def x2(self) -> float: return float(np.max(self.box[:, 0]))
    @property
    def y2(self) -> float: return float(np.max(self.box[:, 1]))
    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def width(self) -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1


@dataclass
class LayoutBlock:
    """Bir veya daha fazla LayoutBox'tan oluşan mantıksal blok."""
    boxes: List[LayoutBox] = field(default_factory=list)
    block_type: BlockType = BlockType.BODY
    column: int = 0

    @property
    def text(self) -> str:
        return " ".join(b.text for b in self.boxes if b.text.strip())

    @property
    def y1(self) -> float:
        return min(b.y1 for b in self.boxes) if self.boxes else 0.0


@dataclass
class DocumentLayout:
    """Tüm belge için layout çıktısı."""
    blocks: List[LayoutBlock] = field(default_factory=list)
    num_columns: int = 1
    image_width: int = 0
    image_height: int = 0

    def to_structured_text(
        self,
        heading_suffix: str = "\n",
        paragraph_sep: str = "\n\n",
        column_sep: str = "\n\n--- --- ---\n\n"
    ) -> str:
        """
        Bloklardan düzenli metin üret.

        - Başlıklar tek satır + boşluk
        - Paragraflar çift satır sonu ile ayrılır
        - Sütun geçişlerinde ayırıcı
        """
        if not self.blocks:
            return ""

        parts: List[str] = []
        last_col = self.blocks[0].column

        for blk in self.blocks:
            if blk.column != last_col:
                parts.append(column_sep)
                last_col = blk.column

            txt = blk.text.strip()
            if not txt:
                continue

            if blk.block_type in (BlockType.HEADING, BlockType.SUBHEAD):
                parts.append(txt + heading_suffix)
            else:
                parts.append(txt)

        return paragraph_sep.join(p for p in parts if p not in ("", column_sep)).replace(
            paragraph_sep + column_sep, column_sep
        )


# ---------------------------------------------------------------------------
# Ana analizör
# ---------------------------------------------------------------------------

class LayoutAnalyzer:
    """
    Metin kutularından belge layout'u çıkarır.

    Adımlar:
        1. Font yüksekliğini tahmin et  →  başlık / gövde / açıklama sınıflandırması
        2. Projeksiyon profili          →  sütun sayısı ve sınırları
        3. Sütun ataması
        4. Z-düzeninde okuma sırası
        5. Mantıksal blok oluşturma
    """

    def __init__(
        self,
        # Başlık/gövde eşikleri (gövde ortalamasına oran)
        heading_height_ratio: float  = 1.5,
        subhead_height_ratio: float  = 1.25,
        caption_height_ratio: float  = 0.75,
        # Sütun tespiti
        column_gap_ratio: float      = 0.04,   # görüntü genişliğine göre min boşluk
        max_columns: int             = 4,
        # Blok birleştirme
        line_merge_gap_ratio: float  = 0.5,    # ortalama satır yüksekliğine oran
        paragraph_gap_ratio: float   = 1.2,
    ):
        self.heading_height_ratio  = heading_height_ratio
        self.subhead_height_ratio  = subhead_height_ratio
        self.caption_height_ratio  = caption_height_ratio
        self.column_gap_ratio      = column_gap_ratio
        self.max_columns           = max_columns
        self.line_merge_gap_ratio  = line_merge_gap_ratio
        self.paragraph_gap_ratio   = paragraph_gap_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text_boxes: list,            # TextBox veya TextLine gibi nesneler
        image_width: int  = 0,
        image_height: int = 0
    ) -> DocumentLayout:
        """
        Metin kutularından DocumentLayout üret.

        Args:
            text_boxes: pipeline.TextBox veya postprocessing.TextLine listesi
                        (box: [4,2] numpy, text: str, confidence: float)
            image_width:  görüntü genişliği (0 = kutulardan tahmin)
            image_height: görüntü yüksekliği

        Returns:
            DocumentLayout
        """
        if not text_boxes:
            return DocumentLayout(image_width=image_width, image_height=image_height)

        # Normalize giriş → LayoutBox listesi
        lboxes = self._to_layout_boxes(text_boxes)

        if image_width == 0:
            image_width = int(max(b.x2 for b in lboxes)) + 1

        # 1. Yükseklik dağılımı → başlık sınıflandırması
        lboxes = self._classify_by_height(lboxes)

        # 2. Sütun tespiti
        column_boundaries = self._detect_columns(lboxes, image_width)
        num_cols = len(column_boundaries)

        # 3. Her kutuya sütun ata
        lboxes = self._assign_columns(lboxes, column_boundaries)

        # 4. Z-düzeninde sırala
        lboxes = self._sort_reading_order(lboxes)

        # 5. Mantıksal bloklar oluştur
        blocks = self._build_blocks(lboxes)

        return DocumentLayout(
            blocks=blocks,
            num_columns=num_cols,
            image_width=image_width,
            image_height=image_height
        )

    # ------------------------------------------------------------------
    # Adım 1 – Yüksekliğe göre sınıflandırma
    # ------------------------------------------------------------------

    def _classify_by_height(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        heights = [b.height for b in boxes if b.height > 0]
        if not heights:
            return boxes

        median_h = float(np.median(heights))

        for b in boxes:
            if median_h == 0:
                b.block_type = BlockType.BODY
                continue

            ratio = b.height / median_h

            if ratio >= self.heading_height_ratio:
                b.block_type = BlockType.HEADING
            elif ratio >= self.subhead_height_ratio:
                b.block_type = BlockType.SUBHEAD
            elif ratio <= self.caption_height_ratio:
                b.block_type = BlockType.CAPTION
            else:
                b.block_type = BlockType.BODY

        return boxes

    # ------------------------------------------------------------------
    # Adım 2 – Sütun tespiti (yatay projeksiyon profili)
    # ------------------------------------------------------------------

    def _detect_columns(
        self, boxes: List[LayoutBox], image_width: int
    ) -> List[Tuple[float, float]]:
        """
        Yatay eksende 'boş koridor' arar.

        Returns:
            [(col_x1, col_x2), ...]   soldan sağa sıralı sütun aralıkları
        """
        if image_width <= 0:
            return [(0.0, 1e9)]

        # Her piksel sütununa toplam metin kapsamı
        profile = np.zeros(image_width, dtype=np.float32)
        for b in boxes:
            x1 = max(0, int(b.x1))
            x2 = min(image_width - 1, int(b.x2))
            if x2 > x1:
                profile[x1:x2 + 1] += 1.0

        # Smoothing
        kernel = np.ones(max(3, image_width // 100)) / max(3, image_width // 100)
        smooth = np.convolve(profile, kernel, mode='same')

        # Eşiğin altındaki koridorlar → sütun sınırı adayı
        min_gap_px = int(image_width * self.column_gap_ratio)
        gap_mask = smooth < 0.1

        # Sürekli boşlukları bul
        gaps: List[Tuple[int, int]] = []
        in_gap = False
        gap_start = 0
        margin_px = int(image_width * 0.05)  # Kenar bosluklarini yoksay (%5)

        for x, is_gap in enumerate(gap_mask):
            if is_gap and not in_gap:
                gap_start = x
                in_gap = True
            elif not is_gap and in_gap:
                # Kenarda baslamiyorsa ve yeterince genisse gercek sutun boslugu
                if (x - gap_start >= min_gap_px
                        and gap_start > margin_px
                        and x < image_width - margin_px):
                    gaps.append((gap_start, x - 1))
                in_gap = False

        # 1 sütun → sütun sınırı yok
        if not gaps or len(gaps) > self.max_columns - 1:
            return [(0.0, float(image_width))]

        # Sütun aralıklarını üret
        boundaries: List[Tuple[float, float]] = []
        prev = 0.0
        for g_start, g_end in gaps:
            boundaries.append((prev, float(g_start)))
            prev = float(g_end + 1)
        boundaries.append((prev, float(image_width)))

        return boundaries if len(boundaries) >= 2 else [(0.0, float(image_width))]

    # ------------------------------------------------------------------
    # Adım 3 – Sütun ataması
    # ------------------------------------------------------------------

    def _assign_columns(
        self,
        boxes: List[LayoutBox],
        boundaries: List[Tuple[float, float]]
    ) -> List[LayoutBox]:
        for b in boxes:
            best_col = 0
            best_overlap = -1.0
            for i, (c_x1, c_x2) in enumerate(boundaries):
                overlap = min(b.x2, c_x2) - max(b.x1, c_x1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = i
            b.column = best_col
        return boxes

    # ------------------------------------------------------------------
    # Adım 4 – Okuma sırası
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_reading_order(boxes: List[LayoutBox]) -> List[LayoutBox]:
        """Sütun öncelikli, sonra Y, sonra X sıralaması."""
        return sorted(boxes, key=lambda b: (b.column, b.y1, b.x1))

    # ------------------------------------------------------------------
    # Adım 5 – Mantıksal blok oluşturma
    # ------------------------------------------------------------------

    def _build_blocks(self, boxes: List[LayoutBox]) -> List[LayoutBlock]:
        """
        Aynı sütun içindeki yakın satırları birleştir.
        Başlık bulunca yeni blok başlat.
        """
        if not boxes:
            return []

        heights = [b.height for b in boxes if b.height > 0]
        median_h = float(np.median(heights)) if heights else 20.0
        line_gap   = median_h * self.line_merge_gap_ratio
        para_gap   = median_h * self.paragraph_gap_ratio

        blocks: List[LayoutBlock] = []
        current = LayoutBlock(
            boxes=[boxes[0]],
            block_type=boxes[0].block_type,
            column=boxes[0].column
        )

        for prev_box, cur_box in zip(boxes, boxes[1:]):
            # Farklı sütun veya başlık → her zaman yeni blok
            if cur_box.column != prev_box.column:
                blocks.append(current)
                current = LayoutBlock(
                    boxes=[cur_box],
                    block_type=cur_box.block_type,
                    column=cur_box.column
                )
                continue

            gap = cur_box.y1 - prev_box.y2
            is_heading = cur_box.block_type in (BlockType.HEADING, BlockType.SUBHEAD)

            if is_heading or gap > para_gap:
                blocks.append(current)
                current = LayoutBlock(
                    boxes=[cur_box],
                    block_type=cur_box.block_type,
                    column=cur_box.column
                )
            elif gap <= line_gap:
                # Aynı blok, satır birleştir
                current.boxes.append(cur_box)
            else:
                # Paragraf geçişi — yeni blok ama aynı tip
                blocks.append(current)
                current = LayoutBlock(
                    boxes=[cur_box],
                    block_type=cur_box.block_type,
                    column=cur_box.column
                )

        blocks.append(current)
        return [b for b in blocks if b.text.strip()]

    # ------------------------------------------------------------------
    # Yardımcı: giriş normalize
    # ------------------------------------------------------------------

    @staticmethod
    def _to_layout_boxes(text_boxes: list) -> List[LayoutBox]:
        result = []
        for tb in text_boxes:
            text       = getattr(tb, 'text', '')
            box        = getattr(tb, 'box', None)
            confidence = getattr(tb, 'confidence', 0.0)

            if box is None or len(box) == 0:
                continue

            result.append(LayoutBox(
                text=text,
                box=np.asarray(box, dtype=np.float32),
                confidence=float(confidence)
            ))
        return result