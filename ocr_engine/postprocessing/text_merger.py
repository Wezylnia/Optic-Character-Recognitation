"""
Metin birlestirme modulu - satirlari ve paragraflari birlestirme
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextLine:
    """Metin satiri"""
    text: str
    box: np.ndarray  # [4, 2]
    confidence: float = 0.0
    
    @property
    def center_y(self) -> float:
        return np.mean(self.box[:, 1])
    
    @property
    def center_x(self) -> float:
        return np.mean(self.box[:, 0])
    
    @property
    def height(self) -> float:
        return np.max(self.box[:, 1]) - np.min(self.box[:, 1])
    
    @property
    def width(self) -> float:
        return np.max(self.box[:, 0]) - np.min(self.box[:, 0])
    
    @property
    def left(self) -> float:
        return np.min(self.box[:, 0])
    
    @property
    def right(self) -> float:
        return np.max(self.box[:, 0])
    
    @property
    def top(self) -> float:
        return np.min(self.box[:, 1])
    
    @property
    def bottom(self) -> float:
        return np.max(self.box[:, 1])


class TextMerger:
    """Metin kutularini birlestirme sinifi"""
    
    def __init__(
        self,
        line_threshold: float = 10.0,
        paragraph_threshold: float = 20.0,
        word_spacing_ratio: float = 1.5
    ):
        """
        Args:
            line_threshold: Ayni satir icin Y toleransi (piksel)
            paragraph_threshold: Ayni paragraf icin Y boslugu (piksel)
            word_spacing_ratio: Kelime araligi orani (karakter genisligine gore)
        """
        self.line_threshold = line_threshold
        self.paragraph_threshold = paragraph_threshold
        self.word_spacing_ratio = word_spacing_ratio
    
    def merge_to_lines(
        self,
        text_boxes: List[TextLine]
    ) -> List[TextLine]:
        """
        Metin kutularini satirlara birlestir
        
        Args:
            text_boxes: Metin kutusu listesi
            
        Returns:
            Birlestirilmis satir listesi
        """
        if not text_boxes:
            return []
        
        # Y koordinatina gore grupla
        lines = self._group_by_line(text_boxes)
        
        # Her satiri birlestir
        merged_lines = []
        for line_boxes in lines:
            merged = self._merge_line(line_boxes)
            merged_lines.append(merged)
        
        return merged_lines
    
    def merge_to_paragraphs(
        self,
        text_boxes: List[TextLine]
    ) -> List[str]:
        """
        Metin kutularini paragraflara birlestir
        
        Args:
            text_boxes: Metin kutusu listesi
            
        Returns:
            Paragraf listesi
        """
        # Oncelikle satirlara birlestir
        lines = self.merge_to_lines(text_boxes)
        
        if not lines:
            return []
        
        # Paragraflara grupla
        paragraphs = []
        current_paragraph = [lines[0].text]
        last_bottom = lines[0].bottom
        
        for line in lines[1:]:
            gap = line.top - last_bottom
            
            if gap > self.paragraph_threshold:
                # Yeni paragraf
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [line.text]
            else:
                # Ayni paragraf
                current_paragraph.append(line.text)
            
            last_bottom = line.bottom
        
        # Son paragraf
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def merge_to_text(
        self,
        text_boxes: List[TextLine],
        preserve_lines: bool = True
    ) -> str:
        """
        Tum metin kutularini tek metne birlestir
        
        Args:
            text_boxes: Metin kutusu listesi
            preserve_lines: Satir sonlarini koru
            
        Returns:
            Birlestirilmis metin
        """
        if preserve_lines:
            lines = self.merge_to_lines(text_boxes)
            return '\n'.join([line.text for line in lines])
        else:
            paragraphs = self.merge_to_paragraphs(text_boxes)
            return '\n\n'.join(paragraphs)
    
    def _group_by_line(
        self,
        text_boxes: List[TextLine]
    ) -> List[List[TextLine]]:
        """Kutuları satırlara grupla"""
        if not text_boxes:
            return []
        
        # Y koordinatina gore sirala
        sorted_boxes = sorted(text_boxes, key=lambda x: x.center_y)
        
        lines = []
        current_line = [sorted_boxes[0]]
        current_y = sorted_boxes[0].center_y
        
        for box in sorted_boxes[1:]:
            # Ayni satirda mi?
            if abs(box.center_y - current_y) <= self.line_threshold:
                current_line.append(box)
            else:
                # Yeni satir
                lines.append(current_line)
                current_line = [box]
                current_y = box.center_y
        
        # Son satir
        if current_line:
            lines.append(current_line)
        
        # Her satiri X koordinatina gore sirala
        for line in lines:
            line.sort(key=lambda x: x.left)
        
        return lines
    
    def _merge_line(self, boxes: List[TextLine]) -> TextLine:
        """Bir satirdaki kutuları birlestir"""
        if len(boxes) == 1:
            return boxes[0]
        
        # X koordinatina gore sirali olmali
        boxes = sorted(boxes, key=lambda x: x.left)
        
        # Metinleri birlestir
        texts = []
        last_right = boxes[0].left
        avg_char_width = sum(b.width / max(len(b.text), 1) for b in boxes) / len(boxes)
        
        for box in boxes:
            gap = box.left - last_right
            
            # Bosluk ekle
            if gap > avg_char_width * self.word_spacing_ratio:
                texts.append(' ')
            
            texts.append(box.text)
            last_right = box.right
        
        merged_text = ''.join(texts)
        
        # Birlesik kutu
        all_points = np.vstack([b.box for b in boxes])
        merged_box = np.array([
            [np.min(all_points[:, 0]), np.min(all_points[:, 1])],
            [np.max(all_points[:, 0]), np.min(all_points[:, 1])],
            [np.max(all_points[:, 0]), np.max(all_points[:, 1])],
            [np.min(all_points[:, 0]), np.max(all_points[:, 1])]
        ])
        
        # Ortalama guven
        avg_confidence = sum(b.confidence for b in boxes) / len(boxes)
        
        return TextLine(
            text=merged_text,
            box=merged_box,
            confidence=avg_confidence
        )


class TableDetector:
    """Basit tablo algilama"""
    
    def __init__(
        self,
        min_rows: int = 2,
        min_cols: int = 2,
        alignment_threshold: float = 10.0
    ):
        """
        Args:
            min_rows: Minimum satir sayisi
            min_cols: Minimum sutun sayisi
            alignment_threshold: Hizalama toleransi
        """
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.alignment_threshold = alignment_threshold
    
    def detect_table(
        self,
        text_boxes: List[TextLine]
    ) -> Optional[List[List[str]]]:
        """
        Tablo yapisini tespit et
        
        Args:
            text_boxes: Metin kutusu listesi
            
        Returns:
            Tablo (2D liste) veya None
        """
        if len(text_boxes) < self.min_rows * self.min_cols:
            return None
        
        # X ve Y pozisyonlarini cluster'la
        x_positions = [box.center_x for box in text_boxes]
        y_positions = [box.center_y for box in text_boxes]
        
        # Basit clustering
        x_clusters = self._cluster_positions(x_positions)
        y_clusters = self._cluster_positions(y_positions)
        
        if len(x_clusters) < self.min_cols or len(y_clusters) < self.min_rows:
            return None
        
        # Tablo olustur
        table = [['' for _ in range(len(x_clusters))] for _ in range(len(y_clusters))]
        
        for box in text_boxes:
            row = self._find_cluster(box.center_y, y_clusters)
            col = self._find_cluster(box.center_x, x_clusters)
            
            if row is not None and col is not None:
                if table[row][col]:
                    table[row][col] += ' ' + box.text
                else:
                    table[row][col] = box.text
        
        return table
    
    def _cluster_positions(self, positions: List[float]) -> List[float]:
        """Pozisyonlari cluster'la"""
        if not positions:
            return []
        
        sorted_pos = sorted(set(positions))
        clusters = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - clusters[-1] > self.alignment_threshold:
                clusters.append(pos)
        
        return clusters
    
    def _find_cluster(
        self,
        value: float,
        clusters: List[float]
    ) -> Optional[int]:
        """Degerin hangi cluster'a ait oldugunu bul"""
        for i, center in enumerate(clusters):
            if abs(value - center) <= self.alignment_threshold:
                return i
        return None
    
    def table_to_text(
        self,
        table: List[List[str]],
        separator: str = '\t'
    ) -> str:
        """Tabloyu metne donustur"""
        lines = []
        for row in table:
            lines.append(separator.join(row))
        return '\n'.join(lines)
