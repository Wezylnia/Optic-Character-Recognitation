"""
Sentetik metin gorseli uretici
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import random

from ocr_engine.recognition.vocab import Vocabulary


class SyntheticTextGenerator:
    """Sentetik metin gorseli uretici"""
    
    def __init__(
        self,
        vocab: Vocabulary,
        font_paths: Optional[List[str]] = None,
        font_sizes: Tuple[int, int] = (20, 48),
        image_height: int = 32,
        max_text_length: int = 25,
        bg_colors: List[Tuple[int, int, int]] = None,
        text_colors: List[Tuple[int, int, int]] = None
    ):
        """
        Args:
            vocab: Vocabulary nesnesi
            font_paths: Font dosya yollari
            font_sizes: Font boyut araligi
            image_height: Cikis gorsel yuksekligi
            max_text_length: Maksimum metin uzunlugu
            bg_colors: Arka plan renkleri
            text_colors: Metin renkleri
        """
        self.vocab = vocab
        self.font_sizes = font_sizes
        self.image_height = image_height
        self.max_text_length = max_text_length
        
        # Varsayilan renkler
        self.bg_colors = bg_colors or [
            (255, 255, 255),  # Beyaz
            (245, 245, 245),  # Acik gri
            (255, 255, 240),  # Krem
            (240, 248, 255),  # Acik mavi
        ]
        
        self.text_colors = text_colors or [
            (0, 0, 0),        # Siyah
            (50, 50, 50),     # Koyu gri
            (0, 0, 128),      # Koyu mavi
            (128, 0, 0),      # Koyu kirmizi
        ]
        
        # OpenCV font
        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        ]
    
    def generate(
        self,
        text: Optional[str] = None,
        min_length: int = 3,
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Sentetik metin gorseli uret
        
        Args:
            text: Kullanilacak metin (None ise rastgele)
            min_length: Minimum metin uzunlugu
            max_length: Maksimum metin uzunlugu
            
        Returns:
            (gorsel, metin) tuple
        """
        if max_length is None:
            max_length = self.max_text_length
        
        # Metin olustur
        if text is None:
            text = self._generate_random_text(min_length, max_length)
        
        # Font sec
        font = random.choice(self.fonts)
        font_scale = random.uniform(0.5, 1.5)
        thickness = random.randint(1, 2)
        
        # Renkler sec
        bg_color = random.choice(self.bg_colors)
        text_color = random.choice(self.text_colors)
        
        # Metin boyutunu hesapla
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Padding
        padding = random.randint(5, 15)
        
        # Gorsel boyutu
        img_h = text_h + baseline + 2 * padding
        img_w = text_w + 2 * padding
        
        # Gorsel olustur
        image = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)
        
        # Metin yaz
        x = padding
        y = padding + text_h
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
        
        # Hedef yukseklige boyutlandir
        scale = self.image_height / img_h
        new_w = int(img_w * scale)
        image = cv2.resize(image, (new_w, self.image_height))
        
        return image, text
    
    def _generate_random_text(
        self,
        min_length: int,
        max_length: int
    ) -> str:
        """Rastgele metin olustur"""
        chars = self.vocab.get_all_chars()
        
        # Bosluk haric karakterler
        non_space = [c for c in chars if c != ' ']
        
        length = random.randint(min_length, max_length)
        
        # Kelime bazli veya karakter bazli
        if random.random() < 0.7:
            # Kelime benzeli
            words = []
            current_len = 0
            while current_len < length:
                word_len = random.randint(2, min(8, length - current_len))
                word = ''.join(random.choices(non_space, k=word_len))
                words.append(word)
                current_len += word_len + 1
            return ' '.join(words)
        else:
            # Rastgele karakterler
            return ''.join(random.choices(chars, k=length))
    
    def generate_batch(
        self,
        batch_size: int,
        texts: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Batch gorsel uret"""
        images = []
        labels = []
        
        for i in range(batch_size):
            text = texts[i] if texts else None
            image, label = self.generate(text)
            images.append(image)
            labels.append(label)
        
        return images, labels
