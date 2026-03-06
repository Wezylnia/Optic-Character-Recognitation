"""Temel goruntu isleme yardimci fonksiyonlari"""

import cv2
import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path


class ImageProcessor:
    """Goruntu yukleme, olcekleme ve normalizasyon."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 960),
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.target_size = target_size
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std  = np.array(normalize_std,  dtype=np.float32)

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Gorsel yuklenemedi: {image_path}")
        return image

    def resize_with_aspect_ratio(
        self,
        image: np.ndarray,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """En-boy oranini koruyarak max boyuta indirir (buyutmez)."""
        max_width  = max_width  or self.target_size[0]
        max_height = max_height or self.target_size[1]
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
        return image.copy(), 1.0

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """ImageNet normalizasyonu (RGB float32)."""
        image = image.astype(np.float32) / 255.0
        return (image - self.normalize_mean) / self.normalize_std