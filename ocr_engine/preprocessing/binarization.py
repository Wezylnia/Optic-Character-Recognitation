"""Goruntu binarization islemi."""

import cv2
import numpy as np


class Binarizer:
    """Farkli esikleme yontemleri ile goruntu binarizasyonu."""

    def __init__(self, method: str = "adaptive", block_size: int = 11, c: int = 2):
        self.method = method
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c

    def binarize(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        fn = getattr(self, f'_{self.method}_threshold', None)
        if fn is None:
            raise ValueError(f"Bilinmeyen binarization yontemi: {self.method}")
        return fn(gray)

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.block_size, self.c
        )

    def _otsu_threshold(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _sauvola_threshold(self, gray: np.ndarray, k: float = 0.5, r: float = 128) -> np.ndarray:
        """T(x,y) = mean * (1 + k * ((std / r) - 1))"""
        mean  = cv2.blur(gray.astype(np.float64), (self.block_size, self.block_size))
        msq   = cv2.blur(gray.astype(np.float64) ** 2, (self.block_size, self.block_size))
        std   = np.sqrt(np.maximum(msq - mean ** 2, 0))
        thresh = mean * (1 + k * ((std / r) - 1))
        binary = np.zeros_like(gray)
        binary[gray > thresh] = 255
        return binary.astype(np.uint8)

    def _niblack_threshold(self, gray: np.ndarray, k: float = -0.2) -> np.ndarray:
        """T(x,y) = mean + k * std"""
        mean  = cv2.blur(gray.astype(np.float64), (self.block_size, self.block_size))
        msq   = cv2.blur(gray.astype(np.float64) ** 2, (self.block_size, self.block_size))
        std   = np.sqrt(np.maximum(msq - mean ** 2, 0))
        thresh = mean + k * std
        binary = np.zeros_like(gray)
        binary[gray > thresh] = 255
        return binary.astype(np.uint8)