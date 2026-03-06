"""Goruntu gurultu giderme."""

import cv2
import numpy as np


class Denoiser:
    """Bilateral / Gaussian / NLMeans / Median filtre ile gurultu giderme."""

    def __init__(self, method: str = "bilateral", strength: int = 10):
        self.method = method
        self.strength = strength

    def denoise(self, image: np.ndarray) -> np.ndarray:
        fn = getattr(self, f'_{self.method}_filter', None)
        if fn is None:
            raise ValueError(f"Bilinmeyen denoise yontemi: {self.method}")
        return fn(image)

    def _bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        d = max(5, self.strength // 2)
        s = self.strength * 7.5
        return cv2.bilateralFilter(image, d, s, s)

    def _gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        k = max(3, self.strength) | 1
        return cv2.GaussianBlur(image, (k, k), 0)

    def _nlmeans_filter(self, image: np.ndarray) -> np.ndarray:
        h = self.strength
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, h, 7, 21)

    def _median_filter(self, image: np.ndarray) -> np.ndarray:
        k = max(3, self.strength) | 1
        return cv2.medianBlur(image, k)