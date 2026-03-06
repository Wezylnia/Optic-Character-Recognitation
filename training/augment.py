"""
Augmentation — albumentations tabanlı.

RecognitionAugmentor : kırpılmış text görselleri için (grayscale)
DetectionAugmentor   : tam sayfa detection görüntüleri için (BGR)
"""

import cv2
import numpy as np
from typing import List, Tuple

import albumentations as A


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _apply(transform: A.Compose, image: np.ndarray) -> np.ndarray:
    """Albumentations transform'unu 2-D veya 3-D görsele uygular."""
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    if image.ndim == 2:                          # (H, W) → (H, W, 1)
        out = transform(image=image[:, :, None])["image"][:, :, 0]
    else:
        out = transform(image=image)["image"]
    return out


# ── Recognition ───────────────────────────────────────────────────────────────

def _build_recognition_transform() -> A.Compose:
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(25.0, 100.0), p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.Affine(shear=(-10, 10), scale=(0.85, 1.15), p=0.15),
        A.ElasticTransform(alpha=20, sigma=4, p=0.15),
        A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.1),
        A.InvertImg(p=0.05),
        A.CoarseDropout(
            max_holes=1, max_height=16, max_width=64,
            min_height=4, min_width=8, fill_value=128, p=0.12,
        ),
        A.ImageCompression(quality_lower=30, quality_upper=80, p=0.1),
    ])


class RecognitionAugmentor:
    """
    Kırpılmış metin görüntüleri için augmentation.
    dataset'te ``augmentor=RecognitionAugmentor()`` şeklinde kullanılır.
    """

    def __init__(self):
        self._tfm = _build_recognition_transform()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return _apply(self._tfm, image)


# ── Detection ─────────────────────────────────────────────────────────────────

def _build_detection_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


class DetectionAugmentor:
    """
    Detection eğitimi için: (image, boxes) → (image, boxes).

    Not: Geometrik dönüşümler (rotate, flip) box koordinatlarını etkilemez;
    sadece renk/fotometrik augmentation uygulanır. Dönüşüm gerektiren
    augmentation için albumentations'ın keypoints API'si kullanılabilir.
    """

    def __init__(self):
        self._tfm = _build_detection_transform()

    def __call__(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        augmented = self._tfm(image=image)
        return augmented["image"], boxes