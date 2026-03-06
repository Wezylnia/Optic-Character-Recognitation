"""Goruntu aci duzeltme (deskew)."""

import cv2
import numpy as np
from typing import Tuple, Optional


class Deskewer:
    """Goruntu egim acisini tespit edip duzeltir."""

    def __init__(self, max_angle: float = 45.0):
        self.max_angle = max_angle

    def deskew(self, image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
        if angle is None:
            angle = self.detect_angle(image)
        if abs(angle) > self.max_angle or abs(angle) < 0.5:
            return image.copy(), 0.0
        return self.rotate(image, angle), angle

    def detect_angle(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        angles = [a for a in [
            self._detect_angle_hough(gray),
            self._detect_angle_minrect(gray),
            self._detect_angle_projection(gray),
        ] if a is not None]
        return float(np.median(angles)) if angles else 0.0

    def _detect_angle_hough(self, gray: np.ndarray) -> Optional[float]:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return None
        angles = [
            np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
            for x1, y1, x2, y2 in lines[:, 0]
            if x2 - x1 != 0 and abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi) < self.max_angle
        ]
        return float(np.median(angles)) if angles else None

    def _detect_angle_minrect(self, gray: np.ndarray) -> Optional[float]:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 100:
            return None
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        return -angle

    def _detect_angle_projection(
        self,
        gray: np.ndarray,
        angle_range: Tuple[float, float] = (-15, 15),
        step: float = 0.5
    ) -> Optional[float]:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        best_angle, best_var = 0.0, 0.0
        for angle in np.arange(angle_range[0], angle_range[1] + step, step):
            var = np.var(np.sum(self.rotate(binary, angle, border_value=0), axis=1))
            if var > best_var:
                best_var, best_angle = var, angle
        return best_angle

    def rotate(self, image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        bv = (border_value,) * 3 if image.ndim == 3 else border_value
        return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=bv)