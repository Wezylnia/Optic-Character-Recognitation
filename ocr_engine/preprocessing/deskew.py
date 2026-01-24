"""
Goruntu aci duzeltme (deskew) islemleri
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class Deskewer:
    """Goruntu aci duzeltme sinifi"""
    
    def __init__(self, max_angle: float = 45.0):
        """
        Args:
            max_angle: Maksimum duzeltme acisi (derece)
        """
        self.max_angle = max_angle
    
    def deskew(
        self,
        image: np.ndarray,
        angle: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Gorseli duzeltir
        
        Args:
            image: Giris gorseli
            angle: Aci (None ise otomatik tespit)
            
        Returns:
            Duzeltilmis gorsel ve tespit edilen aci
        """
        if angle is None:
            angle = self.detect_angle(image)
        
        # Aci cok buyukse duzeltme
        if abs(angle) > self.max_angle:
            return image.copy(), 0.0
        
        # Aci cok kucukse (yaklasik duz)
        if abs(angle) < 0.5:
            return image.copy(), 0.0
        
        rotated = self.rotate(image, angle)
        return rotated, angle
    
    def detect_angle(self, image: np.ndarray) -> float:
        """
        Gorsel egim acisini tespit eder
        
        Args:
            image: Giris gorseli
            
        Returns:
            Egim acisi (derece)
        """
        # Gri tonlamaya cevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Birden fazla yontem dene ve en iyi sonucu sec
        angles = []
        
        # Yontem 1: Hough Lines
        angle_hough = self._detect_angle_hough(gray)
        if angle_hough is not None:
            angles.append(angle_hough)
        
        # Yontem 2: minAreaRect
        angle_rect = self._detect_angle_minrect(gray)
        if angle_rect is not None:
            angles.append(angle_rect)
        
        # Yontem 3: Projection profile
        angle_proj = self._detect_angle_projection(gray)
        if angle_proj is not None:
            angles.append(angle_proj)
        
        if not angles:
            return 0.0
        
        # Median al (outlier'lara karsi daha direncli)
        return float(np.median(angles))
    
    def _detect_angle_hough(self, gray: np.ndarray) -> Optional[float]:
        """Hough Lines ile aci tespiti"""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough Lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return None
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                if abs(angle) < self.max_angle:
                    angles.append(angle)
        
        if not angles:
            return None
        
        return float(np.median(angles))
    
    def _detect_angle_minrect(self, gray: np.ndarray) -> Optional[float]:
        """minAreaRect ile aci tespiti"""
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Koordinatlari bul
        coords = np.column_stack(np.where(binary > 0))
        
        if len(coords) < 100:
            return None
        
        # Minimum alan dikdortgeni
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        # Aciyi duzelt
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        return -angle  # Ters cevir (duzeltme icin)
    
    def _detect_angle_projection(
        self,
        gray: np.ndarray,
        angle_range: Tuple[float, float] = (-15, 15),
        step: float = 0.5
    ) -> Optional[float]:
        """Projection profile ile aci tespiti"""
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        best_angle = 0.0
        best_variance = 0.0
        
        for angle in np.arange(angle_range[0], angle_range[1] + step, step):
            # Dondur
            rotated = self.rotate(binary, angle, border_value=0)
            
            # Yatay projeksiyon profili
            projection = np.sum(rotated, axis=1)
            
            # Varyans hesapla (en yuksek varyans = en iyi hizalama)
            variance = np.var(projection)
            
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        return best_angle
    
    def rotate(
        self,
        image: np.ndarray,
        angle: float,
        border_value: int = 255
    ) -> np.ndarray:
        """
        Gorseli belirli bir aci kadar dondurur
        
        Args:
            image: Giris gorseli
            angle: Donme acisi (derece, saat yonunun tersine pozitif)
            border_value: Sinir dolgu degeri
            
        Returns:
            Dondurulmus gorsel
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotasyon matrisi
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Yeni boyutlari hesapla
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Matrisi ayarla
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        # Border degeri
        if len(image.shape) == 3:
            border = (border_value, border_value, border_value)
        else:
            border = border_value
        
        # Dondur
        rotated = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border
        )
        
        return rotated
    
    def auto_crop_rotated(self, image: np.ndarray) -> np.ndarray:
        """
        Dondurulmus gorseldeki bos alanlari kirpar
        
        Args:
            image: Dondurulmus gorsel
            
        Returns:
            Kirpilmis gorsel
        """
        # Gri tonlamaya cevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold (beyaz arka plan varsayimi)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Icerik olan bolgeyi bul
        coords = cv2.findNonZero(binary)
        
        if coords is None:
            return image
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Biraz padding ekle
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w].copy()
