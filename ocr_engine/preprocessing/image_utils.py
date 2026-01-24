"""
Temel goruntu isleme yardimci fonksiyonlari
"""

import cv2
import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path


class ImageProcessor:
    """Temel goruntu isleme sinifi"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 960),
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        """
        Args:
            target_size: Maksimum genislik ve yukseklik (width, height)
            normalize_mean: Normalizasyon ortalama degerleri (RGB)
            normalize_std: Normalizasyon standart sapma degerleri (RGB)
        """
        self.target_size = target_size
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Gorsel dosyasini yukler
        
        Args:
            image_path: Gorsel dosya yolu
            
        Returns:
            BGR formatinda numpy array
        """
        image_path = str(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Gorsel yuklenemedi: {image_path}")
        
        return image
    
    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """BGR'den RGB'ye donusturur"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def to_bgr(self, image: np.ndarray) -> np.ndarray:
        """RGB'den BGR'ye donusturur"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Gri tonlamaya donusturur"""
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def resize_with_aspect_ratio(
        self,
        image: np.ndarray,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        En-boy oranini koruyarak boyutlandirir
        
        Args:
            image: Giris gorseli
            max_width: Maksimum genislik
            max_height: Maksimum yukseklik
            
        Returns:
            Boyutlandirilmis gorsel ve olcekleme faktoru
        """
        if max_width is None:
            max_width = self.target_size[0]
        if max_height is None:
            max_height = self.target_size[1]
        
        h, w = image.shape[:2]
        
        # Olcekleme faktorunu hesapla
        scale_w = max_width / w if w > max_width else 1.0
        scale_h = max_height / h if h > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale
        
        return image.copy(), 1.0
    
    def resize_to_height(
        self,
        image: np.ndarray,
        target_height: int = 32
    ) -> Tuple[np.ndarray, float]:
        """
        Belirli bir yukseklige boyutlandirir (genislik orantili)
        
        Args:
            image: Giris gorseli
            target_height: Hedef yukseklik
            
        Returns:
            Boyutlandirilmis gorsel ve olcekleme faktoru
        """
        h, w = image.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    def pad_to_width(
        self,
        image: np.ndarray,
        target_width: int,
        pad_value: int = 0
    ) -> np.ndarray:
        """
        Gorseli belirli bir genislige padding ekleyerek genisletir
        
        Args:
            image: Giris gorseli
            target_width: Hedef genislik
            pad_value: Padding degeri
            
        Returns:
            Padding eklenmis gorsel
        """
        h, w = image.shape[:2]
        
        if w >= target_width:
            return image[:, :target_width]
        
        pad_width = target_width - w
        
        if len(image.shape) == 2:
            padded = np.full((h, target_width), pad_value, dtype=image.dtype)
            padded[:, :w] = image
        else:
            c = image.shape[2]
            padded = np.full((h, target_width, c), pad_value, dtype=image.dtype)
            padded[:, :w, :] = image
        
        return padded
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Gorseli normalize eder (ImageNet standardlari)
        
        Args:
            image: RGB formatinda gorsel (0-255)
            
        Returns:
            Normalize edilmis gorsel
        """
        image = image.astype(np.float32) / 255.0
        image = (image - self.normalize_mean) / self.normalize_std
        return image
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize edilmis gorseli geri donusturur
        
        Args:
            image: Normalize edilmis gorsel
            
        Returns:
            0-255 araliginda gorsel
        """
        image = image * self.normalize_std + self.normalize_mean
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image
    
    def crop(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        padding: int = 0
    ) -> np.ndarray:
        """
        Gorselden bir bolge keser
        
        Args:
            image: Giris gorseli
            x1, y1: Sol ust kosesi
            x2, y2: Sag alt kosesi
            padding: Ekstra padding miktari
            
        Returns:
            Kesilmis bolge
        """
        h, w = image.shape[:2]
        
        # Padding ekle ve sinirlari kontrol et
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2].copy()
    
    def crop_polygon(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        target_height: int = 32
    ) -> np.ndarray:
        """
        Dortgen olmayan bir bolgeyi keser ve duzlestirir
        
        Args:
            image: Giris gorseli
            polygon: 4 noktali poligon koordinatlari [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            target_height: Hedef yukseklik
            
        Returns:
            Duzlestirilmis gorsel
        """
        polygon = np.array(polygon, dtype=np.float32)
        
        # Bounding rect hesapla
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        if width < height:
            width, height = height, width
        
        # Perspektif donusumu
        src_pts = polygon.astype(np.float32)
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        # Yukseklige gore boyutlandir
        if height > 0:
            scale = target_height / height
            new_width = max(1, int(width * scale))
            warped = cv2.resize(warped, (new_width, target_height))
        
        return warped
    
    def adjust_brightness_contrast(
        self,
        image: np.ndarray,
        brightness: float = 0,
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Parlaklik ve kontrast ayarla
        
        Args:
            image: Giris gorseli
            brightness: Parlaklik degeri (-127, 127)
            contrast: Kontrast carpani (0-3)
            
        Returns:
            Ayarlanmis gorsel
        """
        adjusted = image.astype(np.float32)
        adjusted = contrast * adjusted + brightness
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Histogram esitleme uygular
        
        Args:
            image: Gri tonlamali gorsel
            
        Returns:
            Histogram esitlenmis gorsel
        """
        if len(image.shape) == 3:
            # Renkli gorsel icin LAB uzayinda esitle
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
