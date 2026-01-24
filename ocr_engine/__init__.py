"""
OCR Engine - Python tabanli optik karakter tanima motoru

Bu modul, gorsellerden metin cikarma icin gerekli tum bilesenleri icerir:
- Preprocessing: Goruntu on isleme
- Detection: Metin bolge tespiti (DBNet)
- Recognition: Metin tanima (CRNN)
- Postprocessing: Yazim duzeltme ve birlestirme
"""

from .pipeline import OCRPipeline

__version__ = "1.0.0"
__all__ = ["OCRPipeline"]
