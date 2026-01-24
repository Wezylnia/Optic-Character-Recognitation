"""
Detection modulu - Metin bolge tespiti (DBNet)
"""

from .model import DBNet
from .postprocess import DBPostProcessor

__all__ = ["DBNet", "DBPostProcessor"]
