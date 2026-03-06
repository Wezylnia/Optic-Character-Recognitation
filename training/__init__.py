"""
Training modülü - Temiz ve kütüphane tabanlı uygulama.

Modüller:
  synthetic  -> SyntheticTextGenerator
  augment    -> RecognitionAugmentor, DetectionAugmentor  (albumentations)
  dataset    -> RecognitionDataset, DetectionDataset,
                collate_recognition, collate_attention
  trainer    -> RecognitionTrainer
  train      -> CLI giriş noktası  (python training/train.py --help)
"""

from .augment import DetectionAugmentor, RecognitionAugmentor
from .dataset import (
    DetectionDataset,
    RecognitionDataset,
    collate_attention,
    collate_recognition,
)
from .trainer import RecognitionTrainer