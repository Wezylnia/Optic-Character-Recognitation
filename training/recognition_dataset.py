"""
Metin tanima veri seti, collate fonksiyonu ve DataLoader yaratici
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable
from pathlib import Path
import json
import random

from ocr_engine.recognition.vocab import Vocabulary
from .synthetic import SyntheticTextGenerator


class RecognitionDataset(Dataset):
    """Metin tanima veri seti"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        annotation_file: Optional[str] = None,
        vocab: Optional[Vocabulary] = None,
        image_height: int = 32,
        image_width: int = 256,
        augmentor: Optional[Callable] = None,
        synthetic_ratio: float = 0.0,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Gorsel klasoru
            annotation_file: Annotation dosyasi (JSON)
            vocab: Vocabulary nesnesi
            image_height: Gorsel yuksekligi
            image_width: Gorsel genisligi
            augmentor: Augmentation fonksiyonu
            synthetic_ratio: Sentetik veri orani (0-1)
            max_samples: Maksimum ornek sayisi
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.vocab = vocab or Vocabulary()
        self.image_height = image_height
        self.image_width = image_width
        self.augmentor = augmentor
        self.synthetic_ratio = synthetic_ratio
        
        # Veri yukle
        self.samples = []
        
        if annotation_file and Path(annotation_file).exists():
            self._load_annotations(annotation_file)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Sentetik uretici
        if synthetic_ratio > 0:
            self.synthetic_generator = SyntheticTextGenerator(
                vocab=self.vocab,
                image_height=image_height
            )
        else:
            self.synthetic_generator = None
    
    def _load_annotations(self, annotation_file: str):
        """Annotation dosyasini yukle"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_path = item.get('image_path', item.get('image'))
            text = item.get('text', item.get('label'))
            
            if image_path and text:
                if self.data_dir:
                    image_path = self.data_dir / image_path
                self.samples.append({
                    'image_path': str(image_path),
                    'text': text
                })
    
    def __len__(self) -> int:
        base_len = len(self.samples)
        if self.synthetic_ratio >= 1.0:
            # Tamamen sentetik: sabit buyuk bir epoch uzunlugu kullan
            return max(base_len, 10000)
        if self.synthetic_ratio > 0:
            # Inflation: sentetik orani hesaba katarak toplam boyutu genislet
            return int(base_len / max(1e-9, 1 - self.synthetic_ratio))
        return base_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sentetik mi gercek mi
        use_synthetic = (
            self.synthetic_generator is not None and
            (len(self.samples) == 0 or random.random() < self.synthetic_ratio)
        )
        
        if use_synthetic:
            image, text = self.synthetic_generator.generate()
        else:
            sample = self.samples[idx % len(self.samples)]
            image = cv2.imread(sample['image_path'])
            text = sample['text']
            
            if image is None:
                # Fallback: sentetik uretici varsa kullan, yoksa bir sonraki ornege gec
                if self.synthetic_generator is not None:
                    image, text = self.synthetic_generator.generate()
                else:
                    return self.__getitem__((idx + 1) % max(len(self.samples), 1))
        
        # Grayscale'e cevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Augmentation
        if self.augmentor:
            image = self.augmentor(image)
        
        # Boyutlandir
        image = self._resize_and_pad(image)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Tensor'e cevir
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        
        # Label encode
        label = self.vocab.encode(text)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'label_length': len(label),
            'text': text
        }
    
    def _resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """Boyutlandir ve padding ekle"""
        h, w = image.shape[:2]
        
        # Yukseklige gore olcekle
        scale = self.image_height / h
        new_w = int(w * scale)
        
        image = cv2.resize(image, (new_w, self.image_height))
        
        # Genislik kontrolu
        if new_w > self.image_width:
            image = cv2.resize(image, (self.image_width, self.image_height))
        elif new_w < self.image_width:
            # Padding
            padded = np.zeros((self.image_height, self.image_width), dtype=image.dtype)
            padded[:, :new_w] = image
            image = padded
        
        return image


def collate_recognition(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Recognition batch collate fonksiyonu"""
    images = torch.stack([item['image'] for item in batch])
    
    # Label'lari padding ile birlestir
    labels = [item['label'] for item in batch]
    label_lengths = torch.tensor([item['label_length'] for item in batch])
    
    # Max uzunluga padding
    max_len = max(len(l) for l in labels)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    return {
        'images': images,
        'labels': padded_labels,
        'label_lengths': label_lengths,
        'texts': [item['text'] for item in batch]
    }


def create_recognition_dataloader(
    data_dir: Optional[str] = None,
    annotation_file: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    synthetic_ratio: float = 0.5
) -> DataLoader:
    """Recognition DataLoader olustur"""
    from .augmentation_recognition import RecognitionAugmentor
    
    augmentor = RecognitionAugmentor() if augment else None
    
    dataset = RecognitionDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        augmentor=augmentor,
        synthetic_ratio=synthetic_ratio
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_recognition,
        pin_memory=True
    )
