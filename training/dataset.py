"""
Veri seti ve veri yukleyici sinifları
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
import json
import random

from ocr_engine.recognition.vocab import Vocabulary


class SyntheticTextGenerator:
    """Sentetik metin gorseli uretici"""
    
    def __init__(
        self,
        vocab: Vocabulary,
        font_paths: Optional[List[str]] = None,
        font_sizes: Tuple[int, int] = (20, 48),
        image_height: int = 32,
        max_text_length: int = 25,
        bg_colors: List[Tuple[int, int, int]] = None,
        text_colors: List[Tuple[int, int, int]] = None
    ):
        """
        Args:
            vocab: Vocabulary nesnesi
            font_paths: Font dosya yollari
            font_sizes: Font boyut araligi
            image_height: Cikis gorsel yuksekligi
            max_text_length: Maksimum metin uzunlugu
            bg_colors: Arka plan renkleri
            text_colors: Metin renkleri
        """
        self.vocab = vocab
        self.font_sizes = font_sizes
        self.image_height = image_height
        self.max_text_length = max_text_length
        
        # Varsayilan renkler
        self.bg_colors = bg_colors or [
            (255, 255, 255),  # Beyaz
            (245, 245, 245),  # Acik gri
            (255, 255, 240),  # Krem
            (240, 248, 255),  # Acik mavi
        ]
        
        self.text_colors = text_colors or [
            (0, 0, 0),        # Siyah
            (50, 50, 50),     # Koyu gri
            (0, 0, 128),      # Koyu mavi
            (128, 0, 0),      # Koyu kirmizi
        ]
        
        # OpenCV font
        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        ]
    
    def generate(
        self,
        text: Optional[str] = None,
        min_length: int = 3,
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Sentetik metin gorseli uret
        
        Args:
            text: Kullanilacak metin (None ise rastgele)
            min_length: Minimum metin uzunlugu
            max_length: Maksimum metin uzunlugu
            
        Returns:
            (gorsel, metin) tuple
        """
        if max_length is None:
            max_length = self.max_text_length
        
        # Metin olustur
        if text is None:
            text = self._generate_random_text(min_length, max_length)
        
        # Font sec
        font = random.choice(self.fonts)
        font_scale = random.uniform(0.5, 1.5)
        thickness = random.randint(1, 2)
        
        # Renkler sec
        bg_color = random.choice(self.bg_colors)
        text_color = random.choice(self.text_colors)
        
        # Metin boyutunu hesapla
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Padding
        padding = random.randint(5, 15)
        
        # Gorsel boyutu
        img_h = text_h + baseline + 2 * padding
        img_w = text_w + 2 * padding
        
        # Gorsel olustur
        image = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)
        
        # Metin yaz
        x = padding
        y = padding + text_h
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
        
        # Hedef yukseklige boyutlandir
        scale = self.image_height / img_h
        new_w = int(img_w * scale)
        image = cv2.resize(image, (new_w, self.image_height))
        
        return image, text
    
    def _generate_random_text(
        self,
        min_length: int,
        max_length: int
    ) -> str:
        """Rastgele metin olustur"""
        chars = self.vocab.get_all_chars()
        
        # Bosluk haric karakterler
        non_space = [c for c in chars if c != ' ']
        
        length = random.randint(min_length, max_length)
        
        # Kelime bazli veya karakter bazli
        if random.random() < 0.7:
            # Kelime benzeri
            words = []
            current_len = 0
            while current_len < length:
                word_len = random.randint(2, min(8, length - current_len))
                word = ''.join(random.choices(non_space, k=word_len))
                words.append(word)
                current_len += word_len + 1
            return ' '.join(words)
        else:
            # Rastgele karakterler
            return ''.join(random.choices(chars, k=length))
    
    def generate_batch(
        self,
        batch_size: int,
        texts: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Batch gorsel uret"""
        images = []
        labels = []
        
        for i in range(batch_size):
            text = texts[i] if texts else None
            image, label = self.generate(text)
            images.append(image)
            labels.append(label)
        
        return images, labels


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
        if self.synthetic_ratio > 0:
            return int(base_len / (1 - self.synthetic_ratio))
        return max(base_len, 1000)  # Minimum 1000 ornek
    
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
                # Fallback to synthetic
                image, text = self.synthetic_generator.generate()
        
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


class DetectionDataset(Dataset):
    """Metin tespit veri seti"""
    
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        input_size: Tuple[int, int] = (640, 640),
        augmentor: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Gorsel klasoru
            annotation_file: Annotation dosyasi
            input_size: Model girdi boyutu (width, height)
            augmentor: Augmentation fonksiyonu
            max_samples: Maksimum ornek sayisi
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augmentor = augmentor
        
        # Veri yukle
        self.samples = self._load_annotations(annotation_file)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Annotation yukle"""
        samples = []
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_path = self.data_dir / item['image_path']
            boxes = [np.array(box, dtype=np.float32) for box in item['boxes']]
            
            samples.append({
                'image_path': str(image_path),
                'boxes': boxes
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Gorsel yukle
        image = cv2.imread(sample['image_path'])
        boxes = [box.copy() for box in sample['boxes']]
        
        # Augmentation
        if self.augmentor:
            image, boxes = self.augmentor(image, boxes)
        
        # Boyutlandir
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.input_size)
        
        # Kutulari olcekle
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        
        scaled_boxes = []
        for box in boxes:
            scaled = box.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y
            scaled_boxes.append(scaled)
        
        # Ground truth map'ler olustur
        prob_map, thresh_map, mask = self._generate_maps(scaled_boxes)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Tensor'e cevir
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0).float()
        thresh_tensor = torch.from_numpy(thresh_map).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'prob_map': prob_tensor,
            'thresh_map': thresh_tensor,
            'mask': mask_tensor
        }
    
    def _generate_maps(
        self,
        boxes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ground truth map'leri olustur"""
        h, w = self.input_size[1], self.input_size[0]
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        
        for box in boxes:
            # Probability map - kutunun icini doldur
            pts = box.astype(np.int32)
            cv2.fillPoly(prob_map, [pts], 1.0)
            
            # Threshold map - kenar bolgesi
            cv2.polylines(thresh_map, [pts], True, 0.5, 2)
        
        return prob_map, thresh_map, mask


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
    from .augmentation import RecognitionAugmentor
    
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
