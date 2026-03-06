"""
Veri setleri — Recognition ve Detection.

RecognitionDataset   : JSON annotation dosyasından metin tanıma veri seti
DetectionDataset     : DBNet eğitimi için detection veri seti
collate_recognition  : DataLoader için collate fonksiyonu
collate_attention    : Attention decoder için SOS/EOS ekleyen collate fabrikası
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset

from ocr_engine.recognition.vocab import Vocabulary
from .synthetic import SyntheticTextGenerator


# ── Recognition ───────────────────────────────────────────────────────────────

class RecognitionDataset(Dataset):
    """
    JSON annotation dosyasından metin tanıma veri seti.

    Beklenen JSON formatı::

        [{"image_path": "...", "text": "..."}, ...]
    """

    def __init__(
        self,
        annotation_file: Optional[str] = None,
        data_dir: Optional[str] = None,
        vocab: Optional[Vocabulary] = None,
        image_height: int = 48,
        image_width: int = 256,
        augmentor: Optional[Callable] = None,
        synthetic_ratio: float = 0.0,
        max_samples: Optional[int] = None,
    ):
        self.data_dir      = Path(data_dir) if data_dir else None
        self.vocab         = vocab or Vocabulary()
        self.image_height  = image_height
        self.image_width   = image_width
        self.augmentor     = augmentor
        self.synthetic_ratio = synthetic_ratio

        self.samples: List[Dict] = []
        if annotation_file and Path(annotation_file).exists():
            self._load(annotation_file)
        if max_samples:
            self.samples = self.samples[:max_samples]

        self.synth = (
            SyntheticTextGenerator(vocab=self.vocab, image_height=image_height)
            if synthetic_ratio > 0
            else None
        )

    # ------------------------------------------------------------------
    def _load(self, annotation_file: str):
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            img = item.get("image_path", item.get("image"))
            txt = item.get("text", item.get("label"))
            if not img or txt is None:
                continue
            if self.data_dir and not Path(img).is_absolute():
                img = str(self.data_dir / img)
            self.samples.append({"image_path": str(img), "text": str(txt)})

    def __len__(self) -> int:
        base = len(self.samples)
        if self.synthetic_ratio >= 1.0:
            return max(base, 10_000)
        if self.synthetic_ratio > 0:
            return int(base / max(1e-9, 1.0 - self.synthetic_ratio))
        return base

    def __getitem__(self, idx: int) -> Dict:
        use_synth = self.synth is not None and (
            len(self.samples) == 0 or random.random() < self.synthetic_ratio
        )
        if use_synth:
            image, text = self.synth.generate()
        else:
            sample = self.samples[idx % len(self.samples)]
            image  = cv2.imread(sample["image_path"])
            text   = sample["text"]
            if image is None:
                if self.synth:
                    image, text = self.synth.generate()
                else:
                    return self.__getitem__((idx + 1) % max(len(self.samples), 1))

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.augmentor:
            image = self.augmentor(image)

        image  = self._resize_pad(image)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        label  = self.vocab.encode(text)
        return {
            "image":        tensor,
            "label":        torch.tensor(label, dtype=torch.long),
            "label_length": len(label),
            "text":         text,
        }

    def _resize_pad(self, image: np.ndarray) -> np.ndarray:
        h, w   = image.shape[:2]
        new_w  = min(int(w * self.image_height / h), self.image_width)
        resized = cv2.resize(image, (new_w, self.image_height))
        if new_w < self.image_width:
            pad = np.zeros((self.image_height, self.image_width), dtype=resized.dtype)
            pad[:, :new_w] = resized
            return pad
        return resized


def collate_recognition(batch: List[Dict]) -> Dict:
    """Değişken uzunluklu etiketleri padding ile birleştirir."""
    images        = torch.stack([b["image"] for b in batch])
    label_lengths = torch.tensor([b["label_length"] for b in batch])
    max_len       = int(label_lengths.max())
    padded        = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ll = b["label_length"]
        padded[i, :ll] = b["label"][:ll]
    return {
        "images":        images,
        "labels":        padded,
        "label_lengths": label_lengths,
        "texts":         [b["text"] for b in batch],
    }


def collate_attention(sos_idx: int, eos_idx: int, pad_idx: int = 0):
    """
    Attention decoder için collate fabrikası.

    Döndürülen fonksiyon, her etikete SOS ekler, EOS ile bitirir ve padding uygular.
    ``DataLoader(..., collate_fn=collate_attention(vocab.sos_idx, vocab.eos_idx))``
    şeklinde kullanılır.
    """
    def _collate(batch: List[Dict]) -> Dict:
        images = torch.stack([b["image"] for b in batch])
        texts  = [b["text"] for b in batch]
        seqs   = [[sos_idx] + b["label"][:b["label_length"]].tolist() + [eos_idx]
                  for b in batch]
        max_len = max(len(s) for s in seqs)
        padded  = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            lengths[i] = len(s)
        return {
            "images":         images,
            "targets":        padded,
            "target_lengths": lengths,
            "texts":          texts,
        }
    return _collate


# ── Detection ─────────────────────────────────────────────────────────────────

class DetectionDataset(Dataset):
    """
    DBNet eğitimi için detection veri seti.

    Beklenen JSON formatı::

        [{"image_path": "...", "boxes": [[[x,y], ...], ...]}, ...]
    """

    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        input_size: Tuple[int, int] = (640, 640),
        augmentor: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir   = Path(data_dir)
        self.input_size = input_size
        self.augmentor  = augmentor
        self.samples    = self._load(annotation_file)
        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load(self, annotation_file: str) -> List[Dict]:
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            {
                "image_path": str(self.data_dir / item["image_path"]),
                "boxes": [np.array(b, dtype=np.float32) for b in item["boxes"]],
            }
            for item in data
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s     = self.samples[idx]
        image = cv2.imread(s["image_path"])
        boxes = [b.copy() for b in s["boxes"]]

        if self.augmentor:
            image, boxes = self.augmentor(image, boxes)

        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.input_size)
        sx    = self.input_size[0] / orig_w
        sy    = self.input_size[1] / orig_h
        boxes = [b * np.array([sx, sy], dtype=np.float32) for b in boxes]

        prob_map, thresh_map, mask = _dbnet_maps(boxes, self.input_size)

        mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image.astype(np.float32) / 255.0 - mean) / std
        return {
            "image":      torch.from_numpy(image).permute(2, 0, 1).float(),
            "prob_map":   torch.from_numpy(prob_map).unsqueeze(0).float(),
            "thresh_map": torch.from_numpy(thresh_map).unsqueeze(0).float(),
            "mask":       torch.from_numpy(mask).unsqueeze(0).float(),
        }


def _dbnet_maps(
    boxes: List[np.ndarray],
    size: Tuple[int, int],
    shrink_ratio: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DBNet için prob / thresh / mask haritaları üretir.

    Orijinal pixel-bazlı döngü yerine cv2.distanceTransform kullanır —
    10-100x daha hızlı.
    """
    w, h        = size
    prob_map    = np.zeros((h, w), dtype=np.float32)
    thresh_map  = np.zeros((h, w), dtype=np.float32)
    mask        = np.ones((h, w),  dtype=np.float32)

    for box in boxes:
        try:
            poly = Polygon(box)
            if not poly.is_valid:
                poly = poly.buffer(0)
            area, perim = poly.area, poly.length
            if area < 1 or perim == 0:
                continue
        except Exception:
            continue

        d   = area * (1 - shrink_ratio ** 2) / perim
        pts = box.astype(np.int64).tolist()

        # ── Prob map: içe doğru küçültülmüş polygon ──────────────────
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrunk = pco.Execute(-d)
        if shrunk:
            cv2.fillPoly(prob_map, [np.array(shrunk[0], dtype=np.int32)], 1.0)

        # ── Threshold map: distance transform tabanlı ─────────────────
        pco2 = pyclipper.PyclipperOffset()
        pco2.AddPath(pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = pco2.Execute(d)
        if not expanded:
            continue

        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(outer, [np.array(expanded[0], dtype=np.int32)], 255)
        if shrunk:
            cv2.fillPoly(inner, [np.array(shrunk[0], dtype=np.int32)], 255)

        # Mesafe: inner sınırından itibaren
        dist     = cv2.distanceTransform(cv2.bitwise_not(inner), cv2.DIST_L2, 5)
        band     = outer > 0
        max_d    = dist[band].max() if band.any() else 1.0
        if max_d > 0:
            t_val = np.where(band, 1.0 - dist / max_d, 0.0)
            np.maximum(thresh_map, t_val, out=thresh_map)

    # Normalize [0.3, 0.7]
    thresh_map = np.clip(thresh_map * 0.4 + 0.3, 0.0, 1.0) * (thresh_map > 0)
    return prob_map, thresh_map, mask
