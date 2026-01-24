"""Egitim ornegi testi"""
import sys
sys.path.insert(0, '.')

import torch
import yaml
from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.model import CRNN
from ocr_engine.recognition.decoder import CTCDecoder
from training.dataset import RecognitionDataset

# Config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Vocab
vocab = Vocabulary()
print(f"Vocab size: {vocab.size}")

# Dataset
rec_cfg = config.get('recognition', {}).get('model', {})
dataset = RecognitionDataset(
    data_dir="DataSets/mnt/ramdisk/max/90kDICT32px",
    annotation_file="data/mjsynth_train.json",
    vocab=vocab,
    image_height=32,
    image_width=256,
    augmentor=None,
    synthetic_ratio=0.0
)

print(f"Dataset size: {len(dataset)}")

# Ilk 3 ornek
for i in range(3):
    sample = dataset[i]
    print(f"\n--- Sample {i} ---")
    print(f"Text: '{sample['text']}'")
    print(f"Label: {sample['label'].tolist()}")
    print(f"Label length: {sample['label_length']}")
    print(f"Image shape: {sample['image'].shape}")
    
    # Decode test
    decoded = vocab.decode(sample['label'].tolist())
    print(f"Decoded back: '{decoded}'")
    print(f"Match: {decoded == sample['text']}")
