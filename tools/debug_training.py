"""Detayli debug"""
import sys
sys.path.insert(0, '.')

import torch
import yaml
from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.model import CRNN
from ocr_engine.recognition.decoder import CTCDecoder
from training.recognition_dataset import RecognitionDataset, collate_recognition
from torch.utils.data import DataLoader

print("="*60)
print("DEBUG RAPORU")
print("="*60)

# Config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Vocab
vocab = Vocabulary()
print(f"\n1. VOCABULARY:")
print(f"   vocab.size = {vocab.size}")
print(f"   vocab.num_classes = {getattr(vocab, 'num_classes', 'YOK!')}")
print(f"   vocab.blank_idx = {vocab.blank_idx}")

# Model
print(f"\n2. MODEL:")
model = CRNN(
    num_classes=vocab.size,  # vocab.num_classes degil!
    input_channels=1,
    hidden_size=256,
    num_layers=2,
    dropout=0.1,
    encoder_type='vgg'
)
print(f"   num_classes = {model.num_classes}")

# Test input
print(f"\n3. INPUT/OUTPUT SHAPES:")
test_input = torch.randn(2, 1, 32, 256)  # batch=2, channel=1, H=32, W=256
print(f"   Input shape: {test_input.shape}")

output = model(test_input)
print(f"   Output shape: {output.shape}")  # [seq_len, batch, num_classes]
print(f"   Sequence length: {output.shape[0]}")

# CTC constraint check
input_length = model.get_sequence_length(256)
print(f"\n4. CTC CONSTRAINT CHECK:")
print(f"   Model sequence length: {input_length}")
print(f"   Tipik label uzunlugu (5 haneli sayi): 5")
print(f"   input_length >= label_length? {input_length >= 5}")

# Dataset sample
print(f"\n5. DATASET SAMPLE:")
dataset = RecognitionDataset(
    data_dir="DataSets/mnt/ramdisk/max/90kDICT32px",
    annotation_file="data/mjsynth_train.json",
    vocab=vocab,
    image_height=32,
    image_width=256,
    augmentor=None,
    synthetic_ratio=0.0
)

sample = dataset[0]
print(f"   Text: '{sample['text']}'")
print(f"   Label: {sample['label'].tolist()}")
print(f"   Label length: {sample['label_length']}")
print(f"   Image shape: {sample['image'].shape}")

# Batch test
print(f"\n6. BATCH DECODE TEST:")
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_recognition)
batch = next(iter(loader))

images = batch['images']
labels = batch['labels']
label_lengths = batch['label_lengths']
texts = batch['texts']

print(f"   Batch images: {images.shape}")
print(f"   Batch labels: {labels.shape}")
print(f"   Label lengths: {label_lengths}")
print(f"   Texts: {texts}")

# Model forward
log_probs = model(images)
print(f"   Log probs shape: {log_probs.shape}")

# Decode
decoder = CTCDecoder(vocab)
preds = decoder.decode_greedy(log_probs)
print(f"   Predictions: {preds}")

# Random output testi
print(f"\n7. RANDOM OUTPUT ANALYSIS:")
print(f"   Log prob range: [{log_probs.min().item():.2f}, {log_probs.max().item():.2f}]")
print(f"   Log prob mean: {log_probs.mean().item():.4f}")

# En yuksek olasilikli class'lar
_, max_indices = log_probs.max(dim=2)
print(f"   Predicted indices (first sample): {max_indices[:, 0].tolist()[:20]}...")

# Blank orani
blank_count = (max_indices == vocab.blank_idx).sum().item()
total = max_indices.numel()
print(f"   Blank prediction ratio: {blank_count}/{total} = {blank_count/total*100:.1f}%")

print("\n" + "="*60)
