"""Hizli decoder testi"""
import sys
sys.path.insert(0, '.')

import torch
from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.decoder import CTCDecoder

# Vocab
vocab = Vocabulary()
print(f"Vocab size: {vocab.size}")
print(f"Blank idx: {vocab.blank_idx}")
print(f"Sample chars: {vocab.chars[:20]}")

# Test decoder
decoder = CTCDecoder(vocab)

# Simule edilmis model ciktisi (5 karakter icin)
seq_len, batch, num_classes = 20, 1, vocab.size
log_probs = torch.randn(seq_len, batch, num_classes)

# "12345" icin fake probabilities
for t in range(5):
    char_idx = vocab.encode("12345"[t])[0]
    log_probs[t*3:t*3+2, 0, char_idx] = 10.0  # Yuksek prob

# Decode
preds = decoder.decode_greedy(log_probs)
print(f"\nDecoded: '{preds[0]}'")
print(f"Length: {len(preds[0])}")
