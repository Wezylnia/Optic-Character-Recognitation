"""Vocabulary testi"""
import sys
sys.path.insert(0, '.')

from ocr_engine.recognition.vocab import Vocabulary

vocab = Vocabulary()

# Test encode/decode
test_text = "12345"
print(f"Original: '{test_text}'")

encoded = vocab.encode(test_text)
print(f"Encoded indices: {encoded}")

for idx in encoded:
    char = vocab.get_char(idx)
    print(f"  {idx} -> '{char}'")

decoded = vocab.decode(encoded)
print(f"Decoded: '{decoded}'")
print(f"Match: {decoded == test_text}")

# Blank test
print(f"\nBlank idx: {vocab.blank_idx}")
print(f"Blank char: '{vocab.get_char(vocab.blank_idx)}'")

# Test with blanks
with_blanks = [vocab.blank_idx, encoded[0], vocab.blank_idx, encoded[1], encoded[2], encoded[3], encoded[4]]
decoded2 = vocab.decode(with_blanks, remove_blank=True)
print(f"\nWith blanks decoded: '{decoded2}'")
