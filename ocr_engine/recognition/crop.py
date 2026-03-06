"""Recognition icin satir crop yardimcilari"""

import cv2
import numpy as np
from typing import List, Tuple


def split_line_to_words(
    gray: np.ndarray,
    box: np.ndarray,
    min_gap_ratio: float = 0.015,
    min_word_ratio: float = 0.025,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Tek satirlik grayscale crop'u dikey projeksiyon ile kelimelerine boler."""
    h, w = gray.shape[:2]
    if w == 0 or h == 0:
        return [(gray, box)]

    min_gap_px  = max(2, int(w * min_gap_ratio))
    min_word_px = max(4, int(w * min_word_ratio))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_col = (255 - binary).sum(axis=0).astype(np.float32)

    k = max(3, min(w // 20, 11))
    if k % 2 == 0:
        k += 1
    ink_smooth = cv2.GaussianBlur(
        ink_col.reshape(1, -1).astype(np.float32), (k, 1), 0
    ).flatten()

    max_ink = ink_smooth.max()
    if max_ink < 1.0:
        return [(gray, box)]

    is_gap = ink_smooth <= max_ink * 0.05

    segments: List[Tuple[int, int]] = []
    in_word, seg_start = False, 0
    for x in range(w):
        if not is_gap[x]:
            if not in_word:
                seg_start = x
                in_word = True
        else:
            if in_word:
                in_word = False
                if x - seg_start >= min_word_px:
                    segments.append((seg_start, x))
    if in_word and w - seg_start >= min_word_px:
        segments.append((seg_start, w))

    if len(segments) < 2:
        return [(gray, box)]

    merged: List[Tuple[int, int]] = [segments[0]]
    for s, e in segments[1:]:
        if s - merged[-1][1] < min_gap_px:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    if len(merged) < 2:
        return [(gray, box)]

    x_min   = int(np.min(box[:, 0]))
    y_min   = int(np.min(box[:, 1]))
    x_max   = int(np.max(box[:, 0]))
    y_max   = int(np.max(box[:, 1]))
    scale_x = max(x_max - x_min, 1) / w

    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for seg_s, seg_e in merged:
        word_gray = gray[:, seg_s:seg_e]
        wx1 = x_min + int(seg_s * scale_x)
        wx2 = x_min + int(seg_e * scale_x)
        word_poly = np.array(
            [[wx1, y_min], [wx2, y_min], [wx2, y_max], [wx1, y_max]],
            dtype=np.float32
        )
        result.append((word_gray, word_poly))

    return result


def compute_ctc_confidence(
    max_probs: np.ndarray,
    max_idx: np.ndarray,
    blank_idx: int = 0
) -> np.ndarray:
    """Blank ve tekrar pozisyonlari haric CTC guven skoru. [B, T] -> [B]"""
    B, T = max_probs.shape
    confidences = np.zeros(B, dtype=np.float32)
    for b in range(B):
        non_blank = []
        prev = -1
        for t in range(T):
            idx = int(max_idx[b, t])
            if idx != blank_idx and idx != prev:
                non_blank.append(float(max_probs[b, t]))
            prev = idx
        confidences[b] = float(np.mean(non_blank)) if non_blank else 0.0
    return confidences
