"""
PaddleOCR detection subprocess worker.

Bu script ayri bir Python process'te calisir — PyTorch ile ayni
process'te olmadigi icin CUDA DLL catismasi (_gpuDeviceProperties
already registered) yasanmaz.

Protokol (stdin/stdout):
  Istek : 4 bayt uzunluk (big-endian uint32) + numpy array bytes (pickle)
  Cevap : JSON satiri  [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] listesi + '\n'
  Kapat : 4 bayt 0x00000000
"""
import os
import sys
import json
import pickle
import struct
import traceback

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

import numpy as np


def load_detector():
    from paddleocr import TextDetection
    det = TextDetection(thresh=0.5, box_thresh=0.6, unclip_ratio=1.5)
    return det


def run():
    # Tum PaddleOCR/paddle loglarini stderr'e yonlendir
    # boylece stdout temiz kalir (protokol icin)
    import io
    old_stdout = sys.stdout
    sys.stdout = sys.stderr   # paddle baslangic loglarini stderr'e gonder

    det = load_detector()

    sys.stdout = old_stdout   # stdout'u geri al

    # Hazir sinyali gonder
    sys.stdout.write('READY\n')
    sys.stdout.flush()

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        # 4 bayt: payload uzunlugu
        header = stdin.read(4)
        if not header or len(header) < 4:
            break
        length = struct.unpack('>I', header)[0]
        if length == 0:
            break  # shutdown

        data = b''
        while len(data) < length:
            chunk = stdin.read(length - len(data))
            if not chunk:
                break
            data += chunk

        try:
            image = pickle.loads(data)
            results = det.predict(image)
            boxes = []
            if results:
                for res in results:
                    raw = res.get('boxes', []) if isinstance(res, dict) else []
                    for item in raw:
                        pts = np.array(item).tolist()
                        if len(pts) == 4 and len(pts[0]) == 2:
                            boxes.append(pts)
            line = json.dumps(boxes) + '\n'
        except Exception:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            line = '[]\n'

        stdout.write(line.encode('utf-8'))
        stdout.flush()


if __name__ == '__main__':
    run()
