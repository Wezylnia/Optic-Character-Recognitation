"""
Real-World Dataset Birleştirici
================================
Desteklenen kaynaklar:
  archive1  → IIIT5K-Word V3.0   (traindata.csv + testdata.csv)
  archive2  → SVT                  (train.xml + test.xml  — bounding-box kroplane)
  archive3  → ICDAR + IIIT5K kopyası (gt.txt + traindata.csv)

Çıktı: data/real_world_train.json  +  data/real_world_test.json
Format: [{"image_path": "abs/path/img.png", "text": "WORD"}, ...]
"""

import os
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# ──────────────────────────────────────────────
BASE = Path("c:/Ocr/DataSets/real-word")
OUT_DIR = Path("c:/Ocr/data")
OUT_TRAIN = OUT_DIR / "real_world_train.json"
OUT_TEST  = OUT_DIR / "real_world_test.json"

# SVT krop kayıt klasörü
SVT_CROPS_DIR = BASE / "archive2" / "word_crops"
# ──────────────────────────────────────────────


def clean_text(t: str) -> str:
    """Başındaki/sonundaki boşluk ve tırnak işaretlerini temizle."""
    return t.strip().strip('"').strip("'").strip()


def is_valid(text: str) -> bool:
    """Boş, tek karakter veya aşırı uzun kelimeleri filtrele."""
    t = clean_text(text)
    if len(t) < 1:
        return False
    if len(t) > 25:
        return False
    # Sadece alfanümerik + bazı özel karakterler
    if not re.match(r'^[A-Za-z0-9\-\.\/\&\']+$', t):
        return False
    return True


# ──────────────────────────────────────────────
# 1. ARCHIVE 1 — IIIT5K-Word V3.0  (CSV)
# ──────────────────────────────────────────────
def load_archive1_csv(csv_path: Path, img_root: Path):
    """
    CSV format: ImgName,GroundTruth,smallLexi,mediumLexi
      ImgName örn: train/1009_2.png  ya da  test/1009_2.png
    img_root = .../IIIT5K-Word_V3.0/IIIT5K/
    """
    entries = []
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} bulunamadı")
        return entries

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # başlık satırını atla
        for row in reader:
            if len(row) < 2:
                continue
            img_rel, text = row[0].strip(), clean_text(row[1])
            # img_rel = "train/1009_2.png"  → sadece dosya adı gerekiyor
            img_filename = Path(img_rel).name
            split = Path(img_rel).parts[0]  # "train" veya "test"
            img_abs = img_root / split / img_filename
            if not is_valid(text):
                continue
            if not img_abs.exists():
                continue
            entries.append({"image_path": str(img_abs), "text": text.upper()})
    return entries


# ──────────────────────────────────────────────
# 2. ARCHIVE 2 — SVT  (XML + krop)
# ──────────────────────────────────────────────
def load_archive2_xml(xml_path: Path, img_dir: Path, crops_dir: Path):
    """
    SVT XML formatı:
      <image><imageName>img/14_03.jpg</imageName>
        <taggedRectangles>
          <taggedRectangle height="75" width="236" x="375" y="253">
            <tag>LIVING</tag>
          </taggedRectangle>
        </taggedRectangles>
      </image>
    Her dikdörtgeni scene image'dan kırpıp crops_dir'e kaydet.
    """
    entries = []
    if not xml_path.exists():
        print(f"  [SKIP] {xml_path} bulunamadı")
        return entries

    crops_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    skipped_img = 0
    skipped_text = 0
    cropped = 0

    for image_el in root.findall('image'):
        img_name_el = image_el.find('imageName')
        if img_name_el is None:
            continue
        img_rel = img_name_el.text.strip()          # "img/14_03.jpg"
        scene_path = img_dir.parent / img_rel       # archive2/img/14_03.jpg
        if not scene_path.exists():
            skipped_img += 1
            continue

        scene = cv2.imread(str(scene_path))
        if scene is None:
            skipped_img += 1
            continue
        h_scene, w_scene = scene.shape[:2]

        scene_stem = Path(img_rel).stem   # "14_03"

        for idx, rect in enumerate(image_el.findall('.//taggedRectangle')):
            tag_el = rect.find('tag')
            if tag_el is None or tag_el.text is None:
                continue
            text = clean_text(tag_el.text)
            if not is_valid(text):
                skipped_text += 1
                continue

            try:
                x = int(rect.get('x', 0))
                y = int(rect.get('y', 0))
                w = int(rect.get('width', 0))
                h = int(rect.get('height', 0))
            except (ValueError, TypeError):
                continue

            # Sınır dışı koordinatları kırp
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_scene, x + w)
            y2 = min(h_scene, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = scene[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_filename = f"{scene_stem}_{idx:03d}_{text}.jpg"
            crop_path = crops_dir / crop_filename
            if not crop_path.exists():
                cv2.imwrite(str(crop_path), crop)
            entries.append({"image_path": str(crop_path), "text": text.upper()})
            cropped += 1

    print(f"  SVT {xml_path.name}: {cropped} krop oluşturuldu | {skipped_img} scene bulunamadı | {skipped_text} metin geçersiz")
    return entries


# ──────────────────────────────────────────────
# 3. ARCHIVE 3 — ICDAR gt.txt
# ──────────────────────────────────────────────
def load_icdar_gt(gt_path: Path, images_dir: Path):
    """
    gt.txt satır formatı: word_NNN.png, "LABEL"
    """
    entries = []
    if not gt_path.exists():
        print(f"  [SKIP] {gt_path} bulunamadı")
        return entries

    with open(gt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # örn:  word_838.png, "BOARD"
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            img_name = parts[0].strip()
            text = clean_text(parts[1])
            if not is_valid(text):
                continue
            img_abs = images_dir / img_name
            if not img_abs.exists():
                continue
            entries.append({"image_path": str(img_abs), "text": text.upper()})
    return entries


# ──────────────────────────────────────────────
# 4. ARCHIVE 3 — IIIT5K kopyası (CSV)
# ──────────────────────────────────────────────
def load_iiit5k_csv(csv_path: Path, images_dir: Path):
    """
    CSV format: ImgName,GroundTruth
      ImgName örn: 912_3.png
    """
    entries = []
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} bulunamadı")
        return entries

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # başlık
        for row in reader:
            if len(row) < 2:
                continue
            img_name = row[0].strip()
            text = clean_text(row[1])
            if not is_valid(text):
                continue
            img_abs = images_dir / img_name
            if not img_abs.exists():
                continue
            entries.append({"image_path": str(img_abs), "text": text.upper()})
    return entries


# ──────────────────────────────────────────────
def deduplicate(entries):
    """image_path bazında tekilleştir."""
    seen = set()
    result = []
    for e in entries:
        k = e["image_path"]
        if k not in seen:
            seen.add(k)
            result.append(e)
    return result


# ──────────────────────────────────────────────
def main():
    train_entries = []
    test_entries  = []

    # ── Archive 1: IIIT5K ──────────────────────
    print("\n[1] Archive1 — IIIT5K-Word V3.0")
    iiit5k_root = BASE / "archive1" / "IIIT5K-Word_V3.0" / "IIIT5K"

    a1_train = load_archive1_csv(BASE / "archive1" / "traindata.csv", iiit5k_root)
    a1_test  = load_archive1_csv(BASE / "archive1" / "testdata.csv",  iiit5k_root)
    print(f"  train: {len(a1_train)}  |  test: {len(a1_test)}")
    train_entries += a1_train
    test_entries  += a1_test

    # ── Archive 2: SVT ─────────────────────────
    print("\n[2] Archive2 — SVT (Street View Text)")
    svt_img_dir = BASE / "archive2" / "img"

    a2_train = load_archive2_xml(BASE / "archive2" / "train.xml", svt_img_dir, SVT_CROPS_DIR / "train")
    a2_test  = load_archive2_xml(BASE / "archive2" / "test.xml",  svt_img_dir, SVT_CROPS_DIR / "test")
    print(f"  train: {len(a2_train)}  |  test: {len(a2_test)}")
    train_entries += a2_train
    test_entries  += a2_test

    # ── Archive 3: ICDAR ───────────────────────
    print("\n[3] Archive3 — ICDAR")
    icdar_base = BASE / "archive3" / "icdar"

    a3i_train = load_icdar_gt(icdar_base / "train" / "gt.txt",  icdar_base / "train" / "images")
    a3i_test  = load_icdar_gt(icdar_base / "test"  / "gt.txt",  icdar_base / "test"  / "images")
    print(f"  train: {len(a3i_train)}  |  test: {len(a3i_test)}")
    train_entries += a3i_train
    test_entries  += a3i_test

    # ── Archive 3: IIIT5K kopyası ──────────────
    print("\n[4] Archive3 — IIIT5K kopyası")
    iiit5k2_base = BASE / "archive3" / "iiit5k"

    a3k_train = load_iiit5k_csv(iiit5k2_base / "train" / "traindata.csv", iiit5k2_base / "train" / "images")
    a3k_test  = load_iiit5k_csv(iiit5k2_base / "test"  / "testdata.csv",  iiit5k2_base / "test"  / "images")
    print(f"  train: {len(a3k_train)}  |  test: {len(a3k_test)}")
    train_entries += a3k_train
    test_entries  += a3k_test

    # ── Tekilleştir & Kaydet ───────────────────
    train_entries = deduplicate(train_entries)
    test_entries  = deduplicate(test_entries)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    with open(OUT_TEST, 'w', encoding='utf-8') as f:
        json.dump(test_entries, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"TRAIN toplam : {len(train_entries):,} sample")
    print(f"TEST  toplam : {len(test_entries):,} sample")
    print(f"TRAIN JSON   : {OUT_TRAIN}")
    print(f"TEST  JSON   : {OUT_TEST}")
    print("Tamamlandi!")


if __name__ == "__main__":
    main()