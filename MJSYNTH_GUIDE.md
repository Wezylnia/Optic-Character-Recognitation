# MJSynth Veri Seti Kullanimi

MJSynth veri setini OCR motorunda kullanmak icin adim adim kilavuz.

## 1. Veri Setini Hazirlama

### Adim 1: MJSynth'i indirin
MJSynth veri setini [https://www.robots.ox.ac.uk/~vgg/data/text/](https://www.robots.ox.ac.uk/~vgg/data/text/) adresinden indirin.

Genellikle su yapiyi icerir:
```
MJSynth/
├── mnt/ramdisk/max/90kDICT32px/
│   ├── 1/
│   ├── 2/
│   └── ...
├── annotation_train.txt
├── annotation_val.txt
└── annotation_test.txt
```

### Adim 2: Veri setini JSON formatina cevirin

**Basit kullanim (ilk 10,000 ornek ile test):**
```bash
python training/convert_mjsynth.py \
    --data_root "path/to/MJSynth" \
    --annotation "path/to/MJSynth/annotation_train.txt" \
    --output "data/mjsynth_train.json" \
    --max_samples 10000
```

**Tam veri seti (butun ornekler):**
```bash
python training/convert_mjsynth.py \
    --data_root "path/to/MJSynth" \
    --annotation "path/to/MJSynth/annotation_train.txt" \
    --output "data/mjsynth_train.json" \
    --split
```

`--split` parametresi ile otomatik olarak train/val/test'e ayirir.

**Daha hizli (validasyon olmadan):**
```bash
python training/convert_mjsynth.py \
    --data_root "path/to/MJSynth" \
    --annotation "path/to/MJSynth/annotation_train.txt" \
    --output "data/mjsynth_train.json" \
    --no_validate
```

### Ornek Komutlar

Eger MJSynth'i `C:\Datasets\MJSynth` klasorune indirdiyseniz:

```bash
# Train seti
python training/convert_mjsynth.py \
    --data_root "C:\Datasets\MJSynth" \
    --annotation "C:\Datasets\MJSynth\annotation_train.txt" \
    --output "data/mjsynth_train.json"

# Validation seti
python training/convert_mjsynth.py \
    --data_root "C:\Datasets\MJSynth" \
    --annotation "C:\Datasets\MJSynth\annotation_val.txt" \
    --output "data/mjsynth_val.json"

# Test seti
python training/convert_mjsynth.py \
    --data_root "C:\Datasets\MJSynth" \
    --annotation "C:\Datasets\MJSynth\annotation_test.txt" \
    --output "data/mjsynth_test.json"
```

## 2. Model Egitimi

JSON dosyalari hazir olduktan sonra egitimi baslatabilirsiniz:

### Hizli Test Egitimi (10 epoch, kucuk veri)
```bash
python training/train_mjsynth.py \
    --data_root "C:\Datasets\MJSynth" \
    --train_json "data/mjsynth_train.json" \
    --val_json "data/mjsynth_val.json" \
    --epochs 10 \
    --batch_size 64 \
    --save_dir "checkpoints/mjsynth_test"
```

### Tam Egitim (50 epoch)
```bash
python training/train_mjsynth.py \
    --data_root "C:\Datasets\MJSynth" \
    --train_json "data/mjsynth_train.json" \
    --val_json "data/mjsynth_val.json" \
    --epochs 50 \
    --batch_size 64 \
    --save_dir "checkpoints/mjsynth"
```

### Egitimi Devam Ettirme
```bash
python training/train_mjsynth.py \
    --data_root "C:\Datasets\MJSynth" \
    --train_json "data/mjsynth_train.json" \
    --val_json "data/mjsynth_val.json" \
    --epochs 100 \
    --resume "checkpoints/mjsynth/recognition_epoch_50.pth" \
    --save_dir "checkpoints/mjsynth"
```

## 3. Egitilmis Modeli Kullanma

Egitim tamamlandiktan sonra en iyi modeli OCR pipeline'da kullanabilirsiniz:

```python
from ocr_engine import OCRPipeline

# En iyi checkpoint ile pipeline olustur
ocr = OCRPipeline(
    recognition_weights="checkpoints/mjsynth/best_recognition.pth"
)

# Gorsel uzerinde test et
result = ocr.recognize("test_image.jpg")
print(result.text)
```

## 4. Tips ve Optimizasyonlar

### GPU Bellek Sorunu
Eger GPU bellek hatasi aliyorsaniz batch size'i azaltin:
```bash
--batch_size 32  # veya 16
```

### Daha Hizli Egitim
- `num_workers` artirin (config.yaml icinde)
- Mixed precision training kullanin (torch.cuda.amp)
- Daha kucuk bir model kullanin (hidden_size: 128)

### Veri Setini Filtreleme
Cok uzun veya cok kisa kelimeleri filtrelemek isterseniz `convert_mjsynth.py`'yi duzenleyebilirsiniz.

## Ornek Cikti

Converter calistiginda su sekilde bir cikti gorursunuz:
```
MJSynth Converter
Data root: C:\Datasets\MJSynth
Annotation: C:\Datasets\MJSynth\annotation_train.txt
Output: data/mjsynth_train.json

Annotation dosyasi okunuyor: C:\Datasets\MJSynth\annotation_train.txt
Toplam 7224612 satir bulundu
Parse ediliyor: 100%|████████████| 7224612/7224612

==================================================
VERI SETI ISTATISTIKLERI
==================================================
Toplam ornek sayisi: 7224612

Metin uzunluklari:
  - Min: 1
  - Max: 23
  - Ortalama: 5.89

En sik 10 kelime:
  - 'the': 45892 kez
  - 'and': 23456 kez
  ...

Benzersiz kelime sayisi: 88172

✓ Basariyla kaydedildi: 7224612 ornek
✓ Dosya boyutu: 845.23 MB
```

## Sorun Giderme

**Soru:** "Gorsel dosyalari bulunamiyor" hatasi aliyorum.
**Cevap:** `--data_root` parametresinin dogru oldugunu kontrol edin. Gorsel dosyalari annotation'daki yollara gore aranir.

**Soru:** Egitim cok yavas.
**Cevap:** 
1. `--no_validate` kullanarak validasyonu atlayin (sadece ilk kez)
2. SSD kullanin (HDD yerine)
3. `num_workers` artirin

**Soru:** OOM (Out of Memory) hatasi.
**Cevap:** Batch size'i azaltin: `--batch_size 32` veya `--batch_size 16`
