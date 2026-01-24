# Python OCR Engine

Sifirdan gelistirilmis, GPU destekli Python tabanli OCR (Optik Karakter Tanima) motoru.

## Ozellikler

- **Metin Tespiti**: DBNet (Differentiable Binarization Network) tabanli
- **Metin Tanima**: CRNN (CNN + RNN + CTC) tabanli
- **Dil Destegi**: Turkce ve Ingilizce
- **GPU Destegi**: CUDA ile hizli islem
- **Web API**: FastAPI tabanli REST API
- **Web Arayuzu**: Basit ve kullanici dostu arayuz

## Kurulum

```bash
# Virtual environment olustur
python -m venv venv

# Aktive et (Windows)
venv\Scripts\activate

# Bagimliklar yukle
pip install -r requirements.txt
```

## Kullanim

### Python API

```python
from ocr_engine import OCRPipeline

# OCR motoru olustur
ocr = OCRPipeline()

# Gorsel oku
result = ocr.recognize("image.jpg")
print(result.text)
```

### Web API

```bash
# Sunucuyu baslat
python -m api.main

# Curl ile test
curl -X POST -F "file=@image.jpg" http://localhost:8000/ocr
```

### Web Arayuzu

Tarayicida `http://localhost:8000` adresine gidin.

## Proje Yapisi

```
ocr_engine/          # Ana OCR kutuphanesi
  preprocessing/     # Goruntu on isleme
  detection/         # Metin tespiti (DBNet)
  recognition/       # Metin tanima (CRNN)
  postprocessing/    # Son isleme

training/            # Egitim scriptleri
api/                 # FastAPI uygulamasi
web/                 # Web arayuzu
data/                # Veri ve fontlar
tests/               # Test dosyalari
```

## Egitim

### Sentetik Veri Uretimi

```bash
python training/generate_synthetic.py --output data/synthetic --num_samples 10000
```

### Model Egitimi

```bash
# Detection modeli egit
python training/train_detection.py --config config.yaml

# Recognition modeli egit
python training/train_recognition.py --config config.yaml
```

## Yapilandirma

Tum ayarlar `config.yaml` dosyasinda bulunur:

- `general`: Genel ayarlar (device, workers)
- `preprocessing`: Goruntu on isleme ayarlari
- `detection`: Metin tespit modeli ayarlari
- `recognition`: Metin tanima modeli ayarlari
- `training`: Egitim ayarlari
- `api`: Web API ayarlari

## Lisans

MIT License
