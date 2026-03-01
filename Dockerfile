# -----------------------------------------------------------------------
# OCR Engine — Docker Image
# Base: PyTorch 2.x + CUDA 12.1 (cuda builds) / CPU-only build
#
# Build (CPU):
#   docker build -t ocr-engine .
#
# Build (GPU):
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
#                -t ocr-engine:gpu .
#
# Run:
#   docker run -p 8000:8000 ocr-engine
#   docker run --gpus all -p 8000:8000 ocr-engine:gpu
# -----------------------------------------------------------------------
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

LABEL maintainer="ocr-engine"
LABEL description="DBNet + CRNN tabanli OCR sistemi"

# Sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python bagimliliklari (once yalnizca requirements — layer cache icin)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodu
COPY . .

# Model agirlik dizinlerini olustur (bos, mount noktalari)
RUN mkdir -p \
    ocr_engine/detection/weights \
    ocr_engine/recognition/weights \
    checkpoints

# Port
EXPOSE 8000

# Saglık kontrolu
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Baslat
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]