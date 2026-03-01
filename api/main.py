"""
FastAPI ana uygulama
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import yaml

from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama yasam dongusu — startup ve shutdown"""
    # --- Startup ---
    print("OCR Engine API baslatiliyor...")
    from api.routes import get_ocr_pipeline
    try:
        get_ocr_pipeline()
        print("OCR pipeline hazir!")
    except Exception as e:
        print(f"[UYARI] Pipeline baslatilamadi, ilk istekte tekrar denenecek: {e}")
    print("API hazir!")

    yield  # Uygulama calisiyor

    # --- Shutdown ---
    print("OCR Engine API kapatiliyor...")

# Config yukle
config_path = Path(__file__).parent.parent / 'config.yaml'
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
else:
    config = {'api': {'host': '0.0.0.0', 'port': 8000}}

api_config = config.get('api', {})

# FastAPI uygulamasi
app = FastAPI(
    title="OCR Engine API",
    lifespan=lifespan,
    description="""
    Python tabanli OCR (Optik Karakter Tanima) motoru.
    
    ## Ozellikler
    
    - Metin tespiti (DBNet)
    - Metin tanima (CRNN)
    - Turkce ve Ingilizce dil destegi
    - Tablo algilama
    - Toplu islem destegi
    
    ## Kullanim
    
    1. `/ocr` endpoint'ine gorsel yukleyin
    2. JSON formatinda sonuc alin
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1", tags=["OCR"])

# Static files (web UI)
web_dir = Path(__file__).parent.parent / 'web'
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/")
    async def root():
        """Ana sayfa - Web UI"""
        return FileResponse(str(web_dir / 'index.html'))
else:
    @app.get("/")
    async def root():
        """Ana sayfa"""
        return {
            "message": "OCR Engine API",
            "docs": "/docs",
            "health": "/api/v1/health"
        }


def run():
    """Uygulamayi calistir"""
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    import os
    dev_mode = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=dev_mode,
        log_level="info"
    )


if __name__ == "__main__":
    run()
