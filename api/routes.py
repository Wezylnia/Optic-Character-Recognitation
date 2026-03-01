"""
API endpoint'leri
"""

import io
import time
from typing import List
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from api.schemas import (
    OCRResponse, ErrorResponse, BatchOCRResponse,
    HealthResponse, TextBlock, BoundingBox,
    OutputFormat
)

# Router
router = APIRouter()

# OCR Pipeline (global, lazy init)
_ocr_pipeline = None


def get_ocr_pipeline():
    """OCR pipeline'i getir (lazy initialization)"""
    global _ocr_pipeline
    
    if _ocr_pipeline is None:
        from ocr_engine import OCRPipeline
        _ocr_pipeline = OCRPipeline()
    
    return _ocr_pipeline


def image_to_numpy(file_content: bytes) -> np.ndarray:
    """Dosya icerigini numpy array'e donustur"""
    nparr = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Gorsel decode edilemedi")
    
    return image


@router.post(
    "/ocr",
    response_model=OCRResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Tekil OCR",
    description="Tek bir gorselden metin cikarir"
)
async def ocr_single(
    file: UploadFile = File(..., description="Gorsel dosyasi"),
    output_format: OutputFormat = Form(default=OutputFormat.JSON),
    spell_check: bool = Form(default=True),
    language: str = Form(default="tr")
):
    """Tek gorsel icin OCR"""
    
    # Dosya tipi kontrolu
    allowed_types = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya tipi: {file_ext}. Desteklenen: {allowed_types}"
        )
    
    try:
        # Dosyayi oku
        content = await file.read()
        image = image_to_numpy(content)
        
        h, w = image.shape[:2]
        
        # OCR
        pipeline = get_ocr_pipeline()
        result = pipeline.recognize(image, spell_check=spell_check, language=language)
        
        # Yaniti olustur
        blocks = []
        for tb in result.text_boxes:
            blocks.append(TextBlock(
                text=tb.text,
                confidence=tb.confidence,
                bounding_box=BoundingBox(
                    x1=tb.x1,
                    y1=tb.y1,
                    x2=tb.x2,
                    y2=tb.y2,
                    polygon=tb.box.tolist()
                )
            ))
        
        return OCRResponse(
            success=True,
            text=result.text,
            blocks=blocks,
            processing_time=result.processing_time,
            image_size={"width": w, "height": h}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR hatasi: {str(e)}")


@router.post(
    "/ocr/batch",
    response_model=BatchOCRResponse,
    summary="Toplu OCR",
    description="Birden fazla gorselden metin cikarir"
)
async def ocr_batch(
    files: List[UploadFile] = File(..., description="Gorsel dosyalari"),
    output_format: OutputFormat = Form(default=OutputFormat.JSON),
    spell_check: bool = Form(default=True),
    language: str = Form(default="tr")
):
    """Toplu OCR"""
    
    start_time = time.time()
    results = []
    num_failed = 0
    
    pipeline = get_ocr_pipeline()
    
    for file in files:
        try:
            content = await file.read()
            image = image_to_numpy(content)

            h, w = image.shape[:2]
            result = pipeline.recognize(image, spell_check=spell_check, language=language)
            
            blocks = [
                TextBlock(
                    text=tb.text,
                    confidence=tb.confidence,
                    bounding_box=BoundingBox(
                        x1=tb.x1, y1=tb.y1, x2=tb.x2, y2=tb.y2,
                        polygon=tb.box.tolist()
                    )
                )
                for tb in result.text_boxes
            ]
            
            results.append(OCRResponse(
                success=True,
                text=result.text,
                blocks=blocks,
                processing_time=result.processing_time,
                image_size={"width": w, "height": h},
                metadata={"filename": file.filename}
            ))
            
        except Exception as e:
            num_failed += 1
            results.append(OCRResponse(
                success=False,
                text="",
                blocks=[],
                processing_time=0,
                image_size={"width": 0, "height": 0},
                metadata={"filename": file.filename, "error": str(e)}
            ))
    
    total_time = time.time() - start_time
    
    return BatchOCRResponse(
        success=num_failed == 0,
        results=results,
        total_processing_time=total_time,
        num_processed=len(results),
        num_failed=num_failed
    )


@router.post(
    "/ocr/table",
    summary="Tablo OCR",
    description="Gorsel icindeki tablodan metin cikarir (henuz desteklenmiyor)"
)
async def ocr_table(
    file: UploadFile = File(..., description="Gorsel dosyasi")
):
    """Tablo OCR — henuz implement edilmedi"""
    raise HTTPException(
        status_code=501,
        detail="Tablo OCR henuz desteklenmiyor. Yakin zamanda eklenecek."
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Saglik Kontrolu",
    description="API saglik durumunu kontrol eder"
)
async def health_check():
    """Saglik kontrolu"""
    import torch
    
    cuda_available = torch.cuda.is_available()
    
    # Model durumu
    models_loaded = {
        "detection": False,
        "recognition": False
    }
    
    try:
        pipeline = get_ocr_pipeline()
        models_loaded["detection"] = pipeline.detection_model is not None
        models_loaded["recognition"] = pipeline.recognition_model is not None
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cuda_available=cuda_available,
        models_loaded=models_loaded
    )


@router.get(
    "/info",
    summary="API Bilgisi",
    description="API hakkinda bilgi verir"
)
async def api_info():
    """API bilgisi"""
    return {
        "name": "OCR Engine API",
        "version": "1.0.0",
        "description": "Python tabanli OCR motoru",
        "endpoints": {
            "/ocr": "Tekil gorsel OCR",
            "/ocr/batch": "Toplu OCR",
            "/ocr/table": "Tablo OCR",
            "/health": "Saglik kontrolu"
        },
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        "languages": ["tr", "en"]
    }
