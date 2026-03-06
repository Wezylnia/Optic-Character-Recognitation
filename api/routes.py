"""
API endpoint'leri
"""

import io
import time
import base64
from typing import List
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
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

_VIS_PALETTE = [
    (0, 200, 0),
    (0, 120, 255),
    (220, 0, 220),
    (0, 200, 200),
    (255, 60, 60),
]


def _draw_boxes_b64(image: np.ndarray, text_boxes, show_text=True, show_confidence=True) -> str:
    """Bounding box'lari gorsel uzerine ciz, base64 PNG dondur."""
    vis = image.copy()
    for idx, tb in enumerate(text_boxes):
        color = _VIS_PALETTE[idx % len(_VIS_PALETTE)]
        box = tb.box.astype(np.int32)
        cv2.polylines(vis, [box.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)

        if show_text or show_confidence:
            parts = []
            if show_text and tb.text:
                parts.append(tb.text)
            if show_confidence:
                parts.append(f"{tb.confidence*100:.0f}%")
            label = "  ".join(parts)
            x, y = int(box[:, 0].min()), int(box[:, 1].min())
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            by = max(y - 4, th + 4)
            cv2.rectangle(vis, (x, by - th - 4), (x + tw + 4, by + baseline), color, cv2.FILLED)
            cv2.putText(vis, label, (x + 2, by - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    _, buf = cv2.imencode('.png', vis)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')


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
    description="Tek bir gorselden metin cikarir. visualized_image alani base64 PNG icerir — bounding box'lari cizili gorsel."
)
async def ocr_single(
    file: UploadFile = File(..., description="Gorsel dosyasi"),
    output_format: OutputFormat = Form(default=OutputFormat.JSON),
    spell_check: bool = Form(default=True),
    language: str = Form(default="tr"),
    visualize: bool = Form(default=True, description="Bounding box gorselini yanita ekle (base64 PNG)")
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

        # Bounding box gorsel — preprocessed goruntu uzerinde ciz (box koordinatlari ona ait)
        visualized_b64 = None
        if visualize and result.text_boxes:
            vis_img = result.source_image if result.source_image is not None else image
            visualized_b64 = _draw_boxes_b64(vis_img, result.text_boxes)

        return OCRResponse(
            success=True,
            text=result.text,
            blocks=blocks,
            processing_time=result.processing_time,
            image_size={"width": w, "height": h},
            visualized_image=visualized_b64
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


@router.post(
    "/ocr/visualize",
    summary="OCR Gorsel Cikti",
    description="Tespit edilen metin kutularini gorsel uzerine cizer ve PNG olarak dondurur"
)
async def ocr_visualize(
    file: UploadFile = File(..., description="Gorsel dosyasi"),
    spell_check: bool = Form(default=False),
    language: str = Form(default="tr"),
    show_text: bool = Form(default=True, description="Kutu uzerine okunan metni yaz"),
    show_confidence: bool = Form(default=True, description="Guven skorunu goster"),
):
    """EasyOCR'in bounding box'ladigi + bizim okuduğumuz sonucu gorsel olarak dondurur"""

    allowed_types = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Desteklenmeyen dosya tipi: {file_ext}")

    try:
        content = await file.read()
        image = image_to_numpy(content)

        pipeline = get_ocr_pipeline()
        result = pipeline.recognize(image, spell_check=spell_check, language=language)

        b64 = _draw_boxes_b64(image, result.text_boxes,
                              show_text=show_text, show_confidence=show_confidence)
        # base64 prefix'i strip et, ham PNG bytes al
        raw = base64.b64decode(b64.split(",", 1)[1])

        return StreamingResponse(
            io.BytesIO(raw),
            media_type="image/png",
            headers={"X-Box-Count": str(len(result.text_boxes)),
                     "X-Processing-Time": f"{result.processing_time:.3f}"}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualize hatasi: {str(e)}")


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