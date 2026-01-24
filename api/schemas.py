"""
Pydantic modelleri - API request/response schemalar
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class OutputFormat(str, Enum):
    """Cikis formati"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


class OCRRequest(BaseModel):
    """OCR istek modeli"""
    
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Cikis formati"
    )
    detect_tables: bool = Field(
        default=False,
        description="Tablo algilama etkinlestir"
    )
    spell_check: bool = Field(
        default=True,
        description="Yazim kontrolu etkinlestir"
    )
    preserve_layout: bool = Field(
        default=False,
        description="Sayfa duzeni koru"
    )
    language: str = Field(
        default="tr",
        description="Dil (tr, en, auto)"
    )


class BoundingBox(BaseModel):
    """Metin kutusu koordinatlari"""
    
    x1: int = Field(..., description="Sol ust X")
    y1: int = Field(..., description="Sol ust Y")
    x2: int = Field(..., description="Sag alt X")
    y2: int = Field(..., description="Sag alt Y")
    polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Dortgen polygon koordinatlari [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )


class TextBlock(BaseModel):
    """Metin blogu"""
    
    text: str = Field(..., description="Tanınan metin")
    confidence: float = Field(..., description="Güven skoru (0-1)")
    bounding_box: BoundingBox = Field(..., description="Konum")


class TableCell(BaseModel):
    """Tablo hücresi"""
    
    row: int = Field(..., description="Satir indeksi")
    col: int = Field(..., description="Sutun indeksi")
    text: str = Field(..., description="Hücre metni")
    bounding_box: Optional[BoundingBox] = None


class Table(BaseModel):
    """Tablo verisi"""
    
    rows: int = Field(..., description="Satir sayisi")
    cols: int = Field(..., description="Sutun sayisi")
    cells: List[TableCell] = Field(..., description="Hücre listesi")
    bounding_box: Optional[BoundingBox] = None


class OCRResponse(BaseModel):
    """OCR yanit modeli"""
    
    success: bool = Field(..., description="Islem basarili mi")
    text: str = Field(..., description="Tam metin")
    blocks: List[TextBlock] = Field(
        default=[],
        description="Metin bloklari"
    )
    tables: List[Table] = Field(
        default=[],
        description="Tespit edilen tablolar"
    )
    processing_time: float = Field(
        ...,
        description="Isleme suresi (saniye)"
    )
    image_size: Dict[str, int] = Field(
        ...,
        description="Gorsel boyutu {width, height}"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ek metadata"
    )


class ErrorResponse(BaseModel):
    """Hata yaniti"""
    
    success: bool = Field(default=False)
    error: str = Field(..., description="Hata mesaji")
    error_code: str = Field(..., description="Hata kodu")
    details: Optional[Dict[str, Any]] = None


class BatchOCRRequest(BaseModel):
    """Toplu OCR istegi"""
    
    output_format: OutputFormat = Field(default=OutputFormat.JSON)
    spell_check: bool = Field(default=True)


class BatchOCRResponse(BaseModel):
    """Toplu OCR yaniti"""
    
    success: bool = Field(...)
    results: List[OCRResponse] = Field(...)
    total_processing_time: float = Field(...)
    num_processed: int = Field(...)
    num_failed: int = Field(default=0)


class HealthResponse(BaseModel):
    """Saglik kontrolu yaniti"""
    
    status: str = Field(..., description="Servis durumu")
    version: str = Field(..., description="API versiyonu")
    cuda_available: bool = Field(..., description="CUDA mevcut mu")
    models_loaded: Dict[str, bool] = Field(..., description="Model durumu")
