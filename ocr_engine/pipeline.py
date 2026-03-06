"""
OCR Pipeline - Detection ve Recognition entegrasyonu
"""

import cv2
import numpy as np
import torch
from typing import Any, List, Optional, Tuple, Union, Dict
from pathlib import Path
import yaml

from .pipeline_types import TextBox, OCRResult
from .preprocessing import ImageProcessor, Binarizer, Deskewer, Denoiser, ImageEnhancer, PerspectiveCorrector
from .detection import DBNet, DBPostProcessor
from .detection.postprocess import sort_boxes_by_position, adaptive_sort_boxes, AdaptiveLineGrouper
from .recognition import (
    CRNN, CTCDecoder, Vocabulary,
    AttentionCRNN, AttentionDecodeHelper, build_attention_crnn,
)
from .recognition.decoder import CTCPrefixDecoder
from .postprocessing import LayoutAnalyzer, DocumentLayout, SpellChecker


class OCRPipeline:
    """
    Ana OCR pipeline
    
    Gorsellerden metin cikarmak icin detection ve recognition
    modellerini birlestiren end-to-end pipeline.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        detection_weights: Optional[str] = None,
        recognition_weights: Optional[str] = None
    ):
        """
        Args:
            config_path: Yapilandirma dosyasi yolu
            device: Cihaz (cuda veya cpu)
            detection_weights: Detection model agirliklari
            recognition_weights: Recognition model agirliklari
        """
        # Yapilandirma yukle
        self.config = self._load_config(config_path)
        
        # Cihaz ayarla
        if device is None:
            device = self.config.get('general', {}).get('device', 'cuda')
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA mevcut degil, CPU kullaniliyor.")
            device = 'cpu'
        
        self.device = torch.device(device)
        
        # Preprocessing bilesenleri
        self._init_preprocessing()
        
        # Detection modeli
        self._init_detection(detection_weights)
        
        # Recognition modeli
        self._init_recognition(recognition_weights)
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Yapilandirma dosyasini yukle"""
        if config_path is None:
            # Varsayilan config yolu
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Varsayilan yapilandirma
        return {
            'general': {'device': 'cuda'},
            'preprocessing': {
                'target_size': [1280, 960],
                'denoise': {'enabled': True, 'method': 'bilateral', 'strength': 10},
                'deskew': {'enabled': True, 'max_angle': 45}
            },
            'detection': {
                'model': {'backbone': 'resnet18'},
                'inference': {
                    'threshold': 0.3,
                    'box_threshold': 0.5,
                    'max_candidates': 1000,
                    'unclip_ratio': 1.5
                }
            },
            'recognition': {
                'model': {
                    'input_height': 32,
                    'input_width': 256,
                    'hidden_size': 256,
                    'num_layers': 2
                },
                'inference': {'beam_width': 5}
            }
        }
    
    def _init_preprocessing(self):
        """Preprocessing bilesenlerini baslat"""
        preproc_cfg = self.config.get('preprocessing', {})
        
        # Image processor
        self.image_processor = ImageProcessor(
            target_size=tuple(preproc_cfg.get('target_size', [1280, 960]))
        )
        
        # Denoiser
        denoise_cfg = preproc_cfg.get('denoise', {})
        if denoise_cfg.get('enabled', True):
            self.denoiser = Denoiser(
                method=denoise_cfg.get('method', 'bilateral'),
                strength=denoise_cfg.get('strength', 10)
            )
        else:
            self.denoiser = None
        
        # Deskewer
        deskew_cfg = preproc_cfg.get('deskew', {})
        if deskew_cfg.get('enabled', True):
            self.deskewer = Deskewer(
                max_angle=deskew_cfg.get('max_angle', 45)
            )
        else:
            self.deskewer = None
        
        # Binarizer (opsiyonel, recognition icin kullanilabilir)
        self.binarizer = Binarizer(method='adaptive')

        # Goruntu iyilestirme
        enh_cfg = preproc_cfg.get('enhance', {})
        if enh_cfg.get('enabled', True):
            tile = enh_cfg.get('clahe_tile_size', [8, 8])
            self.enhancer = ImageEnhancer(
                clahe_clip_limit=enh_cfg.get('clahe_clip_limit', 2.0),
                clahe_tile_size=tuple(tile),
                sharpen_strength=enh_cfg.get('sharpen_strength', 0.5),
                shadow_removal=enh_cfg.get('shadow_removal', True),
                auto_mode=(enh_cfg.get('mode', 'auto') == 'auto'),
            )
            self._enhance_mode              = enh_cfg.get('mode', 'auto')
            self._enhance_quality_threshold  = enh_cfg.get('auto_quality_threshold', 0.4)
            self._enhance_sharpness_threshold = enh_cfg.get('auto_sharpness_threshold', 50.0)
        else:
            self.enhancer = None

        # Perspektif duzeltici
        persp_cfg = preproc_cfg.get('perspective', {})
        if persp_cfg.get('enabled', True):
            self.perspective_corrector = PerspectiveCorrector(
                min_area_ratio=persp_cfg.get('min_area_ratio', 0.1),
                max_angle_deviation=persp_cfg.get('max_angle_deviation', 30.0),
            )
        else:
            self.perspective_corrector = None

    def _init_detection(self, weights_path: Optional[str]):
        """Detection modelini baslat — EasyOCR CRAFT backend"""
        det_cfg = self.config.get('detection', {})
        use_gpu = (str(self.device) != 'cpu')

        try:
            import easyocr
            self._easy_ocr = easyocr.Reader(['en', 'tr'], gpu=use_gpu, verbose=False)
            self._paddle_proc = None
            self.detection_model = 'easy'  # sentinel
            print("Detection: EasyOCR (CRAFT) yuklendi.")
        except Exception as e:
            print(f"[UYARI] EasyOCR yuklenemedi ({e}), DBNet'e geri donuluyor.")
            self._easy_ocr = None
            self._paddle_proc = None
            model_cfg = det_cfg.get('model', {})
            self.detection_model = DBNet(
                backbone=model_cfg.get('backbone', 'resnet18'),
                pretrained=model_cfg.get('pretrained', True)
            ).to(self.device)
            self.detection_model.eval()
            inf_cfg = det_cfg.get('inference', {})
            self.detection_postprocessor = DBPostProcessor(
                threshold=inf_cfg.get('threshold', 0.3),
                box_threshold=inf_cfg.get('box_threshold', 0.5),
                max_candidates=inf_cfg.get('max_candidates', 1000),
                unclip_ratio=inf_cfg.get('unclip_ratio', 1.5)
            )
    
    def _init_recognition(self, weights_path: Optional[str]):
        """Recognition modelini baslat (CTC veya Attention)"""
        rec_cfg   = self.config.get('recognition', {})
        model_cfg = rec_cfg.get('model', {})
        inf_cfg   = rec_cfg.get('inference', {})
        attn_cfg  = rec_cfg.get('attention', {})

        self.recognition_mode = rec_cfg.get('mode', 'ctc')

        # --- Vocabulary ---
        use_sos_eos = (self.recognition_mode == 'attention')
        self.vocab = Vocabulary(include_sos_eos=use_sos_eos)

        # --- Model ---
        if self.recognition_mode == 'attention':
            self.recognition_model = AttentionCRNN(
                num_classes    = self.vocab.size,
                input_channels = 1,
                hidden_size    = model_cfg.get('hidden_size', 256),
                num_layers     = model_cfg.get('num_layers', 2),
                attn_dim       = attn_cfg.get('attn_dim', 256),
                dropout        = model_cfg.get('dropout', 0.1),
                encoder_type   = 'vgg',
                sos_idx        = self.vocab.sos_idx,
                eos_idx        = self.vocab.eos_idx,
            ).to(self.device)

            # Attention agirlik dosyasi
            a_weights = weights_path or attn_cfg.get('weights_path')
            if a_weights and Path(a_weights).exists():
                ckpt = torch.load(a_weights, map_location=self.device, weights_only=False)
                state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
                self.recognition_model.load_state_dict(state)
                print(f"Attention agirliklari yuklendi: {a_weights}")
            else:
                print("[UYARI] Attention agirlik dosyasi bulunamadi, rastgele agirliklar kullaniliyor.")

            self._attn_decoder = AttentionDecodeHelper(
                self.vocab,
                sos_idx=self.vocab.sos_idx,
                eos_idx=self.vocab.eos_idx,
            )
        else:
            # CTC modu (varsayilan)
            self.recognition_model = CRNN(
                num_classes    = self.vocab.num_classes,
                input_channels = 1,
                hidden_size    = model_cfg.get('hidden_size', 256),
                num_layers     = model_cfg.get('num_layers', 2),
                dropout        = model_cfg.get('dropout', 0.1)
            ).to(self.device)

            ctc_weights = weights_path or rec_cfg.get('weights_path')
            if ctc_weights and Path(ctc_weights).exists():
                ckpt = torch.load(ctc_weights, map_location=self.device, weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt)
                self.recognition_model.load_state_dict(state_dict)
                print(f"Recognition agirliklari yuklendi: {ctc_weights}")
            else:
                print("[UYARI] CTC agirlik dosyasi bulunamadi, rastgele agirliklar kullaniliyor.")

            self._attn_decoder = None

        self.recognition_model.eval()

        # --- CTC Decoderlar ---
        self.decoder = CTCDecoder(self.vocab)
        self.beam_width = inf_cfg.get('beam_width', 5)
        decoder_type   = inf_cfg.get('decoder', 'prefix')

        if decoder_type == 'prefix' and self.beam_width > 1:
            self._prefix_decoder = CTCPrefixDecoder(
                self.vocab,
                beam_width=self.beam_width
            )
        else:
            self._prefix_decoder = None  # greedy kullan

        # --- Layout analizoru ---
        layout_cfg = self.config.get('postprocessing', {}).get('layout', {})
        if layout_cfg.get('enabled', True):
            self._layout_analyzer = LayoutAnalyzer(
                heading_height_ratio  = layout_cfg.get('heading_height_ratio',  1.5),
                subhead_height_ratio  = layout_cfg.get('subhead_height_ratio',  1.25),
                caption_height_ratio  = layout_cfg.get('caption_height_ratio',  0.75),
                column_gap_ratio      = layout_cfg.get('column_gap_ratio',      0.04),
                max_columns           = layout_cfg.get('max_columns',           4),
                line_merge_gap_ratio  = layout_cfg.get('line_merge_gap_ratio',  0.5),
                paragraph_gap_ratio   = layout_cfg.get('paragraph_gap_ratio',   1.2),
            )
        else:
            self._layout_analyzer = None

        # --- Recognition girdi boyutlari ---
        self.rec_input_height = model_cfg.get('input_height', 32)
        self.rec_input_width  = model_cfg.get('input_width',  256)
        self.rec_max_width    = model_cfg.get('max_width',    512)
        self.rec_max_len      = inf_cfg.get('max_length', 100)
        self.variable_width   = inf_cfg.get('variable_width', True)

        # --- Spell Checkers (bir tane per desteklenen dil) ---
        spell_cfg = self.config.get('postprocessing', {}).get('spell_check', {})
        if spell_cfg.get('enabled', False):
            default_lang = spell_cfg.get('language', 'tr')
            max_ed = spell_cfg.get('max_edit_distance', 2)
            self._spell_checkers: Dict[str, SpellChecker] = {
                'tr': SpellChecker(language='tr', max_edit_distance=max_ed),
                'en': SpellChecker(language='en', max_edit_distance=max_ed),
                'both': SpellChecker(language='both', max_edit_distance=max_ed),
            }
            self._default_spell_lang = default_lang
        else:
            self._spell_checkers = {}
            self._default_spell_lang = 'tr'

        # Eski tek-dilli erisim icin uyumluluk yardimcisi
        self._spell_checker: Optional[SpellChecker] = (
            self._spell_checkers.get(self._default_spell_lang)
            if self._spell_checkers else None
        )
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        detect_only: bool = False,
        recognize_only: bool = False,
        boxes: Optional[List[np.ndarray]] = None,
        spell_check: Optional[bool] = None,
        language: str = 'tr',
    ) -> OCRResult:
        """
        Gorseldeki metni tanir

        Args:
            image: Gorsel yolu veya numpy array
            detect_only: Sadece tespit yap, tanima yapma
            recognize_only: Sadece tanima yap (boxes gerekli)
            boxes: Onceden tespit edilmis kutular (recognize_only icin)
            spell_check: Yazim duzeltmeyi zorla ac/kapat (None = config'e gore)
            language: Dil kodu ('tr' veya 'en') — spell checker'a iletilir

        Returns:
            OCRResult nesnesi
        """
        import time
        start_time = time.time()
        
        # Gorseli yukle
        if isinstance(image, (str, Path)):
            image = self.image_processor.load_image(str(image))
        
        original_image = image.copy()
        original_size = image.shape[:2]  # (height, width)
        
        # Preprocessing — detection ve recognition ayni goruntu uzerinde calissin
        image = self._preprocess(image)
        preprocessed_image = image  # hem detection hem recognition bunu kullanir
        
        # Detection
        if not recognize_only:
            boxes = self._detect(preprocessed_image)
            # Adaptif siralama (satir gruplama ile)
            boxes = adaptive_sort_boxes(boxes)
        
        if detect_only or boxes is None or len(boxes) == 0:
            processing_time = time.time() - start_time
            text_boxes = [TextBox(box=box) for box in (boxes or [])]
            return OCRResult(
                text_boxes=text_boxes,
                processing_time=processing_time,
                source_image=preprocessed_image,
            )
        
        # Recognition — preprocessed goruntu uzerinde (detection ile ayni)
        text_boxes = self._recognize(preprocessed_image, boxes)

        # Spell check (config veya cagiran tarafin tercihi dogrultusunda)
        apply_spell = spell_check if spell_check is not None else bool(self._spell_checkers)
        if apply_spell and self._spell_checkers:
            checker = (
                self._spell_checkers.get(language)
                or self._spell_checkers.get(self._default_spell_lang)
            )
            if checker:
                for tb in text_boxes:
                    if tb.text:
                        tb.text = checker.correct(tb.text)

        processing_time = time.time() - start_time

        # Layout analizi
        layout = None
        if self._layout_analyzer is not None and text_boxes:
            h, w = original_size
            layout = self._layout_analyzer.analyze(text_boxes, image_width=w, image_height=h)

        return OCRResult(
            text_boxes=text_boxes,
            processing_time=processing_time,
            layout=layout,
            source_image=preprocessed_image,
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Gorseli on islemden gecir"""
        # Boyutlandir
        image, _ = self.image_processor.resize_with_aspect_ratio(image)

        # Gurultu gider
        if self.denoiser is not None:
            image = self.denoiser.denoise(image)

        # Perspektif duzelt (kamera fotolari icin)
        if self.perspective_corrector is not None:
            image = self.perspective_corrector.correct(image)

        # Aci duzelt
        if self.deskewer is not None:
            image, _ = self.deskewer.deskew(image)

        # Goruntu iyilestirme
        if self.enhancer is not None:
            if self._enhance_mode == 'auto':
                quality = self.enhancer.measure_quality(image)
                if quality['sharpness'] < self._enhance_sharpness_threshold:
                    image = self.enhancer.prepare_for_handwriting(image)
                elif quality['score'] < self._enhance_quality_threshold:
                    image = self.enhancer.prepare_for_scan(image)
                else:
                    image = self.enhancer.enhance(image)
            elif self._enhance_mode == 'document':
                image = self.enhancer.prepare_for_scan(image)
            elif self._enhance_mode == 'handwriting':
                image = self.enhancer.prepare_for_handwriting(image)
            # 'none' -> atla

        return image
    
    def _detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Metin bolgelerini tespit et"""

        # ── EasyOCR CRAFT backend (detection-only, recognition bizim CRNN) ──────
        if self.detection_model == 'easy':
            # mag_ratio>1 kucuk metni buyutup yakalar
            # text_threshold/link_threshold dusurulurse daha fazla aday cikar
            horizontal_list, free_list = self._easy_ocr.detect(
                image,
                min_size=10,
                text_threshold=0.6,
                low_text=0.35,
                link_threshold=0.3,
                canvas_size=2560,
                mag_ratio=1.5,
                slope_ths=0.2,
                ycenter_ths=0.5,
                height_ths=0.5,
                width_ths=0.6,
                add_margin=0.1,
            )
            boxes = []
            if horizontal_list:
                for (x_min, x_max, y_min, y_max) in horizontal_list[0]:
                    pts = np.array([
                        [x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max],
                    ], dtype=np.float32)
                    boxes.append(pts)
            if free_list:
                for pts in free_list[0]:
                    boxes.append(np.array(pts, dtype=np.float32))
            return boxes

        # ── DBNet fallback ─────────────────────────────────────────
        original_size = image.shape[:2]
        det_cfg = self.config.get('detection', {})
        det_size = det_cfg.get('input_size', [640, 640])
        resized = cv2.resize(image, (det_size[0], det_size[1]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = self.image_processor.normalize(rgb)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.detection_model(tensor)
        prob_map = outputs['prob_map'][0, 0].cpu().numpy()
        boxes = self.detection_postprocessor(prob_map, original_size)
        return boxes
    
    @torch.no_grad()
    def _recognize(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        batch_size: int = 32
    ) -> List[TextBox]:
        """
        Tespit edilen bolgelerdeki metni tanir.
        Her satir once kelimelerine ayrilir; her kelime ayri ayri
        CRNN'den gecirilir — boylece CTC'ye giren dizi cok daha kisa
        kalir ve dogruluk onemli olcude artar.
        """
        from .detection.postprocess import correct_box_rotation

        if len(boxes) == 0:
            return []

        # 1. Her satir kutusunu kelime crop'larina bol
        # word_items: (line_idx, word_poly, resized_gray)
        word_items: List[Tuple[int, np.ndarray, np.ndarray]] = []

        for i, box in enumerate(boxes):
            try:
                crop, _ = correct_box_rotation(image, box, angle_threshold=5.0)
                if crop.size == 0:
                    continue

                gray = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        if len(crop.shape) == 3 else crop.copy())

                # Kelimelerine bol (orijinal cozunurlukta, kaliteli projeksiyon icin)
                word_pairs = self._split_line_to_words(gray, box)

                for word_gray, word_poly in word_pairs:
                    h, w = word_gray.shape[:2]
                    if h == 0 or w == 0:
                        continue
                    scale   = self.rec_input_height / h
                    new_w   = max(1, int(w * scale))
                    resized = cv2.resize(word_gray, (new_w, self.rec_input_height))
                    word_items.append((i, word_poly, resized))

            except Exception:
                pass

        if not word_items:
            return [TextBox(box=box, text="", confidence=0.0) for box in boxes]

        # 2. Toplu cikarim — kelime birimleri uzerinden
        line_word_texts: List[List[str]]       = [[] for _ in range(len(boxes))]
        line_word_confs: List[List[float]]     = [[] for _ in range(len(boxes))]
        line_word_polys: List[List[np.ndarray]]= [[] for _ in range(len(boxes))]

        for batch_start in range(0, len(word_items), batch_size):
            batch       = word_items[batch_start:batch_start + batch_size]
            batch_idx   = [it[0] for it in batch]
            batch_polys = [it[1] for it in batch]
            batch_crops = [it[2] for it in batch]

            target_width = min(
                max(c.shape[1] for c in batch_crops),
                self.rec_max_width if self.variable_width else self.rec_input_width
            )

            batch_tensors = []
            for crop in batch_crops:
                h, w = crop.shape[:2]
                if w > target_width:
                    crop = crop[:, :target_width]
                elif w < target_width:
                    padded = np.zeros((h, target_width), dtype=crop.dtype)
                    padded[:, :w] = crop
                    crop = padded
                batch_tensors.append(crop.astype(np.float32) / 255.0)

            batch_tensor = np.stack(batch_tensors, axis=0)
            batch_tensor = torch.from_numpy(batch_tensor).unsqueeze(1).to(self.device)

            # --- Attention modu ---
            if self.recognition_mode == 'attention':
                char_indices, _ = self.recognition_model.predict(
                    batch_tensor, max_len=self.rec_max_len
                )
                texts       = self._attn_decoder.batch_indices_to_texts(char_indices)
                confidences = np.full(len(batch_crops), 0.9, dtype=np.float32)

            # --- CTC modu ---
            else:
                log_probs = self.recognition_model(batch_tensor)

                if self._prefix_decoder is not None:
                    prefix_results = self._prefix_decoder.decode_batch(log_probs)
                    texts          = [t for t, _ in prefix_results]
                    confidences    = np.array(
                        [float(np.clip(np.exp(s), 0.0, 1.0)) for _, s in prefix_results],
                        dtype=np.float32
                    )
                else:
                    texts = self.decoder.decode_greedy(log_probs)
                    with torch.no_grad():
                        probs = torch.exp(log_probs)
                        mp, mi   = probs.max(dim=2)
                        mp_b = mp.permute(1, 0).cpu().numpy()
                        mi_b = mi.permute(1, 0).cpu().numpy()
                    confidences = self._compute_confidence(mp_b, mi_b, self.vocab.blank_idx)

            for j, (line_idx, poly) in enumerate(zip(batch_idx, batch_polys)):
                line_word_texts[line_idx].append(texts[j])
                line_word_confs[line_idx].append(float(confidences[j]))
                line_word_polys[line_idx].append(poly)

        # 3. Kelime duzeyinde TextBox'lar olustur
        result: List[TextBox] = []
        for i, box in enumerate(boxes):
            words = line_word_texts[i]
            confs = line_word_confs[i]
            polys = line_word_polys[i]

            if not words:
                result.append(TextBox(box=box, text="", confidence=0.0))
                continue

            for word, conf, poly in zip(words, confs, polys):
                result.append(TextBox(box=poly, text=word, confidence=conf))

        return result
    
    def _split_line_to_words(
        self,
        gray: np.ndarray,
        box: np.ndarray,
        min_gap_ratio: float = 0.015,
        min_word_ratio: float = 0.025,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Tek satirlik grayscale crop'u dikey projeksiyon ile kelimelerine boler.

        Args:
            gray:           Orijinal cozunurlukta grayscale line crop
            box:            [4, 2] polygon (gorsel koordinatlari)
            min_gap_ratio:  Minimum bosluk genisligi (crop.width orani)
            min_word_ratio: Minimum kelime genisligi (crop.width orani)

        Returns:
            [(word_gray, word_polygon), ...]  —  bolunemezse [(gray, box)]
        """
        h, w = gray.shape[:2]
        if w == 0 or h == 0:
            return [(gray, box)]

        min_gap_px  = max(2, int(w * min_gap_ratio))
        min_word_px = max(4, int(w * min_word_ratio))

        # Otsu binarize: metin koyu (0), arka plan acik (255)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dikey projeksiyon: kolon basina koyu piksel sayisi
        ink_col = (255 - binary).sum(axis=0).astype(np.float32)

        # Yumusat — gurultuyu azalt
        k = max(3, min(w // 20, 11))
        if k % 2 == 0:
            k += 1
        ink_smooth = cv2.GaussianBlur(
            ink_col.reshape(1, -1).astype(np.float32), (k, 1), 0
        ).flatten()

        max_ink = ink_smooth.max()
        if max_ink < 1.0:
            return [(gray, box)]  # tamamen bos

        # Bosluk: maksimumun %5'inin altindaki kolonlar
        is_gap = ink_smooth <= max_ink * 0.05

        # Kelime segmentlerini bul (art arda gelen non-gap kolonlar)
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

        # Cok yakın segmentleri birlestir (minimum bosluk yorumu)
        merged: List[Tuple[int, int]] = [segments[0]]
        for s, e in segments[1:]:
            gap = s - merged[-1][1]
            if gap < min_gap_px:
                merged[-1] = (merged[-1][0], e)  # birlestir
            else:
                merged.append((s, e))

        if len(merged) < 2:
            return [(gray, box)]

        # Crop kolon koordinatlarini gorsel koordinatlara donustur
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

    @staticmethod
    def _compute_confidence(
        max_probs: np.ndarray,
        max_idx: np.ndarray,
        blank_idx: int = 0
    ) -> np.ndarray:
        """
        Duzeltilmis guven skoru: blank ve tekrar pozisyonlari dahil etmez.

        Args:
            max_probs: [B, T] float array - her pozisyon icin max class prob
            max_idx:   [B, T] int array   - argmax sinif indisi
            blank_idx: CTC blank token indexi (vocab'dan alinmali)

        Returns:
            [B] float array - her ornek icin guven skoru
        """
        B, T = max_probs.shape
        confidences = np.zeros(B, dtype=np.float32)
        blank = blank_idx

        for b in range(B):
            non_blank_probs = []
            prev_idx = -1
            for t in range(T):
                idx = int(max_idx[b, t])
                if idx != blank and idx != prev_idx:
                    non_blank_probs.append(float(max_probs[b, t]))
                prev_idx = idx
            confidences[b] = float(np.mean(non_blank_probs)) if non_blank_probs else 0.0

        return confidences

    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        result: OCRResult,
        show_text: bool = True,
        font_scale: float = 0.5,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Sonuclari gorsel uzerinde goster
        
        Args:
            image: Gorsel
            result: OCR sonucu
            show_text: Metni goster
            font_scale: Font boyutu
            thickness: Cizgi kalinligi
            
        Returns:
            Sonuclari gosterilen gorsel
        """
        if isinstance(image, (str, Path)):
            image = self.image_processor.load_image(str(image))
        
        vis = image.copy()
        
        for tb in result.text_boxes:
            # Kutuyu ciz
            pts = tb.box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), thickness)
            
            # Metni yaz
            if show_text and tb.text:
                x, y = tb.x1, tb.y1 - 5
                if y < 15:
                    y = tb.y2 + 15
                
                # Arka plan
                (w, h), _ = cv2.getTextSize(
                    tb.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(vis, (x, y - h - 5), (x + w + 5, y + 5), (0, 255, 0), -1)
                
                # Metin
                cv2.putText(
                    vis, tb.text,
                    (x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    1
                )
        
        return vis
    
    def __call__(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> OCRResult:
        """Kisayol: recognize() metodunu cagir"""
        return self.recognize(image)


def create_pipeline(
    config_path: Optional[str] = None,
    device: Optional[str] = None
) -> OCRPipeline:
    """
    OCR pipeline olustur
    
    Args:
        config_path: Yapilandirma dosyasi
        device: Cihaz (cuda/cpu)
        
    Returns:
        OCRPipeline nesnesi
    """
    return OCRPipeline(config_path=config_path, device=device)