"""OCR Pipeline - detection + recognition entegrasyonu"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from pathlib import Path

from .config import load_config
from .pipeline_types import TextBox, OCRResult
from .preprocessing import Preprocessor
from .detection.postprocess import adaptive_sort_boxes
from .detection.builder import build_detector
from .recognition.builder import build_recognition
from .recognition.crop import split_line_to_words, compute_ctc_confidence


class OCRPipeline:
    """Ana OCR pipeline (detection + recognition)."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        detection_weights: Optional[str] = None,
        recognition_weights: Optional[str] = None
    ):
        self.config = load_config(config_path)
        if device is None:
            device = self.config.get('general', {}).get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA mevcut degil, CPU kullaniliyor.")
            device = 'cpu'
        self.device = torch.device(device)
        self.preprocessor = Preprocessor(self.config)
        self._init_detection(detection_weights)
        self._init_recognition(recognition_weights)
    
    def _init_detection(self, weights_path: Optional[str]):
        mode, paddle, dbnet, postproc = build_detector(self.config, self.device)
        self.detection_model  = mode if mode == 'paddle' else dbnet
        self._paddle_detector = paddle
        if postproc is not None:
            self.detection_postprocessor = postproc
    
    def _init_recognition(self, weights_path: Optional[str]):
        bundle = build_recognition(self.config, self.device, weights_path)
        self.recognition_mode    = bundle.mode
        self.recognition_model   = bundle.model
        self.vocab               = bundle.vocab
        self.decoder             = bundle.decoder
        self._prefix_decoder     = bundle.prefix_decoder
        self._attn_decoder       = bundle.attn_decoder
        self._layout_analyzer    = bundle.layout_analyzer
        self._spell_checkers     = bundle.spell_checkers
        self._default_spell_lang = bundle.default_spell_lang
        self._spell_checker      = bundle.spell_checker
        self.rec_input_height    = bundle.input_height
        self.rec_input_width     = bundle.input_width
        self.rec_max_width       = bundle.max_width
        self.rec_max_len         = bundle.max_len
        self.variable_width      = bundle.variable_width
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        detect_only: bool = False,
        recognize_only: bool = False,
        boxes: Optional[List[np.ndarray]] = None,
        spell_check: Optional[bool] = None,
        language: str = 'tr',
    ) -> OCRResult:
        import time
        start_time = time.time()
        image = self.preprocessor.load(image)
        original_size = image.shape[:2]
        image = self.preprocessor.process(image)
        if not recognize_only:
            boxes = self._detect(image)
            boxes = adaptive_sort_boxes(boxes)
        
        if detect_only or boxes is None or len(boxes) == 0:
            processing_time = time.time() - start_time
            return OCRResult(
                text_boxes=[TextBox(box=box) for box in (boxes or [])],
                processing_time=processing_time,
                source_image=image,
            )
        
        text_boxes = self._recognize(image, boxes)
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
            source_image=image,
        )
    
    def _detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Metin bolgelerini tespit et"""
        if self.detection_model == 'paddle':
            return self._paddle_detector.detect(image)

        # DBNet fallback
        original_size = image.shape[:2]
        det_cfg = self.config.get('detection', {})
        det_size = det_cfg.get('input_size', [640, 640])
        resized = cv2.resize(image, (det_size[0], det_size[1]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = self.preprocessor.image_processor.normalize(rgb)
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
        from .detection.postprocess import correct_box_rotation

        if len(boxes) == 0:
            return []

        # 1. Her satir kutusunu kelime crop'larina bol
        word_items: List[Tuple[int, np.ndarray, np.ndarray]] = []

        for i, box in enumerate(boxes):
            try:
                crop, _ = correct_box_rotation(image, box, angle_threshold=5.0)
                if crop.size == 0:
                    continue

                gray = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        if len(crop.shape) == 3 else crop.copy())

                word_pairs = split_line_to_words(gray, box)

                for word_gray, word_poly in word_pairs:
                    h, w = word_gray.shape[:2]
                    if h == 0 or w == 0:
                        continue
                    # Piksel yüksekliği çok küçükse (telefon fotoğrafı vb.)
                    # önce bicubic ile 2× büyüt, sonra hedef yüksekliğe çek.
                    # Bu, ince detayların (ğ, ş, ı nokta) korunmasını sağlar.
                    if h < 20:
                        word_gray = cv2.resize(
                            word_gray,
                            (max(1, w * 2), h * 2),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        h, w = word_gray.shape[:2]
                    scale   = self.rec_input_height / h
                    new_w   = max(1, int(w * scale))
                    resized = cv2.resize(word_gray, (new_w, self.rec_input_height),
                                         interpolation=cv2.INTER_CUBIC)
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
                    probs = torch.exp(log_probs)
                    mp, mi = probs.max(dim=2)
                    confidences = compute_ctc_confidence(
                        mp.permute(1, 0).cpu().numpy(),
                        mi.permute(1, 0).cpu().numpy(),
                        self.vocab.blank_idx
                    )

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
    
    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        result: OCRResult,
        show_text: bool = True,
        font_scale: float = 0.5,
        thickness: int = 2
    ) -> np.ndarray:
        image = self.preprocessor.load(image)
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
    return OCRPipeline(config_path=config_path, device=device)