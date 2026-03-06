"""Goruntu on-isleme pipeline'i — tek noktadan baslat ve calistir"""

import numpy as np
from pathlib import Path
from typing import Union

from .image_utils import ImageProcessor
from .binarization import Binarizer
from .deskew import Deskewer
from .denoise import Denoiser
from .enhance import ImageEnhancer
from .perspective import PerspectiveCorrector


class Preprocessor:
    """Config dict'ten on-isleme bilesenlerini kurar; load() + process() sunar."""

    def __init__(self, config: dict):
        preproc_cfg = config.get('preprocessing', {})

        self.image_processor = ImageProcessor(
            target_size=tuple(preproc_cfg.get('target_size', [1280, 960]))
        )

        denoise_cfg = preproc_cfg.get('denoise', {})
        self.denoiser = Denoiser(
            method=denoise_cfg.get('method', 'bilateral'),
            strength=denoise_cfg.get('strength', 10)
        ) if denoise_cfg.get('enabled', True) else None

        deskew_cfg = preproc_cfg.get('deskew', {})
        self.deskewer = Deskewer(
            max_angle=deskew_cfg.get('max_angle', 45)
        ) if deskew_cfg.get('enabled', True) else None

        self.binarizer = Binarizer(method='adaptive')

        enh_cfg = preproc_cfg.get('enhance', {})
        if enh_cfg.get('enabled', True):
            tile = enh_cfg.get('clahe_tile_size', [8, 8])
            self.enhancer = ImageEnhancer(
                clahe_clip_limit=enh_cfg.get('clahe_clip_limit', 2.0),
                clahe_tile_size=tuple(tile),
                sharpen_strength=enh_cfg.get('sharpen_strength', 0.5),
                shadow_removal=enh_cfg.get('shadow_removal', True),
                auto_mode=(enh_cfg.get('mode', 'auto') == 'auto'),
                mode=enh_cfg.get('mode', 'auto'),
                quality_threshold=enh_cfg.get('auto_quality_threshold', 0.4),
                sharpness_threshold=enh_cfg.get('auto_sharpness_threshold', 50.0),
            )
        else:
            self.enhancer = None

        persp_cfg = preproc_cfg.get('perspective', {})
        self.perspective_corrector = PerspectiveCorrector(
            min_area_ratio=persp_cfg.get('min_area_ratio', 0.1),
            max_angle_deviation=persp_cfg.get('max_angle_deviation', 30.0),
        ) if persp_cfg.get('enabled', True) else None

    def load(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        return self.image_processor.load_image(str(image))

    def process(self, image: np.ndarray) -> np.ndarray:
        image, _ = self.image_processor.resize_with_aspect_ratio(image)
        if self.denoiser is not None:
            image = self.denoiser.denoise(image)
        if self.perspective_corrector is not None:
            image = self.perspective_corrector.correct(image)
        if self.deskewer is not None:
            image, _ = self.deskewer.deskew(image)
        if self.enhancer is not None:
            image = self.enhancer.process(image)
        return image
