import torch

from .paddle_worker import PaddleDetector
from .model import DBNet
from .postprocess import DBPostProcessor


def build_detector(config: dict, device: torch.device):
    """PaddleOCR yükle, başarısız olursa DBNet'e dön.

    Returns (mode, paddle_detector, dbnet_model, dbnet_postprocessor)
    mode == 'paddle' → paddle_detector dolu, dbnet_* None
    mode == 'dbnet'  → dbnet_* dolu, paddle_detector None
    """
    det_cfg = config.get('detection', {})
    try:
        paddle = PaddleDetector()
        print("Detection: PaddleOCR yuklendi.")
        return 'paddle', paddle, None, None
    except Exception as e:
        print(f"[UYARI] PaddleOCR yuklenemedi ({e}), DBNet'e geri donuluyor.")
        model_cfg = det_cfg.get('model', {})
        model = DBNet(
            backbone=model_cfg.get('backbone', 'resnet18'),
            pretrained=model_cfg.get('pretrained', True),
        ).to(device)
        model.eval()
        inf_cfg = det_cfg.get('inference', {})
        postproc = DBPostProcessor(
            threshold=inf_cfg.get('threshold', 0.3),
            box_threshold=inf_cfg.get('box_threshold', 0.5),
            max_candidates=inf_cfg.get('max_candidates', 1000),
            unclip_ratio=inf_cfg.get('unclip_ratio', 1.5),
        )
        return 'dbnet', None, model, postproc
