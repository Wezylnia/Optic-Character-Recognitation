from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from .model import CRNN
from .attention import AttentionCRNN, AttentionDecodeHelper
from .decoder import CTCDecoder, CTCPrefixDecoder
from .vocab import Vocabulary
from ..postprocessing import LayoutAnalyzer, SpellChecker


@dataclass
class RecognitionBundle:
    model: object
    vocab: Vocabulary
    mode: str
    decoder: CTCDecoder
    prefix_decoder: Optional[object]
    attn_decoder: Optional[object]
    layout_analyzer: Optional[object]
    spell_checkers: Dict[str, object]
    default_spell_lang: str
    spell_checker: Optional[object]
    input_height: int
    input_width: int
    max_width: int
    max_len: int
    variable_width: bool


def build_recognition(config: dict, device, weights_path=None) -> RecognitionBundle:
    rec_cfg   = config.get('recognition', {})
    model_cfg = rec_cfg.get('model', {})
    inf_cfg   = rec_cfg.get('inference', {})
    attn_cfg  = rec_cfg.get('attention', {})

    mode  = rec_cfg.get('mode', 'ctc')
    vocab = Vocabulary(include_sos_eos=(mode == 'attention'))

    if mode == 'attention':
        model = AttentionCRNN(
            num_classes    = vocab.size,
            input_channels = 1,
            hidden_size    = model_cfg.get('hidden_size', 256),
            num_layers     = model_cfg.get('num_layers', 2),
            attn_dim       = attn_cfg.get('attn_dim', 256),
            dropout        = model_cfg.get('dropout', 0.1),
            encoder_type   = 'resnet34',
            sos_idx        = vocab.sos_idx,
            eos_idx        = vocab.eos_idx,
        ).to(device)
        w = weights_path or attn_cfg.get('weights_path')
        if w and Path(w).exists():
            ckpt = torch.load(w, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model', ckpt)))
            print(f"Attention agirliklari yuklendi: {w}")
        else:
            print("[UYARI] Attention agirlik dosyasi bulunamadi, rastgele agirliklar kullaniliyor.")
        attn_decoder   = AttentionDecodeHelper(vocab, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
        prefix_decoder = None
    else:
        model = CRNN(
            num_classes    = vocab.num_classes,
            input_channels = 1,
            hidden_size    = model_cfg.get('hidden_size', 256),
            num_layers     = model_cfg.get('num_layers', 2),
            dropout        = model_cfg.get('dropout', 0.1),
        ).to(device)
        w = weights_path or rec_cfg.get('weights_path')
        if w and Path(w).exists():
            ckpt = torch.load(w, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"Recognition agirliklari yuklendi: {w}")
        else:
            print("[UYARI] CTC agirlik dosyasi bulunamadi, rastgele agirliklar kullaniliyor.")
        attn_decoder = None
        bw = inf_cfg.get('beam_width', 5)
        prefix_decoder = (
            CTCPrefixDecoder(vocab, beam_width=bw)
            if inf_cfg.get('decoder', 'prefix') == 'prefix' and bw > 1 else None
        )

    model.eval()

    layout_cfg = config.get('postprocessing', {}).get('layout', {})
    layout_analyzer = LayoutAnalyzer(
        heading_height_ratio  = layout_cfg.get('heading_height_ratio',  1.5),
        subhead_height_ratio  = layout_cfg.get('subhead_height_ratio',  1.25),
        caption_height_ratio  = layout_cfg.get('caption_height_ratio',  0.75),
        column_gap_ratio      = layout_cfg.get('column_gap_ratio',      0.04),
        max_columns           = layout_cfg.get('max_columns',           4),
        line_merge_gap_ratio  = layout_cfg.get('line_merge_gap_ratio',  0.5),
        paragraph_gap_ratio   = layout_cfg.get('paragraph_gap_ratio',   1.2),
    ) if layout_cfg.get('enabled', True) else None

    spell_cfg = config.get('postprocessing', {}).get('spell_check', {})
    if spell_cfg.get('enabled', False):
        default_lang   = spell_cfg.get('language', 'tr')
        max_ed         = spell_cfg.get('max_edit_distance', 2)
        spell_checkers: Dict[str, SpellChecker] = {
            'tr':   SpellChecker(language='tr',   max_edit_distance=max_ed),
            'en':   SpellChecker(language='en',   max_edit_distance=max_ed),
            'both': SpellChecker(language='both', max_edit_distance=max_ed),
        }
    else:
        default_lang   = 'tr'
        spell_checkers = {}

    return RecognitionBundle(
        model            = model,
        vocab            = vocab,
        mode             = mode,
        decoder          = CTCDecoder(vocab),
        prefix_decoder   = prefix_decoder,
        attn_decoder     = attn_decoder,
        layout_analyzer  = layout_analyzer,
        spell_checkers   = spell_checkers,
        default_spell_lang = default_lang,
        spell_checker    = spell_checkers.get(default_lang) if spell_checkers else None,
        input_height     = model_cfg.get('input_height', 48),
        input_width      = model_cfg.get('input_width',  256),
        max_width        = model_cfg.get('max_width',    640),
        max_len          = inf_cfg.get('max_length', 100),
        variable_width   = inf_cfg.get('variable_width', True),
    )
