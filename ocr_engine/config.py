"""Config yukleme yardimcisi"""

from pathlib import Path
import yaml


_DEFAULTS = {
    'general': {'device': 'cuda'},
    'preprocessing': {
        'target_size': [1280, 960],
        'denoise': {'enabled': True, 'method': 'bilateral', 'strength': 10},
        'deskew': {'enabled': True, 'max_angle': 45},
    },
    'detection': {
        'model': {'backbone': 'resnet18'},
        'inference': {
            'threshold': 0.3,
            'box_threshold': 0.5,
            'max_candidates': 1000,
            'unclip_ratio': 1.5,
        },
    },
    'recognition': {
        'model': {
            'input_height': 48,
            'input_width': 256,
            'hidden_size': 256,
            'num_layers': 2,
        },
        'inference': {'beam_width': 5},
    },
}


def load_config(config_path=None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return _DEFAULTS
