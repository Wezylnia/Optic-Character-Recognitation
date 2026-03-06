"""
training/train.py — Birleşik OCR tanıma eğitim giriş noktası.

Kullanım örnekleri::

    # Yeni eğitim (CTC)
    python training/train.py --train_json data/mjsynth_train.json --epochs 80

    # Checkpoint'ten devam
    python training/train.py --train_json data/real_world_train.json \\
                              --val_json   data/real_world_test.json  \\
                              --resume     checkpoints/stage1_full/best_model.pth \\
                              --epochs 120 --lr 5e-5 --augment

    # Attention modu
    python training/train.py --train_json data/mjsynth_train.json --mode attention
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ocr_engine.recognition.vocab import Vocabulary
from training.augment import RecognitionAugmentor
from training.dataset import RecognitionDataset, collate_attention, collate_recognition
from training.trainer import RecognitionTrainer


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_hardware():
    """CPU affinity, CUDA ve thread optimizasyonları."""
    try:
        import psutil
        psutil.Process(os.getpid()).cpu_affinity(list(range(14)))
        print("[CPU] Affinity: 0-13 (P-core 0-7, E-core 8-13) | 14,15 serbest")
    except Exception as e:
        print(f"[CPU] Affinity ayarlanamadı: {e}")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[CUDA] {torch.cuda.get_device_name(0)} | "
              f"cudnn.benchmark=True, tf32=True")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCR Tanıma Modeli Eğitimi (CTC / Attention)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Veri
    p.add_argument("--train_json",   required=True,       help="Eğitim JSON dosyası")
    p.add_argument("--val_json",     default=None,        help="Validasyon JSON (opsiyonel)")
    p.add_argument("--val_split",    type=float, default=0.05,
                   help="val_json yoksa train'den ayrılacak oran")
    p.add_argument("--data_root",    default=None,        help="Görsel kök klasörü (JSON mutlak ise gerekli değil)")
    # Eğitim
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=96)
    p.add_argument("--lr",           type=float, default=None,
                   help="LR override (checkpoint LR'sini ezer)")
    p.add_argument("--num_workers",  type=int,   default=None)
    p.add_argument("--augment",      action="store_true", default=False)
    p.add_argument("--mode",         choices=["ctc", "attention"], default="ctc")
    # Checkpoint
    p.add_argument("--resume",       default=None,        help="Devam edilecek checkpoint")
    p.add_argument("--save_dir",     default="checkpoints/run")
    p.add_argument("--reset-best-acc", action="store_true", default=False)
    # Diğer
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--quiet",        action="store_true", default=False,
                   help="tqdm'siz mod (CI / pipe çıktı)")
    p.add_argument("--device",       default="cuda")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    setup_hardware()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("  OCR RECOGNITION — EĞİTİM")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # Config ──────────────────────────────────────────────────────────
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # config'deki modu CLI argümanıyla ezmek isteyebiliriz
    config.setdefault("recognition", {})["mode"] = args.mode

    # Vocabulary ──────────────────────────────────────────────────────
    vocab = Vocabulary()
    print(f"[VOCAB] Boyut: {vocab.size}")

    # Augmentor ───────────────────────────────────────────────────────
    augmentor = RecognitionAugmentor() if args.augment else None
    print(f"[AUGMENT] {'Aktif' if augmentor else 'Kapalı'}")

    # Dataset boyutları ───────────────────────────────────────────────
    rec_cfg = config.get("recognition", {}).get("model", {})
    image_h  = rec_cfg.get("input_height", 32)
    image_w  = rec_cfg.get("input_width",  256)

    def make_dataset(json_path, with_aug=False):
        return RecognitionDataset(
            data_dir      = args.data_root,
            annotation_file=json_path,
            vocab         = vocab,
            image_height  = image_h,
            image_width   = image_w,
            augmentor     = augmentor if with_aug else None,
            synthetic_ratio=0.0,
        )

    print(f"[DATA] Train: {args.train_json}")
    train_ds = make_dataset(args.train_json, with_aug=True)

    if args.val_json:
        print(f"[DATA] Val  : {args.val_json}")
        val_ds = make_dataset(args.val_json, with_aug=False)
    else:
        from torch.utils.data import random_split
        val_size  = int(len(train_ds) * args.val_split)
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    print(f"[SPLIT] Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Collate function ────────────────────────────────────────────────
    if args.mode == "attention":
        collate_fn = collate_attention(vocab.sos_idx, vocab.eos_idx)
    else:
        collate_fn = collate_recognition

    nw  = args.num_workers if args.num_workers is not None else (4 if torch.cuda.is_available() else 0)
    pf  = 4 if nw > 0 else None
    pw  = nw > 0

    train_loader = DataLoader(
        train_ds,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = nw,
        collate_fn   = collate_fn,
        pin_memory   = True,
        persistent_workers = pw,
        prefetch_factor    = pf,
        drop_last    = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size   = args.batch_size * 2,
        shuffle      = False,
        num_workers  = nw,
        collate_fn   = collate_fn,
        pin_memory   = True,
        persistent_workers = pw,
        prefetch_factor    = pf,
    )

    # Trainer ─────────────────────────────────────────────────────────
    trainer = RecognitionTrainer(config, vocab, device=args.device)

    if args.resume:
        trainer.load(
            args.resume,
            reset_best_acc = args.reset_best_acc,
            lr_override    = args.lr,
        )
    elif args.lr is not None:
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr

    # Eğitim ──────────────────────────────────────────────────────────
    trainer.train(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = args.epochs,
        save_dir     = str(ROOT / args.save_dir),
    )


if __name__ == "__main__":
    main()
