"""
MJSynth veri seti ile model egitimi icin hazir script
"""

import sys
from pathlib import Path

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.model import CRNN, CRNNLoss
from training.dataset import RecognitionDataset, collate_recognition
from training.augmentation import RecognitionAugmentor
from training.train_recognition import RecognitionTrainer


def main():
    parser = argparse.ArgumentParser(description='MJSynth ile Recognition Egitimi')
    parser.add_argument('--data_root', type=str, required=True,
                        help='MJSynth gorsel klasoru (mnt/ramdisk/max/90kDICT32px)')
    parser.add_argument('--train_json', type=str, required=True,
                        help='Train JSON dosyasi (convert_mjsynth.py ile olusturulmus)')
    parser.add_argument('--val_json', type=str, default=None,
                        help='Validation JSON dosyasi')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config dosyasi')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch sayisi')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch boyutu')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint yolu (devam etmek icin)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/mjsynth',
                        help='Model kayit klasoru')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Cihaz (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MJSynth Recognition Egitimi")
    print("="*60)
    
    # Config yukle
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Vocabulary
    vocab = Vocabulary()
    print(f"Vocabulary boyutu: {vocab.size}")
    
    # Dataset
    rec_cfg = config.get('recognition', {}).get('model', {})
    
    print(f"\nTrain veri seti yukleniyor: {args.train_json}")
    train_dataset = RecognitionDataset(
        data_dir=args.data_root,
        annotation_file=args.train_json,
        vocab=vocab,
        image_height=rec_cfg.get('input_height', 32),
        image_width=rec_cfg.get('input_width', 256),
        augmentor=None,  # Ilk egitimde augmentation yok (hizli test)
        synthetic_ratio=0.0  # Tamamen gerçek veri
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows icin 0 (OpenCV worker hatasi)
        collate_fn=collate_recognition,
        pin_memory=True
    )
    
    print(f"[OK] Train: {len(train_dataset)} ornek")
    
    # Validation
    val_loader = None
    if args.val_json:
        print(f"\nValidation veri seti yukleniyor: {args.val_json}")
        val_dataset = RecognitionDataset(
            data_dir=args.data_root,
            annotation_file=args.val_json,
            vocab=vocab,
            image_height=rec_cfg.get('input_height', 32),
            image_width=rec_cfg.get('input_width', 256),
            augmentor=None,  # Validation'da augmentation yok
            synthetic_ratio=0.0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_recognition
        )
        
        print(f"[OK] Validation: {len(val_dataset)} ornek")
    
    # Trainer
    print(f"\nTrainer olusturuluyor...")
    trainer = RecognitionTrainer(config, vocab=vocab, device=args.device)
    
    # Resume
    if args.resume:
        print(f"Checkpoint yukleniyor: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Egitim
    print(f"\nEgitim basliyor...")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save dir: {args.save_dir}")
    print()
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
