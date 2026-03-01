"""
AttentionCRNN Egitim Scripti

El yazisi ve egri metinler icin Attention tabanli OCR modelini egitir.

Kullanim:
    python training/train_attention.py \
        --data_root DataSets/mnt/ramdisk/max/90kDICT32px \
        --train_json data/mjsynth_train.json \
        --config config.yaml \
        --epochs 80 \
        --device cuda
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine.recognition.vocab import Vocabulary
from ocr_engine.recognition.attention import (
    AttentionCRNN,
    AttentionLoss,
    AttentionDecodeHelper,
    build_attention_crnn,
)
from training.recognition_dataset import RecognitionDataset, collate_recognition
from training.augmentation_recognition import RecognitionAugmentor


# ---------------------------------------------------------------------------
# Dataset wrapper: SOS/EOS destekli collate
# ---------------------------------------------------------------------------

def collate_attention(batch, sos_idx: int, eos_idx: int, pad_idx: int = 0):
    """
    Attention decoder icin SOS/EOS ile padding uygular.

    Girdi batch'i: collate_recognition ciktisina benzer yapida.
    Her 'labels' tensoru SOS ile baslatilir, EOS ile bitirilir, padding yapilir.
    """
    images      = torch.stack([b['image'] for b in batch])
    texts       = [b['text']         for b in batch]
    label_lists = [b['label']        for b in batch]    # List[List[int]]

    # SOS + label + EOS
    targets_with_se = []
    for lab in label_lists:
        seq = [sos_idx] + list(lab) + [eos_idx]
        targets_with_se.append(seq)

    max_len = max(len(s) for s in targets_with_se)
    padded  = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, seq in enumerate(targets_with_se):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        lengths[i] = len(seq)

    return {
        'images':         images,
        'targets':        padded,
        'target_lengths': lengths,
        'texts':          texts,
    }


# ---------------------------------------------------------------------------
# Dataset: lab listesi dondurecek sekilde wrap
# ---------------------------------------------------------------------------

class AttentionRecognitionDataset(RecognitionDataset):
    """RecognitionDataset'e label listesi cikarma ekler."""

    def __getitem__(self, idx: int) -> dict:
        item  = super().__getitem__(idx)          # returns {'image', 'label', 'label_length', 'text'}
        text  = item['text']
        label_len = int(item['label_length'])
        label = item['label'][:label_len].tolist()
        return {
            'image': item['image'],
            'text':  text,
            'label': label,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AttentionTrainer:

    def __init__(
        self,
        config: dict,
        vocab: Vocabulary,
        device: str = 'cuda',
        teacher_forcing_ratio: float = 0.5,
    ):
        self.config   = config
        self.vocab    = vocab
        self.device   = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tf_ratio = teacher_forcing_ratio

        rec_cfg   = config.get('recognition', {})
        attn_cfg  = rec_cfg.get('attention', {})
        model_cfg = rec_cfg.get('model', {})

        self.model = build_attention_crnn(
            num_classes   = vocab.size,
            input_channels= 1,
            hidden_size   = model_cfg.get('hidden_size', 256),
            num_layers    = model_cfg.get('num_layers', 2),
            attn_dim      = attn_cfg.get('attn_dim', 256),
            dropout       = model_cfg.get('dropout', 0.1),
            encoder_type  = 'vgg',
            sos_idx       = vocab.sos_idx,
            eos_idx       = vocab.eos_idx,
        ).to(self.device)

        self.criterion = AttentionLoss(
            pad_idx=vocab.blank_idx,
            label_smoothing=0.1
        )

        self.decode_helper = AttentionDecodeHelper(
            vocab,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx
        )

        train_cfg  = config.get('training', {}).get('recognition', {})
        self.optim = AdamW(
            self.model.parameters(),
            lr=train_cfg.get('learning_rate', 5e-4),
            weight_decay=1e-4
        )

        self.best_acc   = 0.0
        self.epoch      = 0
        self.scheduler  = None

    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        save_dir: str,
    ):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # OneCycleLR: warm-up + cosine annealing
        self.scheduler = OneCycleLR(
            self.optim,
            max_lr=1e-3,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1,
        )

        for ep in range(1, epochs + 1):
            self.epoch = ep
            train_metrics = self._train_epoch(train_loader)

            print(
                f"Epoch {ep}/{epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"acc={train_metrics['acc']:.3f}"
            )

            if val_loader is not None and ep % 5 == 0:
                val_metrics = self._val_epoch(val_loader)
                print(f"  [val]  loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.3f}")

                if val_metrics['acc'] > self.best_acc:
                    self.best_acc = val_metrics['acc']
                    self._save(save_path / 'best_attention.pth')
                    print(f"  [best] acc={self.best_acc:.3f}  -> kaydedildi")

            # Periyodik kayit
            if ep % 10 == 0:
                self._save(save_path / f'attention_ep{ep:03d}.pth')

        # Son model
        self._save(save_path / 'attention_final.pth')
        print("Egitim tamamlandi.")

    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        n_correct  = 0
        n_total    = 0

        pbar = tqdm(loader, desc=f"Epoch {self.epoch} [train]", leave=False)
        for batch in pbar:
            images   = batch['images'].to(self.device)
            targets  = batch['targets'].to(self.device)
            t_lens   = batch['target_lengths'].to(self.device)
            texts    = batch['texts']

            # Decoder girdisi: SOS dahil, EOS haric (hedef shift)
            dec_input  = targets[:, :-1]   # [B, T-1]
            dec_target = targets[:, 1:]    # [B, T-1]
            dec_lens   = (t_lens - 1).clamp(min=1)

            self.optim.zero_grad()
            logits, _ = self.model(
                images,
                targets     = dec_input,
                target_lengths = dec_lens,
                teacher_forcing_ratio = self.tf_ratio
            )

            loss = self.criterion(logits, dec_target, dec_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optim.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            # Dogruluk (char indices -> metin)
            with torch.no_grad():
                pred_idx = logits.argmax(-1)   # [B, T]
                preds = self.decode_helper.batch_indices_to_texts(pred_idx)
                for p, t in zip(preds, texts):
                    if p == t:
                        n_correct += 1
                    n_total += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            'loss': total_loss / max(len(loader), 1),
            'acc':  n_correct  / max(n_total, 1),
        }

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        n_correct  = 0
        n_total    = 0

        for batch in loader:
            images   = batch['images'].to(self.device)
            targets  = batch['targets'].to(self.device)
            t_lens   = batch['target_lengths'].to(self.device)
            texts    = batch['texts']

            dec_input  = targets[:, :-1]
            dec_target = targets[:, 1:]
            dec_lens   = (t_lens - 1).clamp(min=1)

            logits, _ = self.model(
                images,
                targets            = dec_input,
                target_lengths     = dec_lens,
                teacher_forcing_ratio = 0.0   # val'de teacher forcing yok
            )

            loss = self.criterion(logits, dec_target, dec_lens)
            total_loss += loss.item()

            pred_idx = logits.argmax(-1)
            preds = self.decode_helper.batch_indices_to_texts(pred_idx)
            for p, t in zip(preds, texts):
                if p == t:
                    n_correct += 1
                n_total += 1

        return {
            'loss': total_loss / max(len(loader), 1),
            'acc':  n_correct  / max(n_total, 1),
        }

    def _save(self, path: Path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim':            self.optim.state_dict(),
            'epoch':            self.epoch,
            'best_acc':         self.best_acc,
            'vocab_size':       self.vocab.size,
        }, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AttentionCRNN Egitimi')
    parser.add_argument('--data_root',  required=True,  help='Gorsel klasoru')
    parser.add_argument('--train_json', required=True,  help='Train annotation JSON')
    parser.add_argument('--val_json',   default=None,   help='Validation JSON (opsiyonel)')
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--epochs',     type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--save_dir',   default='checkpoints/attention')
    parser.add_argument('--resume',     default=None,   help='Checkpoint devam')
    parser.add_argument('--handwriting_mode', action='store_true',
                        help='El yazisi modunda augmentation kullan')
    args = parser.parse_args()

    print("=" * 60)
    print("AttentionCRNN Egitimi")
    print("=" * 60)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # SOS/EOS iceren vocabulary
    vocab = Vocabulary(include_sos_eos=True)
    print(f"Vocabulary: {vocab.size} sinif  (SOS={vocab.sos_idx}, EOS={vocab.eos_idx})")

    rec_cfg = config.get('recognition', {}).get('model', {})

    aug_cfg = config.get('training', {}).get('augmentation', {})
    augmentor = RecognitionAugmentor(
        noise_prob         = aug_cfg.get('noise_prob', 0.3),
        blur_prob          = aug_cfg.get('blur_prob', 0.3),
        elastic_prob       = aug_cfg.get('elastic_prob', 0.3),
        grid_distort_prob  = aug_cfg.get('grid_distort_prob', 0.2),
        handwriting_mode   = args.handwriting_mode or aug_cfg.get('handwriting_mode', False),
    ) if aug_cfg.get('enabled', True) else None

    train_ds = AttentionRecognitionDataset(
        data_dir        = args.data_root,
        annotation_file = args.train_json,
        vocab           = vocab,
        image_height    = rec_cfg.get('input_height', 32),
        image_width     = rec_cfg.get('input_width',  256),
        augmentor       = augmentor,
        synthetic_ratio = 0.0,
    )

    def make_collate(v):
        return lambda b: collate_attention(b, v.sos_idx, v.eos_idx, v.blank_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = make_collate(vocab),
        pin_memory  = True,
    )
    print(f"Train: {len(train_ds)} ornek")

    val_loader = None
    if args.val_json:
        val_ds = AttentionRecognitionDataset(
            data_dir        = args.data_root,
            annotation_file = args.val_json,
            vocab           = vocab,
            image_height    = rec_cfg.get('input_height', 32),
            image_width     = rec_cfg.get('input_width',  256),
            augmentor       = None,
            synthetic_ratio = 0.0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = 0,
            collate_fn  = make_collate(vocab),
        )
        print(f"Val:   {len(val_ds)} ornek")

    trainer = AttentionTrainer(config, vocab, device=args.device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model', ckpt)))
        trainer.optim.load_state_dict(ckpt['optim'])
        trainer.epoch    = ckpt.get('epoch', 0)
        trainer.best_acc = ckpt.get('best_acc', 0.0)
        print(f"Checkpoint yuklendi: epoch={trainer.epoch}  best_acc={trainer.best_acc:.3f}")

    trainer.train(train_loader, val_loader, args.epochs, args.save_dir)


if __name__ == '__main__':
    main()
