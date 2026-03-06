"""
Unified Recognition Trainer — CTC ve Attention modlarını destekler.

Kullanım::

    from training.trainer import RecognitionTrainer
    trainer = RecognitionTrainer(config, vocab)
    trainer.train(train_loader, val_loader, epochs=50, save_dir="checkpoints/run1")
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocr_engine.recognition.attention import (
    AttentionCRNN,
    AttentionDecodeHelper,
    AttentionLoss,
    build_attention_crnn,
)
from ocr_engine.recognition.decoder import CTCDecoder
from ocr_engine.recognition.model import CRNN, CRNNLoss
from ocr_engine.recognition.vocab import Vocabulary


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _metrics(preds, targets):
    """Word accuracy ve character accuracy hesaplar."""
    if not preds:
        return 0.0, 0.0
    word_ok = sum(p == t for p, t in zip(preds, targets))
    char_ok = char_tot = 0
    for p, t in zip(preds, targets):
        ml = min(len(p), len(t))
        char_ok  += sum(p[i] == t[i] for i in range(ml))
        char_tot += max(len(p), len(t))
    return word_ok / len(preds), char_ok / max(char_tot, 1)


def _fmt_time(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}s {m:02d}d {s:02d}sn" if h else f"{m}d {s:02d}sn"


# ── Trainer ───────────────────────────────────────────────────────────────────

class RecognitionTrainer:
    """
    CTC ve Attention OCR modlarını destekleyen birleşik eğitici.

    config['recognition']['mode'] değerine göre:
    - ``"ctc"``       → CRNN + CTC Loss  (varsayılan)
    - ``"attention"`` → AttentionCRNN + CrossEntropy
    """

    def __init__(
        self,
        config: dict,
        vocab: Vocabulary,
        device: str = "cuda",
    ):
        self.config = config
        self.vocab  = vocab
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        self.scaler  = torch.amp.GradScaler("cuda") if self.use_amp else None

        rec_cfg   = config.get("recognition", {})
        self.mode = rec_cfg.get("mode", "ctc")
        model_cfg = rec_cfg.get("model", {})
        attn_cfg  = rec_cfg.get("attention", {})

        # ── Model ──────────────────────────────────────────────────────
        if self.mode == "attention":
            self.model = build_attention_crnn(
                num_classes    = vocab.size,
                input_channels = 1,
                hidden_size    = model_cfg.get("hidden_size", 256),
                num_layers     = model_cfg.get("num_layers", 2),
                attn_dim       = attn_cfg.get("attn_dim", 256),
                dropout        = model_cfg.get("dropout", 0.1),
                encoder_type   = "resnet34",
                sos_idx        = vocab.sos_idx,
                eos_idx        = vocab.eos_idx,
            ).to(self.device)
            self.criterion     = AttentionLoss(pad_idx=vocab.blank_idx, label_smoothing=0.1)
            self.decode_helper = AttentionDecodeHelper(
                vocab, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx
            )
            self.decoder = None
        else:
            self.model = CRNN(
                num_classes    = vocab.num_classes,
                input_channels = 1,
                hidden_size    = model_cfg.get("hidden_size", 256),
                num_layers     = model_cfg.get("num_layers", 2),
                dropout        = model_cfg.get("dropout", 0.1),
                encoder_type   = model_cfg.get("encoder_type", "resnet34"),
            ).to(self.device)
            self.criterion     = CRNNLoss(blank_idx=vocab.blank_idx)
            self.decoder       = CTCDecoder(vocab)
            self.decode_helper = None

        # ── Optimizer & durum değişkenleri ─────────────────────────────
        train_cfg     = config.get("training", {}).get("recognition", {})
        self.lr       = train_cfg.get("learning_rate", 1e-3)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.epoch        = 0
        self.start_epoch  = 0
        self.global_step  = 0
        self.best_val_acc = 0.0
        self.plateau_sched: Optional[ReduceLROnPlateau] = None
        self._pending_plateau = None

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[MODEL] {self.mode.upper()} | {n_params:,} params | "
            f"device={self.device} | AMP={self.use_amp}"
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def _ctc_forward(self, images, labels, label_lengths):
        """CTC loss hesaplar. (images, labels device'da olmalı)"""
        log_probs    = self.model(images)
        B            = images.size(0)
        input_len    = self.model.get_sequence_length(images.size(3))
        input_lengths = torch.full((B,), input_len, dtype=torch.long)
        targets_flat  = torch.cat([labels[i, :label_lengths[i]] for i in range(B)])
        loss          = self.criterion(log_probs, targets_flat, input_lengths, label_lengths)
        return loss, log_probs

    def _attn_forward(self, images, targets, target_lengths):
        """Attention loss hesaplar. targets → [B, T] (SOS dahil, EOS dahil)"""
        dec_input  = targets[:, :-1]              # [B, T-1]: SOS..label
        dec_target = targets[:, 1:]               # [B, T-1]: label..EOS
        dec_lens   = (target_lengths - 1).clamp(min=1)
        logits, _  = self.model(
            images,
            targets=dec_input,
            target_lengths=dec_lens,
            teacher_forcing_ratio=0.5,
        )
        loss = self.criterion(logits, dec_target, dec_lens)
        return loss, logits

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int, total: int) -> dict:
        self.model.train()
        total_loss  = 0.0
        all_preds, all_targets = [], []
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total}", ncols=110, leave=True)
        for batch in pbar:
            images = batch["images"].to(self.device)
            self.optimizer.zero_grad()

            if self.mode == "attention":
                targets        = batch["targets"].to(self.device)
                target_lengths = batch["target_lengths"].to(self.device)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss, logits = self._attn_forward(images, targets, target_lengths)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, logits = self._attn_forward(images, targets, target_lengths)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                texts = batch["texts"]
                if len(all_preds) < 500:
                    with torch.no_grad():
                        preds = self.decode_helper.batch_indices_to_texts(logits.argmax(-1))
                    all_preds.extend(preds)
                    all_targets.extend(texts)
            else:
                labels        = batch["labels"].to(self.device)
                label_lengths = batch["label_lengths"]
                texts         = batch["texts"]
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss, log_probs = self._ctc_forward(images, labels, label_lengths)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, log_probs = self._ctc_forward(images, labels, label_lengths)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                if len(all_preds) < 500:
                    with torch.no_grad():
                        preds = self.decoder.decode_greedy(log_probs)
                    all_preds.extend(preds)
                    all_targets.extend(texts)

            self.global_step += 1
            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        word_acc, char_acc = _metrics(all_preds, all_targets)
        return {
            "loss":     total_loss / len(loader),
            "word_acc": word_acc,
            "char_acc": char_acc,
            "elapsed":  time.time() - t0,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss  = 0.0
        all_preds, all_targets, samples = [], [], []

        for batch in tqdm(loader, desc="Val", ncols=110, leave=False):
            images = batch["images"].to(self.device)

            if self.mode == "attention":
                targets        = batch["targets"].to(self.device)
                target_lengths = batch["target_lengths"].to(self.device)
                loss, logits   = self._attn_forward(images, targets, target_lengths)
                texts          = batch["texts"]
                preds = self.decode_helper.batch_indices_to_texts(logits.argmax(-1))
            else:
                labels        = batch["labels"].to(self.device)
                label_lengths = batch["label_lengths"]
                texts         = batch["texts"]
                loss, log_probs = self._ctc_forward(images, labels, label_lengths)
                preds = self.decoder.decode_greedy(log_probs)

            total_loss += loss.item()
            all_preds.extend(preds)
            all_targets.extend(texts)
            if not samples:
                samples = list(zip(texts[:5], preds[:5]))

        word_acc, char_acc = _metrics(all_preds, all_targets)
        return {
            "loss":     total_loss / len(loader),
            "word_acc": word_acc,
            "char_acc": char_acc,
            "samples":  samples,
        }

    # ── Ana döngü ─────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        save_dir: str,
    ):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.plateau_sched = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )
        if self._pending_plateau is not None:
            self.plateau_sched.load_state_dict(self._pending_plateau)
            self._pending_plateau = None

        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset) if val_loader else 0
        print(
            f"\n{'='*60}\n"
            f" Mod: {self.mode.upper()} | Train: {n_train:,} | Val: {n_val:,}\n"
            f" Epoch {self.start_epoch+1} → {epochs} | LR: {self.lr:.2e}\n"
            f"{'='*60}\n"
        )

        t_start = time.time()
        for epoch in range(self.start_epoch + 1, epochs + 1):
            self.epoch = epoch
            tm = self.train_epoch(train_loader, epoch, epochs)

            print(
                f"\nEpoch {epoch}/{epochs}  "
                f"loss={tm['loss']:.4f}  "
                f"word={tm['word_acc']*100:.1f}%  "
                f"char={tm['char_acc']*100:.1f}%  "
                f"({_fmt_time(tm['elapsed'])})"
            )

            if val_loader:
                vm = self.validate(val_loader)
                print(
                    f"  [val] loss={vm['loss']:.4f}  "
                    f"word={vm['word_acc']*100:.1f}%  "
                    f"char={vm['char_acc']*100:.1f}%"
                )
                for tgt, pred in vm["samples"]:
                    tag = "✓" if tgt == pred else "✗"
                    print(f"    {tag}  '{tgt}' → '{pred}'")
                self.plateau_sched.step(vm["loss"])
                if vm["word_acc"] > self.best_val_acc:
                    self.best_val_acc = vm["word_acc"]
                    self.save(save_dir / "best_model.pth")
                    print(f"  [best] {self.best_val_acc*100:.2f}%")

            self.save(save_dir / f"checkpoint_epoch_{epoch}.pth")
            elapsed = time.time() - t_start
            eta     = elapsed / epoch * (epochs - epoch)
            print(f"  ETA: {_fmt_time(eta)}\n")

        self.save(save_dir / "final_model.pth")
        print(
            f"Eğitim tamamlandı! "
            f"Süre: {_fmt_time(time.time() - t_start)} | "
            f"Best val: {self.best_val_acc*100:.2f}%"
        )

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: Path):
        torch.save(
            {
                "epoch":                       self.epoch,
                "model_state_dict":            self.model.state_dict(),
                "optimizer_state_dict":        self.optimizer.state_dict(),
                "plateau_scheduler_state_dict": (
                    self.plateau_sched.state_dict() if self.plateau_sched else None
                ),
                "best_val_acc":  self.best_val_acc,
                "global_step":   self.global_step,
                "vocab_size":    self.vocab.size,
                "mode":          self.mode,
            },
            path,
        )

    def load(self, path: str, reset_best_acc: bool = False, lr_override: Optional[float] = None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch       = ckpt["epoch"]
        self.start_epoch = ckpt["epoch"]
        self.best_val_acc = 0.0 if reset_best_acc else ckpt.get("best_val_acc", 0.0)
        self.global_step  = ckpt.get("global_step", 0)
        self._pending_plateau = ckpt.get("plateau_scheduler_state_dict")

        if lr_override is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr_override
            print(f"[LR] override: {lr_override:.2e}")

        print(
            f"[CKPT] epoch={self.epoch}  "
            f"best={self.best_val_acc*100:.2f}%  "
            f"→ epoch {self.start_epoch+1}'den devam"
        )
