"""
DBNet Detection Model Egitim Scripti
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine.detection.model import DBNet, DBLoss, build_dbnet
from training.dataset import DetectionDataset
from training.augmentation import Augmentor


class DetectionTrainer:
    """Detection model egitici"""
    
    def __init__(
        self,
        config: dict,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = self._build_model()
        
        # Loss
        self.criterion = DBLoss()
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Egitim metrikleri
        self.best_loss = float('inf')
        self.epoch = 0
    
    def _build_model(self) -> DBNet:
        """Model olustur"""
        model_cfg = self.config.get('detection', {}).get('model', {})
        
        model = DBNet(
            backbone=model_cfg.get('backbone', 'resnet18'),
            pretrained=model_cfg.get('pretrained', True)
        )
        
        return model.to(self.device)
    
    def _build_optimizer(self):
        """Optimizer olustur"""
        train_cfg = self.config.get('training', {}).get('detection', {})
        lr = train_cfg.get('learning_rate', 0.001)
        
        return Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
    
    def _build_scheduler(self):
        """LR scheduler olustur"""
        train_cfg = self.config.get('training', {}).get('detection', {})
        epochs = train_cfg.get('epochs', 100)
        
        return CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Bir epoch egit"""
        self.model.train()
        
        total_loss = 0
        prob_loss_sum = 0
        binary_loss_sum = 0
        thresh_loss_sum = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # Veriyi GPU'ya tasi
            images = batch['image'].to(self.device)
            gt_prob = batch['prob_map'].to(self.device)
            gt_thresh = batch['thresh_map'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images, return_maps=True)
            
            # Loss
            losses = self.criterion(outputs, gt_prob, gt_thresh, mask)
            loss = losses['total_loss']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Metrikleri guncelle
            total_loss += loss.item()
            prob_loss_sum += losses['prob_loss'].item()
            binary_loss_sum += losses['binary_loss'].item()
            thresh_loss_sum += losses['thresh_loss'].item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'prob': f"{losses['prob_loss'].item():.4f}"
            })
        
        num_batches = len(dataloader)
        
        return {
            'total_loss': total_loss / num_batches,
            'prob_loss': prob_loss_sum / num_batches,
            'binary_loss': binary_loss_sum / num_batches,
            'thresh_loss': thresh_loss_sum / num_batches
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            gt_prob = batch['prob_map'].to(self.device)
            gt_thresh = batch['thresh_map'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            outputs = self.model(images, return_maps=True)
            losses = self.criterion(outputs, gt_prob, gt_thresh, mask)
            
            total_loss += losses['total_loss'].item()
        
        return {
            'val_loss': total_loss / len(dataloader)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        save_dir: str = 'checkpoints'
    ):
        """Egitim dongusu"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        train_cfg = self.config.get('training', {}).get('detection', {})
        save_interval = train_cfg.get('save_interval', 10)
        val_interval = train_cfg.get('val_interval', 5)
        
        for epoch in range(epochs):
            self.epoch = epoch + 1
            
            # Egit
            train_metrics = self.train_epoch(train_loader)
            print(f"\nEpoch {self.epoch} - Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Validation
            if val_loader and (self.epoch % val_interval == 0):
                val_metrics = self.validate(val_loader)
                print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
                
                # En iyi modeli kaydet
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint(save_dir / 'best_detection.pth')
                    print("Best model saved!")
            
            # Periyodik kayit
            if self.epoch % save_interval == 0:
                self.save_checkpoint(save_dir / f'detection_epoch_{self.epoch}.pth')
            
            # LR scheduler
            self.scheduler.step()
        
        # Son modeli kaydet
        self.save_checkpoint(save_dir / 'detection_final.pth')
        print("\nEgitim tamamlandi!")
    
    def save_checkpoint(self, path: str):
        """Checkpoint kaydet"""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, path)
    
    def load_checkpoint(self, path: str):
        """Checkpoint yukle"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint yuklendi: epoch {self.epoch}, best_loss {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='DBNet Detection Egitimi')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config dosyasi')
    parser.add_argument('--data_dir', type=str, required=True, help='Veri klasoru')
    parser.add_argument('--train_ann', type=str, required=True, help='Train annotation dosyasi')
    parser.add_argument('--val_ann', type=str, default=None, help='Validation annotation dosyasi')
    parser.add_argument('--epochs', type=int, default=100, help='Epoch sayisi')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch boyutu')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint yolu')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Kayit klasoru')
    parser.add_argument('--device', type=str, default='cuda', help='Cihaz')
    
    args = parser.parse_args()
    
    # Config yukle
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Dataset
    det_cfg = config.get('detection', {})
    input_size = tuple(det_cfg.get('input_size', [640, 640]))
    
    augmentor = Augmentor() if config.get('training', {}).get('augmentation', {}).get('enabled', True) else None
    
    train_dataset = DetectionDataset(
        data_dir=args.data_dir,
        annotation_file=args.train_ann,
        input_size=input_size,
        augmentor=augmentor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.get('general', {}).get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = None
    if args.val_ann:
        val_dataset = DetectionDataset(
            data_dir=args.data_dir,
            annotation_file=args.val_ann,
            input_size=input_size,
            augmentor=None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    # Trainer
    trainer = DetectionTrainer(config, device=args.device)
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Egit
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
