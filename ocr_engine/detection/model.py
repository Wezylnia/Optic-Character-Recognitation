"""DBNet metin tespit modeli — ResNet50 + FPN + Differentiable Binarization"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple


def ConvBnRelu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = False
) -> nn.Sequential:
    """Conv + BatchNorm + ReLU blogu"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResNetBackbone(nn.Module):
    """ResNet backbone — feature extraction."""
    
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        
        if backbone_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_channels = [64, 128, 256, 512]
        elif backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")
        
        # ResNet katmanlarini ayir
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class FPN(nn.Module):
    """Feature Pyramid Network."""
    
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # Lateral connections (1x1 conv)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1)
            for ch in in_channels
        ])
        
        # Smooth layers (3x3 conv)
        self.smooth_convs = nn.ModuleList([
            ConvBnRelu(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
        
        self.out_channels = out_channels
    
    def forward(self, features):
        laterals = [lc(f) for lc, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], (h, w), mode='bilinear', align_corners=False)
        outputs = [sc(l) for sc, l in zip(self.smooth_convs, laterals)]
        target_h, target_w = outputs[0].shape[2:]
        fused = outputs[0]
        for out in outputs[1:]:
            fused = fused + F.interpolate(out, (target_h, target_w), mode='bilinear', align_corners=False)
        return fused


class DBHead(nn.Module):
    """Differentiable Binarization Head."""
    
    def __init__(self, in_channels: int, k: int = 50):
        super().__init__()
        self.k = k  # Binarization aggressiveness
        
        # Probability map branch
        self.prob_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
        # Threshold map branch
        self.thresh_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_maps=False):
        prob_map   = self.prob_conv(x)
        thresh_map = self.thresh_conv(x)
        binary_map = torch.reciprocal(1 + torch.exp(-self.k * (prob_map - thresh_map)))
        result = {'prob_map': prob_map, 'binary_map': binary_map}
        if return_maps:
            result['thresh_map'] = thresh_map
        return result


class DBNet(nn.Module):
    """DBNet — Differentiable Binarization Network."""
    
    def __init__(self, backbone="resnet50", pretrained=True, fpn_channels=256, k=50):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, pretrained)
        self.fpn      = FPN(self.backbone.out_channels, fpn_channels)
        self.head     = DBHead(fpn_channels, k)
        self._init_weights()
    
    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.fpn.apply(_init)
        self.head.apply(_init)
    
    def forward(self, x, return_maps=False):
        return self.head(self.fpn(self.backbone(x)), return_maps)

    @torch.no_grad()
    def predict(self, x, threshold=0.3):
        self.eval()
        return (self.forward(x)['prob_map'] > threshold).float()


class DBLoss(nn.Module):
    """L_total = L_prob + alpha*L_binary + beta*L_thresh"""
    
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.ohem_ratio = ohem_ratio
        self.eps   = eps
    
    def forward(self, outputs, gt_prob, gt_thresh, gt_mask):
        prob_map   = outputs['prob_map']
        binary_map = outputs['binary_map']
        thresh_map = outputs.get('thresh_map')
        prob_loss   = self._balanced_bce_loss(prob_map, gt_prob, gt_mask)
        binary_loss = self._dice_loss(binary_map, gt_prob, gt_mask)
        thresh_loss = self._l1_loss(thresh_map, gt_thresh, gt_prob) if thresh_map is not None else torch.tensor(0.0, device=prob_map.device)
        total = prob_loss + self.alpha * binary_loss + self.beta * thresh_loss
        return {'total_loss': total, 'prob_loss': prob_loss, 'binary_loss': binary_loss, 'thresh_loss': thresh_loss}
    
    def _balanced_bce_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Balanced BCE loss with OHEM"""
        positive = (gt > 0.5).float() * mask
        negative = (gt <= 0.5).float() * mask
        
        positive_count = int(positive.sum())
        negative_count = min(
            int(negative.sum()),
            positive_count * self.ohem_ratio
        )
        
        # BCE loss
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        
        # Positive loss
        positive_loss = (loss * positive).sum()
        
        # Negative loss (hard negative mining)
        negative_loss = loss * negative
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)
        negative_loss = negative_loss.sum()
        
        total_count = positive_count + negative_count
        if total_count > 0:
            return (positive_loss + negative_loss) / total_count
        return torch.tensor(0.0, device=pred.device)
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Dice loss"""
        pred = pred * mask
        gt = gt * mask
        
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice
    
    def _l1_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 loss for threshold map"""
        mask = (mask > 0.5).float()
        loss = torch.abs(pred - gt) * mask
        return loss.sum() / (mask.sum() + self.eps)


def build_dbnet(backbone="resnet50", pretrained=True, weights_path=None):
    model = DBNet(backbone=backbone, pretrained=pretrained)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    return model