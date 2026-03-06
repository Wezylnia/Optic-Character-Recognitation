"""CRNN metin tanima modeli — ResNet34 + BiLSTM + CTC"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return self.linear(self.drop(out))




class ResNet34Encoder(nn.Module):
    """
    ResNet34 tabanli OCR encoder — layer4 dahil, custom stride ameliyati uygulanmis.

    Neden stride ameliyati?
      ResNet34'un layer2/3/4 ilk blogundaki stride=(2,2) hem yuksekligi hem
      genisligi yariya indirir. OCR'da genislik = CTC zaman ekseni oldugu icin
      her fazladan /2 isleminde time-step sayisi yariya dusuyor, bu da CTC'nin
      tekrar eden karakterleri (OO, KK, EE) blank ile ayirma sansini yok ediyor.
      Cozum: stride=(2,2) -> stride=(2,1) ile sadece yuksekligi boluyoruz.

    Akis (H=48, W=256 girdisi icin):
      stem  (stride=2)           : [B,  64, 24, 128]
      maxpool (stride=2)         : [B,  64, 12,  64]
      layer1 (stride=1)          : [B,  64, 12,  64]
      layer2 (stride=(2,1))      : [B, 128,  6,  64]
      layer3 (stride=(2,1))      : [B, 256,  3,  64]
      layer4 (stride=(2,1))      : [B, 512,  2,  64]
      AdaptiveAvgPool((1, None)) : [B, 512,  1,  64]  <- W//4, VGG ile ayni

    Cikis: [B, 512, 1, W//4]
    """

    def __init__(self, input_channels: int = 1, pretrained: bool = True):
        super().__init__()

        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34(weights=weights)
        except (ImportError, AttributeError):
            try:
                import torchvision.models as _tvm
                backbone = _tvm.resnet34(pretrained=pretrained)
            except TypeError:
                import torchvision.models as _tvm
                backbone = _tvm.resnet34()

        # conv1'i gri tonlu (1-kanalli) girdi icin uyarla
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        self.stem    = nn.Sequential(new_conv, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool           # stride=2: H//4, W//4

        self.layer1  = backbone.layer1            # stride=1 — genislik degismez
        self.layer2  = self._patch_stride(backbone.layer2)   # (2,1)
        self.layer3  = self._patch_stride(backbone.layer3)   # (2,1)
        self.layer4  = self._patch_stride(backbone.layer4)   # (2,1)

        self.pool    = nn.AdaptiveAvgPool2d((1, None))        # H -> 1

        self.output_channels = 512

    @staticmethod
    def _patch_stride(layer: nn.Sequential) -> nn.Sequential:
        """
        Bir ResNet katmanindaki ilk blogun stride=(2,2) degerini stride=(2,1) yapar.
        Hem conv1 hem downsample shortcut guncellenir; ImageNet agirliklari korunur.
        """
        block = layer[0]

        # conv1: stride (2,2) -> (2,1)
        old_c = block.conv1
        new_c = nn.Conv2d(
            old_c.in_channels, old_c.out_channels,
            kernel_size=old_c.kernel_size,
            stride=(2, 1),
            padding=old_c.padding,
            bias=old_c.bias is not None,
        )
        with torch.no_grad():
            new_c.weight.data.copy_(old_c.weight.data)
        block.conv1 = new_c

        # downsample shortcut: stride (2,2) -> (2,1)
        if block.downsample is not None:
            old_ds = block.downsample[0]
            new_ds = nn.Conv2d(
                old_ds.in_channels, old_ds.out_channels,
                kernel_size=old_ds.kernel_size,
                stride=(2, 1),
                padding=old_ds.padding,
                bias=old_ds.bias is not None,
            )
            with torch.no_grad():
                new_ds.weight.data.copy_(old_ds.weight.data)
            block.downsample[0] = new_ds

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  — H=48 (varsayilan) veya H=32

        Returns:
            [B, 512, 1, W//4]
        """
        x = self.stem(x)     # [B,  64, H/2,  W/2]
        x = self.maxpool(x)  # [B,  64, H/4,  W/4]
        x = self.layer1(x)   # [B,  64, H/4,  W/4]
        x = self.layer2(x)   # [B, 128, H/8,  W/4]
        x = self.layer3(x)   # [B, 256, H/16, W/4]
        x = self.layer4(x)   # [B, 512, H/32, W/4]
        x = self.pool(x)     # [B, 512, 1,    W/4]
        return x


class CRNN(nn.Module):
    """ResNet34 + 2×BiLSTM + CTC. Giris: [B, 1, 48, W]. Cikis: log-probs [W//4, B, C]."""

    def __init__(self, num_classes, input_channels=1, hidden_size=256, num_layers=2, dropout=0.1, **_):
        super().__init__()
        self.encoder = ResNet34Encoder(input_channels)
        enc_ch = self.encoder.output_channels
        self.sequence = nn.Sequential(*[
            BidirectionalLSTM(
                enc_ch if i == 0 else hidden_size,
                hidden_size, hidden_size,
                dropout if i < num_layers - 1 else 0
            ) for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.encoder(x)
        assert feat.size(2) == 1, f"Encoder H={feat.size(2)}, expected 1"
        feat = feat.squeeze(2).permute(0, 2, 1)           # [B, W//4, 512]
        out  = self.output(self.sequence(feat))           # [B, W//4, C]
        return F.log_softmax(out.permute(1, 0, 2), dim=2) # [W//4, B, C]

    def get_sequence_length(self, input_width):
        return input_width // 4


class CRNNLoss(nn.Module):
    def __init__(self, blank_idx=0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


def build_crnn(num_classes, input_channels=1, hidden_size=256, num_layers=2,
               dropout=0.1, weights_path=None, **kwargs):
    model = CRNN(num_classes, input_channels, hidden_size, num_layers, dropout)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    return model