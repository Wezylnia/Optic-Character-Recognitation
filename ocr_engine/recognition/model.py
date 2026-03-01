"""
CRNN (CNN + RNN + CTC) metin tanima modeli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM blogu"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
            
        Returns:
            [batch, seq_len, output_size]
        """
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.linear(output)
        return output


class VGGEncoder(nn.Module):
    """VGG-style CNN encoder for feature extraction"""
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        # Kanal sayilari: 64 -> 128 -> 256 -> 256 -> 512 -> 512 -> 512
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # H/2
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # H/4
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1))  # H/8, W ayni
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1))  # H/16, W ayni
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=2, padding=0)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.output_channels = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
            
        Returns:
            [batch, channels, 1, width'] - height 1'e dusurulur
        """
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Conv block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool3(x)
        
        # Conv block 4
        x = self.relu(self.bn4(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        # Conv block 5
        x = self.relu(self.bn5(self.conv5(x)))
        
        return x


class ResNetEncoder(nn.Module):
    """ResNet-style CNN encoder (alternatif)"""
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResBlock 1
        self.block1 = self._make_block(32, 64, stride=2)  # H/2
        
        # ResBlock 2
        self.block2 = self._make_block(64, 128, stride=(2, 1))  # H/4
        
        # ResBlock 3
        self.block3 = self._make_block(128, 256, stride=(2, 1))  # H/8
        
        # ResBlock 4
        self.block4 = self._make_block(256, 512, stride=(2, 1))  # H/16
        
        self.conv_out = nn.Conv2d(512, 512, kernel_size=(2, 1))
        self.bn_out = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU(inplace=True)
        self.output_channels = 512
    
    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Residual block olustur"""
        if isinstance(stride, int):
            stride = (stride, stride)
        
        downsample = None
        if stride != (1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.relu(self.bn_out(self.conv_out(x)))
        
        return x


class ResBlock(nn.Module):
    """Basic residual block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CRNN(nn.Module):
    """
    CRNN - Convolutional Recurrent Neural Network
    
    Metin tanima icin CNN + BiLSTM + CTC mimarisi.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "vgg"
    ):
        """
        Args:
            num_classes: Karakter sayisi (vocab size + blank)
            input_channels: Giris kanal sayisi (1=grayscale, 3=RGB)
            hidden_size: LSTM hidden size
            num_layers: LSTM katman sayisi
            dropout: Dropout orani
            encoder_type: CNN encoder tipi (vgg veya resnet)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # CNN Encoder
        if encoder_type == "vgg":
            self.encoder = VGGEncoder(input_channels)
        elif encoder_type == "resnet":
            self.encoder = ResNetEncoder(input_channels)
        else:
            raise ValueError(f"Bilinmeyen encoder tipi: {encoder_type}")
        
        encoder_output = self.encoder.output_channels
        
        # Sequence modeling (BiLSTM)
        self.sequence = nn.Sequential()
        
        for i in range(num_layers):
            input_dim = encoder_output if i == 0 else hidden_size
            self.sequence.add_module(
                f'bilstm_{i}',
                BidirectionalLSTM(
                    input_dim,
                    hidden_size,
                    hidden_size,
                    dropout if i < num_layers - 1 else 0
                )
            )
        
        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Agirliklari initialize et"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Giris gorseli [batch, channels, height, width]
               height = 32 olmali
               
        Returns:
            Log probabilities [seq_len, batch, num_classes]
        """
        # CNN feature extraction
        # [batch, channels, height, width] -> [batch, 512, 1, width']
        conv_out = self.encoder(x)
        
        # Reshape for RNN
        # [batch, 512, 1, width'] -> [batch, width', 512]
        batch, channels, height, width = conv_out.size()
        assert height == 1, (
            f"CRNN encoder ciktisi height=1 olmali, ancak {height} geldi. "
            f"Giris gorselinin yuksekligi encoder'in beklentisiyle uyusmuyor. "
            f"Input shape: {x.shape}"
        )
        
        conv_out = conv_out.squeeze(2)  # [batch, channels, width]
        conv_out = conv_out.permute(0, 2, 1)  # [batch, width, channels]
        
        # Sequence modeling
        # [batch, width, channels] -> [batch, width, hidden_size]
        rnn_out = self.sequence(conv_out)
        
        # Output layer
        # [batch, width, hidden_size] -> [batch, width, num_classes]
        output = self.output(rnn_out)
        
        # CTC icin: [seq_len, batch, num_classes]
        output = output.permute(1, 0, 2)
        
        # Log softmax
        output = F.log_softmax(output, dim=2)
        
        return output
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference icin
        
        Args:
            x: Giris gorseli [batch, channels, height, width]
            
        Returns:
            Log probabilities [seq_len, batch, num_classes]
        """
        self.eval()
        return self.forward(x)
    
    def get_sequence_length(self, input_width: int) -> int:
        """
        Giris genisligine gore cikis sequence uzunlugunu hesapla

        Args:
            input_width: Giris gorseli genisligi

        Returns:
            Cikis sequence uzunlugu
        """
        if isinstance(self.encoder, ResNetEncoder):
            # block1: stride=(2,2) → W/2
            # block2-4: stride=(2,1) → W unchanged
            # conv_out: kernel_w=1 → W unchanged
            return input_width // 2
        else:
            # VGG encoder:
            # Pool1: W/2, Pool2: W/2, Pool3/4: W unchanged, Conv5: W-1
            return (input_width // 4) - 1


class CRNNLoss(nn.Module):
    """CTC Loss wrapper"""
    
    def __init__(self, blank_idx: int = 0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            log_probs: [seq_len, batch, num_classes]
            targets: [sum(target_lengths)] veya [batch, max_target_len]
            input_lengths: [batch]
            target_lengths: [batch]
            
        Returns:
            CTC loss
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def build_crnn(
    num_classes: int,
    input_channels: int = 1,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    encoder_type: str = "vgg",
    weights_path: Optional[str] = None
) -> CRNN:
    """
    CRNN model olustur
    
    Args:
        num_classes: Karakter sayisi
        input_channels: Giris kanal sayisi
        hidden_size: LSTM hidden size
        num_layers: LSTM katman sayisi
        dropout: Dropout orani
        encoder_type: CNN encoder tipi
        weights_path: Model agirlik dosyasi
        
    Returns:
        CRNN model
    """
    model = CRNN(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        encoder_type=encoder_type
    )
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
    
    return model
