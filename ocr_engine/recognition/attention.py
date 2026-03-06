"""
Attention tabanli metin tanima modulu

El yazisi ve egri/bozuk metinler icin CTC yerine Attention Decoder kullanir.
CRNN encoder cikisina Bahdanau (additive) attention mekanizmasi uygular.

Avantajlari (CTC'ye gore):
  - El yazisinda daha yuksek dogruluk
  - Karakter hizalamasi gorunulebilir
  - Uzun ve karmasik metinlerde daha kararsiz degil
  - Specials (noktalama, rakam) tespitinde daha guvenilir

Kullanim:
    from ocr_engine.recognition.attention import AttentionCRNN, build_attention_crnn
    model = build_attention_crnn(num_classes=vocab.size)
    logits, attn_weights = model(images, targets, target_lengths)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import takewhile
from typing import Optional, Tuple

from .model import ResNet34Encoder, BidirectionalLSTM


# ---------------------------------------------------------------------------
# Attention mekanizmasi
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_out  : [batch, T, attn_dim]
        proj_enc = self.W_enc(enc_out)

        # dec_hidden: [batch, 1, attn_dim]
        proj_dec = self.W_dec(dec_hidden).unsqueeze(1)

        # energy: [batch, T, 1] -> [batch, T]
        energy = self.v(torch.tanh(proj_enc + proj_dec)).squeeze(-1)

        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))

        attn_w = F.softmax(energy, dim=-1)           # [batch, T]

        # NaN'leri sifirla (tum pozisyonlar maskeli ise)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

        context = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)  # [batch, enc_dim]
        return context, attn_w


# ---------------------------------------------------------------------------
# Attention Decoder
# ---------------------------------------------------------------------------

class AttentionDecoder(nn.Module):

    def __init__(
        self,
        num_classes: int,
        enc_dim: int = 512,
        dec_hidden: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dec_hidden = dec_hidden

        # Karakter embedding
        self.embedding = nn.Embedding(num_classes, 64)

        # Attention
        self.attention = BahdanauAttention(enc_dim, dec_hidden, attn_dim)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=64 + enc_dim,   # embedding + context
            hidden_size=dec_hidden,
            num_layers=1,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(dec_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        prev_char: torch.Tensor,
        hidden: torch.Tensor,
        enc_out: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [batch] -> [batch, 64]
        emb = self.embedding(prev_char)

        # Decoder gizli durumu [1, batch, dim] -> [batch, dim]
        h = hidden.squeeze(0)

        context, attn_w = self.attention(enc_out, h, mask)

        # GRU giris: [batch, 1, 64 + enc_dim]
        gru_input = torch.cat([emb, context], dim=-1).unsqueeze(1)
        gru_out, hidden = self.gru(gru_input, hidden)   # [batch, 1, dec_dim]
        gru_out = self.dropout(gru_out.squeeze(1))      # [batch, dec_dim]

        logit = self.output_proj(gru_out)               # [batch, num_classes]
        return logit, hidden, attn_w

    def forward(
        self,
        enc_out: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        max_len: int = 100,
        teacher_forcing_ratio: float = 0.5,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = enc_out.size(0)
        device = enc_out.device

        # Baslangic gizli durumu sifir
        hidden = torch.zeros(1, batch, self.dec_hidden, device=device)

        # Baslangic token
        prev_char = torch.full((batch,), sos_idx, dtype=torch.long, device=device)

        is_train = (targets is not None)
        seq_len = targets.size(1) if is_train else max_len

        all_logits = []
        all_attns = []

        for t in range(seq_len):
            logit, hidden, attn_w = self.forward_step(prev_char, hidden, enc_out)
            all_logits.append(logit.unsqueeze(1))   # [batch, 1, C]
            all_attns.append(attn_w.unsqueeze(1))   # [batch, 1, T]

            if is_train:
                use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
                prev_char = targets[:, t] if use_teacher else logit.argmax(-1)
            else:
                prev_char = logit.argmax(-1)
                # Tum batch EOS'e ulastiysa dur
                if (prev_char == eos_idx).all():
                    break

        all_logits = torch.cat(all_logits, dim=1)  # [batch, seq, C]
        all_attns = torch.cat(all_attns, dim=1)    # [batch, seq, T]
        return all_logits, all_attns


# ---------------------------------------------------------------------------
# AttentionCRNN (end-to-end)
# ---------------------------------------------------------------------------

class AttentionCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        attn_dim: int = 256,
        dropout: float = 0.1,
        sos_idx: int = 1,
        eos_idx: int = 2,
        encoder_type: str = 'resnet34',  # gelecekte farkli encoder destegi icin
    ):
        super().__init__()

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.num_classes = num_classes

        # CNN Encoder
        self.encoder = ResNet34Encoder(input_channels)

        enc_channels = self.encoder.output_channels  # 512

        # 2. BiLSTM sequence modelleme
        self.bilstm_layers = nn.ModuleList()
        for i in range(num_layers):
            inp = enc_channels if i == 0 else hidden_size
            self.bilstm_layers.append(
                BidirectionalLSTM(inp, hidden_size, hidden_size, dropout if i < num_layers - 1 else 0)
            )

        # 3. Attention Decoder
        self.decoder = AttentionDecoder(
            num_classes=num_classes,
            enc_dim=hidden_size,
            dec_hidden=hidden_size * 2,
            attn_dim=attn_dim,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """CNN + BiLSTM ile goruntu encode et."""
        # CNN: [B, C, H, W] -> [B, 512, 1, W']
        feat = self.encoder(x)

        # [B, 512, 1, W'] -> [B, W', 512]
        B, C, H, W = feat.size()
        feat = feat.squeeze(2).permute(0, 2, 1)

        # BiLSTM katmanlari: [B, W', 512] -> [B, W', hidden]
        for layer in self.bilstm_layers:
            feat = layer(feat)

        return feat  # [B, W', hidden]

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out = self._encode(x)   # [B, T_enc, hidden]

        logits, attns = self.decoder(
            enc_out,
            targets=targets,
            target_lengths=target_lengths,
            teacher_forcing_ratio=teacher_forcing_ratio,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx
        )

        return logits, attns

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        max_len: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        enc_out = self._encode(x)
        logits, attns = self.decoder(
            enc_out,
            targets=None,
            max_len=max_len,
            teacher_forcing_ratio=0.0,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx
        )
        char_indices = logits.argmax(-1)        # [B, seq_len]
        return char_indices, attns

    def get_sequence_length(self, input_width: int) -> int:
        return input_width // 4


# ---------------------------------------------------------------------------
# Cross-Entropy Loss (Attention icin)
# ---------------------------------------------------------------------------

class AttentionLoss(nn.Module):
    """Attention decoder icin masked cross-entropy loss."""

    def __init__(self, pad_idx: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        B, seq_len, C = logits.size()
        tgt_len = targets.size(1)
        min_len = min(seq_len, tgt_len)

        loss = self.criterion(
            logits[:, :min_len, :].reshape(-1, C),
            targets[:, :min_len].reshape(-1)
        )
        return loss


# ---------------------------------------------------------------------------
# Decoder: indislerden metin olusturma
# ---------------------------------------------------------------------------

class AttentionDecodeHelper:
    """AttentionCRNN cikislarini metne donusturur."""

    def __init__(self, vocab, sos_idx: int = 1, eos_idx: int = 2):
        self.vocab = vocab
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def indices_to_text(self, indices: torch.Tensor) -> str:
        """[seq_len] tensorunu metne cevir — EOS'ta dur."""
        skip = {self.sos_idx, self.vocab.blank_idx, self.vocab.unk_idx}
        return ''.join(
            ch for idx in takewhile(lambda i: i != self.eos_idx, indices.tolist())
            if idx not in skip and (ch := self.vocab.idx_to_char.get(idx, ''))
        )

    def batch_indices_to_texts(self, batch_indices: torch.Tensor) -> list:
        return [self.indices_to_text(row) for row in batch_indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_attention_crnn(
    num_classes: int,
    input_channels: int = 1,
    hidden_size: int = 256,
    num_layers: int = 2,
    attn_dim: int = 256,
    dropout: float = 0.1,
    sos_idx: int = 1,
    eos_idx: int = 2,
    weights_path: Optional[str] = None
) -> AttentionCRNN:
    model = AttentionCRNN(
        num_classes=num_classes, input_channels=input_channels,
        hidden_size=hidden_size, num_layers=num_layers,
        attn_dim=attn_dim, dropout=dropout,
        sos_idx=sos_idx, eos_idx=eos_idx,
    )
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    return model