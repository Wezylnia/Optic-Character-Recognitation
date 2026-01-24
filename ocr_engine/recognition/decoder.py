"""
CTC Decoder - Model cikislarindan metin uretme
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from .vocab import Vocabulary


class CTCDecoder:
    """CTC cikislarini metne donusturur"""
    
    def __init__(self, vocab: Vocabulary):
        """
        Args:
            vocab: Vocabulary nesnesi
        """
        self.vocab = vocab
        self.blank_idx = vocab.blank_idx
    
    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Greedy decoding - en yuksek olasilikli yolu sec
        
        Args:
            log_probs: [seq_len, batch, num_classes]
            lengths: [batch] - her ornek icin sequence uzunlugu
            
        Returns:
            Decode edilmis metin listesi
        """
        # Argmax
        _, max_indices = log_probs.max(dim=2)  # [seq_len, batch]
        max_indices = max_indices.permute(1, 0)  # [batch, seq_len]
        
        if lengths is None:
            lengths = [max_indices.shape[1]] * max_indices.shape[0]
        else:
            lengths = lengths.cpu().tolist()
        
        decoded = []
        for indices, length in zip(max_indices, lengths):
            indices = indices[:length].cpu().tolist()
            text = self._collapse_repeated(indices)
            decoded.append(text)
        
        return decoded
    
    def decode_beam(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 5,
        lengths: Optional[torch.Tensor] = None
    ) -> List[Tuple[str, float]]:
        """
        Beam search decoding - daha iyi sonuclar icin
        
        Args:
            log_probs: [seq_len, batch, num_classes]
            beam_width: Beam genisligi
            lengths: [batch] - her ornek icin sequence uzunlugu
            
        Returns:
            (metin, skor) tuple listesi
        """
        batch_size = log_probs.shape[1]
        
        if lengths is None:
            lengths = [log_probs.shape[0]] * batch_size
        else:
            lengths = lengths.cpu().tolist()
        
        results = []
        
        for b in range(batch_size):
            seq_len = lengths[b]
            probs = log_probs[:seq_len, b, :].cpu().numpy()  # [seq_len, num_classes]
            
            # Beam search
            beams = [([], 0.0)]  # (prefix, log_prob)
            
            for t in range(seq_len):
                new_beams = []
                
                for prefix, score in beams:
                    for c in range(probs.shape[1]):
                        new_score = score + probs[t, c]
                        
                        if c == self.blank_idx:
                            # Blank - prefix'i koru
                            new_beams.append((prefix.copy(), new_score))
                        else:
                            # Karakter - prefix'e ekle
                            new_prefix = prefix.copy()
                            
                            # Tekrarli karakterleri birlestir (CTC)
                            if len(new_prefix) == 0 or new_prefix[-1] != c:
                                new_prefix.append(c)
                            
                            new_beams.append((new_prefix, new_score))
                
                # En iyi beam_width kadarini tut
                new_beams.sort(key=lambda x: x[1], reverse=True)
                
                # Ayni prefix'e sahip olanları birlestir
                merged = {}
                for prefix, score in new_beams:
                    key = tuple(prefix)
                    if key not in merged or merged[key] < score:
                        merged[key] = score
                
                beams = [(list(k), v) for k, v in merged.items()]
                beams.sort(key=lambda x: x[1], reverse=True)
                beams = beams[:beam_width]
            
            # En iyi sonucu al
            if beams:
                best_prefix, best_score = beams[0]
                text = self.vocab.decode(best_prefix, remove_blank=False)
                results.append((text, best_score))
            else:
                results.append(("", 0.0))
        
        return results
    
    def _collapse_repeated(self, indices: List[int]) -> str:
        """
        Tekrarli karakterleri birlestir ve blank'lari kaldir
        
        Args:
            indices: Index listesi
            
        Returns:
            Decode edilmis metin
        """
        collapsed = []
        prev_idx = None
        
        for idx in indices:
            # Blank atla
            if idx == self.blank_idx:
                prev_idx = None
                continue
            
            # Tekrar atla
            if idx == prev_idx:
                continue
            
            collapsed.append(idx)
            prev_idx = idx
        
        return self.vocab.decode(collapsed, remove_blank=True)
    
    def decode_batch(
        self,
        log_probs: torch.Tensor,
        method: str = "greedy",
        beam_width: int = 5,
        lengths: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Batch decoding
        
        Args:
            log_probs: [seq_len, batch, num_classes]
            method: Decoding yontemi (greedy veya beam)
            beam_width: Beam search icin beam genisligi
            lengths: [batch] - her ornek icin sequence uzunlugu
            
        Returns:
            Decode edilmis metin listesi
        """
        if method == "greedy":
            return self.decode_greedy(log_probs, lengths)
        elif method == "beam":
            results = self.decode_beam(log_probs, beam_width, lengths)
            return [text for text, _ in results]
        else:
            raise ValueError(f"Bilinmeyen decoding yontemi: {method}")


class CTCPrefixDecoder:
    """
    CTC Prefix Beam Search Decoder
    
    Daha dogru beam search implementasyonu
    """
    
    def __init__(
        self,
        vocab: Vocabulary,
        beam_width: int = 10,
        blank_idx: Optional[int] = None
    ):
        self.vocab = vocab
        self.beam_width = beam_width
        self.blank_idx = blank_idx if blank_idx is not None else vocab.blank_idx
    
    def decode(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Tek bir sequence icin decode
        
        Args:
            log_probs: [seq_len, num_classes]
            length: Sequence uzunlugu
            
        Returns:
            (metin, log_prob)
        """
        T, V = log_probs.shape
        
        if length is not None:
            T = min(T, length)
            log_probs = log_probs[:T]
        
        # Beams: prefix -> (prob_blank, prob_non_blank)
        # prob_blank: prefix'in blank ile bittigi olasilik
        # prob_non_blank: prefix'in karakter ile bittigi olasilik
        beams = {(): (0.0, float('-inf'))}  # Bos prefix ile basla
        
        for t in range(T):
            new_beams = {}
            
            for prefix, (pb, pnb) in beams.items():
                # Mevcut prefix'in toplam log olasiligi
                p = np.logaddexp(pb, pnb)
                
                # Her karakter icin
                for c in range(V):
                    lp = log_probs[t, c]
                    
                    if c == self.blank_idx:
                        # Blank - prefix ayni kalir
                        key = prefix
                        new_pb = np.logaddexp(pb + lp, pnb + lp)
                        
                        if key in new_beams:
                            new_beams[key] = (
                                np.logaddexp(new_beams[key][0], new_pb),
                                new_beams[key][1]
                            )
                        else:
                            new_beams[key] = (new_pb, float('-inf'))
                    
                    else:
                        # Karakter
                        # Eger son karakter ayni ise, sadece blank'tan gelebilir
                        # Degilse hem blank hem non-blank'tan gelebilir
                        
                        if len(prefix) > 0 and prefix[-1] == c:
                            # Tekrar - sadece blank'tan
                            new_pnb = pb + lp
                        else:
                            # Yeni karakter
                            new_pnb = p + lp
                        
                        key = prefix + (c,)
                        
                        if key in new_beams:
                            new_beams[key] = (
                                new_beams[key][0],
                                np.logaddexp(new_beams[key][1], new_pnb)
                            )
                        else:
                            new_beams[key] = (float('-inf'), new_pnb)
            
            # En iyi beam_width kadarini tut
            scored = [
                (prefix, np.logaddexp(pb, pnb))
                for prefix, (pb, pnb) in new_beams.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            beams = {}
            for prefix, _ in scored[:self.beam_width]:
                beams[prefix] = new_beams[prefix]
        
        # En iyi sonucu al
        if not beams:
            return "", float('-inf')
        
        best_prefix = max(
            beams.keys(),
            key=lambda p: np.logaddexp(beams[p][0], beams[p][1])
        )
        best_score = np.logaddexp(beams[best_prefix][0], beams[best_prefix][1])
        
        text = self.vocab.decode(list(best_prefix), remove_blank=True)
        
        return text, best_score
    
    def decode_batch(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[Tuple[str, float]]:
        """
        Batch decoding
        
        Args:
            log_probs: [seq_len, batch, num_classes]
            lengths: [batch]
            
        Returns:
            List of (text, score)
        """
        log_probs = log_probs.cpu().numpy()
        batch_size = log_probs.shape[1]
        
        if lengths is None:
            lengths = [log_probs.shape[0]] * batch_size
        else:
            lengths = lengths.cpu().tolist()
        
        results = []
        for b in range(batch_size):
            probs = log_probs[:, b, :]
            text, score = self.decode(probs, lengths[b])
            results.append((text, score))
        
        return results
