"""
Karakter sozlugu (vocabulary) yonetimi
"""

from typing import List, Dict, Optional
import json
from pathlib import Path


class Vocabulary:
    """Karakter sozlugu sinifi"""
    
    # Varsayilan Turkce + Ingilizce karakter seti
    DEFAULT_CHARS = (
        # Rakamlar
        "0123456789"
        # Kucuk harfler (Ingilizce)
        "abcdefghijklmnopqrstuvwxyz"
        # Buyuk harfler (Ingilizce)
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Turkce ozel karakterler
        "çÇğĞıİöÖşŞüÜ"
        # Noktalama isaretleri
        ".,;:!?-'\"()[]{}/"
        # Ozel karakterler
        "@#$%&*+=<>~`\\|^_"
        # Bosluk
        " "
    )
    
    # CTC blank token
    BLANK_TOKEN = "<BLANK>"
    
    # Bilinmeyen karakter token
    UNK_TOKEN = "<UNK>"
    
    def __init__(
        self,
        chars: Optional[str] = None,
        include_blank: bool = True,
        include_unk: bool = True
    ):
        """
        Args:
            chars: Karakter dizisi (None ise varsayilan)
            include_blank: CTC blank token ekle
            include_unk: Bilinmeyen token ekle
        """
        if chars is None:
            chars = self.DEFAULT_CHARS
        
        self.chars = chars
        self.include_blank = include_blank
        self.include_unk = include_unk
        
        # Karakter -> index ve index -> karakter mapping'leri olustur
        self._build_vocab()
    
    def _build_vocab(self):
        """Vocabulary mapping'lerini olustur"""
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
        idx = 0
        
        # Blank token (CTC icin index 0)
        if self.include_blank:
            self.char_to_idx[self.BLANK_TOKEN] = idx
            self.idx_to_char[idx] = self.BLANK_TOKEN
            self.blank_idx = idx
            idx += 1
        else:
            self.blank_idx = -1
        
        # Bilinmeyen token
        if self.include_unk:
            self.char_to_idx[self.UNK_TOKEN] = idx
            self.idx_to_char[idx] = self.UNK_TOKEN
            self.unk_idx = idx
            idx += 1
        else:
            self.unk_idx = -1
        
        # Normal karakterler
        for char in self.chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        self._size = idx
    
    @property
    def size(self) -> int:
        """Vocabulary boyutu (blank ve unk dahil)"""
        return self._size
    
    @property
    def num_classes(self) -> int:
        """Model icin sinif sayisi (size ile ayni)"""
        return self._size
    
    def encode(self, text: str) -> List[int]:
        """
        Metni index listesine donustur
        
        Args:
            text: Giris metni
            
        Returns:
            Index listesi
        """
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            elif self.include_unk:
                indices.append(self.unk_idx)
            # else: karakteri atla
        
        return indices
    
    def decode(
        self,
        indices: List[int],
        remove_blank: bool = True,
        remove_unk: bool = False
    ) -> str:
        """
        Index listesini metne donustur
        
        Args:
            indices: Index listesi
            remove_blank: Blank token'lari kaldir
            remove_unk: UNK token'lari kaldir
            
        Returns:
            Metin
        """
        chars = []
        for idx in indices:
            if idx not in self.idx_to_char:
                continue
            
            char = self.idx_to_char[idx]
            
            if remove_blank and char == self.BLANK_TOKEN:
                continue
            if remove_unk and char == self.UNK_TOKEN:
                continue
            
            chars.append(char)
        
        return ''.join(chars)
    
    def __len__(self) -> int:
        return self._size
    
    def __contains__(self, char: str) -> bool:
        return char in self.char_to_idx
    
    def get_char(self, idx: int) -> str:
        """Index'e karsilik gelen karakteri getir"""
        return self.idx_to_char.get(idx, self.UNK_TOKEN)
    
    def get_idx(self, char: str) -> int:
        """Karaktere karsilik gelen index'i getir"""
        return self.char_to_idx.get(char, self.unk_idx)
    
    def save(self, path: str):
        """Vocabulary'yi dosyaya kaydet"""
        data = {
            'chars': self.chars,
            'include_blank': self.include_blank,
            'include_unk': self.include_unk,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Vocabulary'yi dosyadan yukle"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            chars=data['chars'],
            include_blank=data['include_blank'],
            include_unk=data['include_unk']
        )
        
        return vocab
    
    def get_all_chars(self, include_special: bool = False) -> List[str]:
        """
        Tum karakterleri listele
        
        Args:
            include_special: Ozel token'lari dahil et (BLANK, UNK)
            
        Returns:
            Karakter listesi
        """
        if include_special:
            return list(self.char_to_idx.keys())
        else:
            chars = []
            for char in self.char_to_idx.keys():
                if char not in [self.BLANK_TOKEN, self.UNK_TOKEN]:
                    chars.append(char)
            return chars
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size}, chars='{self.chars[:20]}...')"


# Hazir vocabulary'ler
def get_turkish_vocab() -> Vocabulary:
    """Turkce icin optimize edilmis vocabulary"""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "çÇğĞıİöÖşŞüÜ"
        ".,;:!?-'\"()[]{}/ "
    )
    return Vocabulary(chars)


def get_english_vocab() -> Vocabulary:
    """Ingilizce icin vocabulary"""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ".,;:!?-'\"()[]{}/ "
    )
    return Vocabulary(chars)


def get_alphanumeric_vocab() -> Vocabulary:
    """Sadece alfanumerik karakterler"""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " "
    )
    return Vocabulary(chars)


def get_full_vocab() -> Vocabulary:
    """Tam karakter seti"""
    return Vocabulary()  # Varsayilan kullan
