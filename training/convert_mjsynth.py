"""
MJSynth veri setini bizim format'a donusturme scripti
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def parse_mjsynth_annotation(annotation_file: str, data_root: str) -> list:
    """
    MJSynth annotation dosyasini parse et
    
    MJSynth format (genellikle):
    path/to/image.jpg WORD
    veya
    ./mnt/ramdisk/.../1_WORD_00001.jpg
    
    Args:
        annotation_file: Annotation dosya yolu
        data_root: Gorsel dosyalarinin kok dizini
        
    Returns:
        [{"image_path": "...", "text": "..."}, ...] listesi
    """
    samples = []
    
    print(f"Annotation dosyasi okunuyor: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Toplam {len(lines)} satir bulundu")
    
    for line in tqdm(lines, desc="Parse ediliyor"):
        line = line.strip()
        if not line:
            continue
        
        # MJSynth format: "./path/ID_WORD_LEXICONINDEX.jpg LEXICONINDEX"
        # Biz kelimeyi DOSYA ADINDAN almaliyiz, annotation'daki sayi lexicon index!
        
        if ' ' in line:
            parts = line.split(' ')
            image_path = parts[0]
            # DIKKAT: parts[1] lexicon index, kelime degil!
            # Kelimeyi dosya adindan cikar
        else:
            image_path = line
        
        # Dosya adindan kelimeyi cikar: ID_WORD_LEXICONINDEX.jpg -> WORD
        filename = Path(image_path).stem  # uzantisiz dosya adi
        parts = filename.split('_')
        if len(parts) >= 3:
            # Format: 100_FURLONG_31322 -> FURLONG
            text = parts[1]
        elif len(parts) == 2:
            # Format: 100_FURLONG -> FURLONG
            text = parts[1]
        else:
            text = filename
        
        # Path'i düzelt
        # MJSynth genellikle "./mnt/ramdisk/..." gibi yollar içerir
        # Bunları data_root'a göre düzelt
        if image_path.startswith('./'):
            image_path = image_path[2:]
        
        # Mutlak yol oluştur
        if not os.path.isabs(image_path):
            full_path = os.path.join(data_root, image_path)
        else:
            full_path = image_path
        
        # Dosya var mı kontrol et (opsiyonel, yavaşlatabilir)
        # if not os.path.exists(full_path):
        #     continue
        
        samples.append({
            "image_path": image_path,  # Göreceli yol
            "text": text
        })
    
    return samples


def analyze_dataset(samples: list):
    """Veri seti istatistikleri"""
    print("\n" + "="*50)
    print("VERI SETI ISTATISTIKLERI")
    print("="*50)
    
    print(f"Toplam ornek sayisi: {len(samples)}")
    
    # Metin uzunluklari
    text_lengths = [len(s['text']) for s in samples]
    print(f"\nMetin uzunluklari:")
    print(f"  - Min: {min(text_lengths)}")
    print(f"  - Max: {max(text_lengths)}")
    print(f"  - Ortalama: {sum(text_lengths) / len(text_lengths):.2f}")
    
    # En sik kelimeler
    texts = [s['text'] for s in samples]
    counter = Counter(texts)
    print(f"\nEn sik 10 kelime:")
    for word, count in counter.most_common(10):
        print(f"  - '{word}': {count} kez")
    
    # Benzersiz kelime sayisi
    print(f"\nBenzersiz kelime sayisi: {len(counter)}")
    
    # Karakter dagilimi
    all_chars = set(''.join(texts))
    print(f"\nBenzersiz karakter sayisi: {len(all_chars)}")
    print(f"Karakterler: {''.join(sorted(all_chars)[:50])}...")


def convert_mjsynth(
    data_root: str,
    annotation_file: str,
    output_file: str,
    max_samples: int = None,
    validate: bool = True
):
    """
    MJSynth'i bizim formata cevir
    
    Args:
        data_root: MJSynth kok dizini
        annotation_file: Annotation dosyasi (orn: annotation_train.txt)
        output_file: Cikis JSON dosyasi
        max_samples: Maksimum ornek sayisi (None = hepsi)
        validate: Gorsel dosyalarinin varligini kontrol et
    """
    print(f"MJSynth Converter")
    print(f"Data root: {data_root}")
    print(f"Annotation: {annotation_file}")
    print(f"Output: {output_file}")
    print()
    
    # Parse annotation
    samples = parse_mjsynth_annotation(annotation_file, data_root)
    
    # Limit
    if max_samples and max_samples < len(samples):
        print(f"\nIlk {max_samples} ornek aliniyor...")
        samples = samples[:max_samples]
    
    # Validate (opsiyonel)
    if validate:
        print("\nGorsel dosyalari kontrol ediliyor...")
        valid_samples = []
        invalid_count = 0
        
        for sample in tqdm(samples, desc="Validating"):
            full_path = os.path.join(data_root, sample['image_path'])
            if os.path.exists(full_path):
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        print(f"Gecerli: {len(valid_samples)}, Gecersiz: {invalid_count}")
        samples = valid_samples
    
    # Istatistikler
    analyze_dataset(samples)
    
    # JSON'a yaz
    print(f"\nJSON dosyasi olusturuluyor: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Basariyla kaydedildi: {len(samples)} ornek")
    print(f"✓ Dosya boyutu: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def split_dataset(
    input_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Veri setini train/val/test'e ayir
    
    Args:
        input_file: Giris JSON dosyasi
        train_ratio: Egitim orani
        val_ratio: Validation orani
        test_ratio: Test orani
    """
    import random
    
    print(f"\nVeri seti bolunuyor...")
    print(f"Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {test_ratio*100}%")
    
    # Load
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Shuffle
    random.shuffle(samples)
    
    # Split
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Save
    base_path = Path(input_file).parent
    base_name = Path(input_file).stem
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    for split_name, split_samples in splits.items():
        output_path = base_path / f"{base_name}_{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_samples, f, ensure_ascii=False, indent=2)
        print(f"✓ {split_name}: {len(split_samples)} ornekler -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MJSynth Converter')
    parser.add_argument('--data_root', type=str, required=True,
                        help='MJSynth kok dizini (gorsel dosyalarinin oldugu yer)')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Annotation dosyasi (orn: annotation_train.txt)')
    parser.add_argument('--output', type=str, default='data/mjsynth.json',
                        help='Cikis JSON dosyasi')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maksimum ornek sayisi (test icin)')
    parser.add_argument('--no_validate', action='store_true',
                        help='Gorsel dosyalarini kontrol etme (daha hizli)')
    parser.add_argument('--split', action='store_true',
                        help='Train/val/test\'e ayir')
    
    args = parser.parse_args()
    
    # Convert
    convert_mjsynth(
        data_root=args.data_root,
        annotation_file=args.annotation,
        output_file=args.output,
        max_samples=args.max_samples,
        validate=not args.no_validate
    )
    
    # Split
    if args.split:
        split_dataset(args.output)
    
    print("\n✓ Tamamlandi!")


if __name__ == '__main__':
    main()
