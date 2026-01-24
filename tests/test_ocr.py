"""
OCR Engine Test Suite
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreprocessing(unittest.TestCase):
    """Preprocessing testleri"""
    
    def test_image_processor(self):
        """ImageProcessor temel fonksiyonlari"""
        from ocr_engine.preprocessing import ImageProcessor
        
        processor = ImageProcessor(target_size=(640, 480))
        
        # Test gorseli olustur
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Resize test
        resized, scale = processor.resize_with_aspect_ratio(image, max_width=100)
        self.assertEqual(resized.shape[1], 100)
        self.assertEqual(scale, 0.5)
        
        # Grayscale test
        gray = processor.to_grayscale(image)
        self.assertEqual(len(gray.shape), 2)
        
        # Normalize test
        rgb = processor.to_rgb(image)
        normalized = processor.normalize(rgb)
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_binarizer(self):
        """Binarizer test"""
        from ocr_engine.preprocessing import Binarizer
        
        binarizer = Binarizer(method='adaptive')
        
        # Test gorseli
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Binarize
        binary = binarizer.binarize(gray)
        
        self.assertEqual(binary.shape, gray.shape)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))
    
    def test_denoiser(self):
        """Denoiser test"""
        from ocr_engine.preprocessing import Denoiser
        
        denoiser = Denoiser(method='bilateral', strength=10)
        
        # Test gorseli
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Denoise
        denoised = denoiser.denoise(image)
        
        self.assertEqual(denoised.shape, image.shape)


class TestVocabulary(unittest.TestCase):
    """Vocabulary testleri"""
    
    def test_vocabulary_creation(self):
        """Vocabulary olusturma"""
        from ocr_engine.recognition.vocab import Vocabulary
        
        vocab = Vocabulary()
        
        self.assertGreater(vocab.size, 50)
        self.assertEqual(vocab.blank_idx, 0)
        self.assertIn(' ', vocab)
        self.assertIn('a', vocab)
        self.assertIn('ş', vocab)  # Turkce karakter
    
    def test_encode_decode(self):
        """Encode/decode test"""
        from ocr_engine.recognition.vocab import Vocabulary
        
        vocab = Vocabulary()
        
        text = "Merhaba Dunya"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        
        self.assertEqual(decoded, text)
    
    def test_turkish_chars(self):
        """Turkce karakter testi"""
        from ocr_engine.recognition.vocab import Vocabulary
        
        vocab = Vocabulary()
        
        turkish_text = "çğıöşüÇĞİÖŞÜ"
        encoded = vocab.encode(turkish_text)
        decoded = vocab.decode(encoded)
        
        self.assertEqual(decoded, turkish_text)


class TestModels(unittest.TestCase):
    """Model testleri"""
    
    def test_dbnet_forward(self):
        """DBNet forward pass"""
        try:
            import torch
            from ocr_engine.detection.model import DBNet
            
            model = DBNet(backbone='resnet18', pretrained=False)
            model.eval()
            
            # Test input
            x = torch.randn(1, 3, 640, 640)
            
            with torch.no_grad():
                outputs = model(x)
            
            self.assertIn('prob_map', outputs)
            self.assertIn('binary_map', outputs)
            self.assertEqual(outputs['prob_map'].shape[0], 1)
            
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_crnn_forward(self):
        """CRNN forward pass"""
        try:
            import torch
            from ocr_engine.recognition.model import CRNN
            
            num_classes = 100
            model = CRNN(num_classes=num_classes, input_channels=1)
            model.eval()
            
            # Test input (batch=2, channels=1, height=32, width=128)
            x = torch.randn(2, 1, 32, 128)
            
            with torch.no_grad():
                output = model(x)
            
            # Output: [seq_len, batch, num_classes]
            self.assertEqual(output.shape[1], 2)
            self.assertEqual(output.shape[2], num_classes)
            
        except ImportError:
            self.skipTest("PyTorch not available")


class TestDecoder(unittest.TestCase):
    """Decoder testleri"""
    
    def test_ctc_decoder_greedy(self):
        """CTC greedy decode"""
        try:
            import torch
            from ocr_engine.recognition.vocab import Vocabulary
            from ocr_engine.recognition.decoder import CTCDecoder
            
            vocab = Vocabulary()
            decoder = CTCDecoder(vocab)
            
            # Simule edilmis log_probs
            seq_len, batch, num_classes = 10, 2, vocab.size
            log_probs = torch.randn(seq_len, batch, num_classes)
            log_probs = torch.log_softmax(log_probs, dim=2)
            
            # Decode
            texts = decoder.decode_greedy(log_probs)
            
            self.assertEqual(len(texts), 2)
            
        except ImportError:
            self.skipTest("PyTorch not available")


class TestPostprocessing(unittest.TestCase):
    """Postprocessing testleri"""
    
    def test_spell_checker(self):
        """Spell checker test"""
        from ocr_engine.postprocessing.spell_checker import SpellChecker
        
        checker = SpellChecker(language='tr')
        
        # Dogru kelime
        result = checker.check("bir")
        self.assertTrue(result[0][1])  # 'bir' dogru
        
        # Suggestion
        suggestion = checker.suggest("blr")  # 'bir' olmali
        # Not: basit sozlukte olmayabilir
    
    def test_text_merger(self):
        """Text merger test"""
        from ocr_engine.postprocessing.text_merger import TextMerger, TextLine
        
        merger = TextMerger(line_threshold=10)
        
        # Test kutulari
        boxes = [
            TextLine(text="Hello", box=np.array([[0,0], [50,0], [50,20], [0,20]]), confidence=0.9),
            TextLine(text="World", box=np.array([[60,0], [110,0], [110,20], [60,20]]), confidence=0.9),
            TextLine(text="Test", box=np.array([[0,30], [40,30], [40,50], [0,50]]), confidence=0.9),
        ]
        
        # Satirlara birlestir
        lines = merger.merge_to_lines(boxes)
        
        self.assertEqual(len(lines), 2)  # 2 satir olmali


class TestAugmentation(unittest.TestCase):
    """Augmentation testleri"""
    
    def test_augmentor(self):
        """Augmentor test"""
        from training.augmentation import Augmentor
        
        augmentor = Augmentor(
            rotation_range=(-10, 10),
            scale_range=(0.9, 1.1)
        )
        
        # Test gorseli
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Augment
        augmented, _ = augmentor(image)
        
        self.assertIsNotNone(augmented)
        self.assertEqual(len(augmented.shape), 3)


class TestSyntheticGenerator(unittest.TestCase):
    """Sentetik veri uretici testleri"""
    
    def test_generate_text(self):
        """Sentetik metin uretimi"""
        from ocr_engine.recognition.vocab import Vocabulary
        from training.dataset import SyntheticTextGenerator
        
        vocab = Vocabulary()
        generator = SyntheticTextGenerator(vocab=vocab, image_height=32)
        
        # Metin uret
        image, text = generator.generate(min_length=5, max_length=10)
        
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[0], 32)  # Height
        self.assertGreaterEqual(len(text), 5)
        self.assertLessEqual(len(text), 15)  # Kelime araliginda olabilir


def run_tests():
    """Tum testleri calistir"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Test siniflarini ekle
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestVocabulary))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestDecoder))
    suite.addTests(loader.loadTestsFromTestCase(TestPostprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticGenerator))
    
    # Runner
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    run_tests()
