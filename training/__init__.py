"""
Training modulu - Model egitim scriptleri ve veri setleri

Modüller:
  synthetic                -> SyntheticTextGenerator
  recognition_dataset      -> RecognitionDataset, collate_recognition
  detection_dataset        -> DetectionDataset, SynthTextDataset, ICDARDataset
  augmentation_detection   -> Augmentor, DetectionAugmentor
  augmentation_recognition -> RecognitionAugmentor
  train_detection          -> DetectionTrainer, DetectionMetrics
  train_recognition        -> RecognitionTrainer
  train_attention          -> AttentionTrainer
  train_recognition_mjsynth -> MJSynth advanced trainer (AMP + OneCycleLR)"""