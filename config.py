"""
ISL Bridge Configuration
Central configuration for Indian Sign Language Recognition System
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

# LSTM Model configuration
MODEL_CONFIG = {
    "input_frames": 30,
    "input_dim": 166,
    "hidden_dim": 128,
    "num_layers": 2,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10,
    "confidence_threshold": 0.55,
    "sequence_length": 30
}

# MediaPipe landmark detection
MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1
}

# Camera settings
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30
}

# Translation and TTS
TRANSLATION_CONFIG = {
    "default_language": "hi",
    "supported_languages": ["hi", "en", "ta", "te", "bn", "gu", "mr", "pa"],
    "tts_enabled": True,
    "tts_rate": 150,
    "tts_volume": 0.8
}

def get_available_classes():
    """
    Dynamically detect all gesture classes from the dataset directory.
    Supports letters, numbers, words, and phrases - just add folders and train.
    """
    dataset_path = RAW_DATA_DIR / "Frames_Word_Level"
    
    if not dataset_path.exists():
        return [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
        ]
    
    classes = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            class_name = item.name.upper()
            classes.append(class_name)
    
    def sort_key(name):
        if name.isdigit():
            return (0, int(name))
        elif len(name) == 1 and name.isalpha():
            return (1, name)
        else:
            return (2, name)
    
    classes.sort(key=sort_key)
    return classes

ISL_CLASSES = get_available_classes()

# Feature extraction
FEATURE_CONFIG = {
    "landmark_count": 543,
    "feature_dim": 166,
    "buffer_size": 30,
    "sliding_window": True
}
