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
    "input_frames": 32,
    "input_dim": 166,
    "hidden_dim": 128,
    "num_layers": 2,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10,
    "confidence_threshold": 0.7,
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

# Recognition classes: A-Z, 0-9, and basic words
ISL_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO",
    "HELP", "WATER", "FOOD", "BATHROOM", "STOP", "GO"
]

# Feature extraction
FEATURE_CONFIG = {
    "landmark_count": 543,
    "feature_dim": 166,
    "buffer_size": 30,
    "sliding_window": True
}
