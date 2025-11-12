"""
ISL Bridge Configuration
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPORTS_DIR = PROJECT_ROOT / "exports"

for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True)

FRAMES_DIR = RAW_DATA_DIR / "Frames_Word_Level"
FRAMES_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "input_frames": 30,
    "input_dim": 166,
    "hidden_dim": 256,
    "num_layers": 3,
    "lstm_units": 128,
    "dropout_rate": 0.3,
    "learning_rate": 0.0003,
    "batch_size": 16,
    "epochs": 150,
    "patience": 20,
    "confidence_threshold": 0.75,
    "sequence_length": 30
}

MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1
}

TRANSLATION_CONFIG = {
    "default_language": "hi",
    "supported_languages": ["hi", "en", "ta", "te", "bn", "gu", "mr", "pa"],
    "tts_enabled": True,
    "tts_rate": 150,
    "tts_volume": 1.0,
    "tts_mode": "offline"
}

def get_available_classes():
    """Detect gesture classes from dataset directory"""
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

FEATURE_CONFIG = {
    "landmark_count": 543,
    "feature_dim": 166,
    "buffer_size": 30,
    "sliding_window": True
}

APP_CONFIG = {
    "prediction_interval": 5,
    "prediction_timeout": 2.0,
    "frame_capture_interval": 30,
    "window_width": 1200,
    "window_height": 700,
    "camera_retry_indices": [1, 2, 0],
}

TRAINING_CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "sequence_length": 30,
    "max_samples_per_class": 2000,
    "batch_size_large": 32,
    "batch_size_default": 32,
    "early_stopping_accuracy": 0.95,
    "print_interval": 5,
}

GPU_CONFIG = {
    "enable_gpu": True,
    "enable_mixed_precision": True,
    "enable_cudnn_benchmark": True,
    "gpu_memory_fraction": 0.8,
    "num_workers": 4,
    "pin_memory": True,
}
