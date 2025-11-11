"""
ISL Model Training Script - Unified Version
Combines all training features: GPU optimization, class balancing, data augmentation
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from model import ISLModel
from landmark_extractor import LandmarkExtractor
from config import RAW_DATA_DIR, MODELS_DIR, LOGS_DIR, ISL_CLASSES, TRAINING_CONFIG, MODEL_CONFIG
from gpu_utils import print_device_info, optimize_device, get_training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISLDatasetProcessor:
    
    def __init__(self):
        self.landmark_extractor = LandmarkExtractor()
        self.data_dir = RAW_DATA_DIR / "Frames_Word_Level"
        self.processed_data = []
        self.labels = []
        
    def load_dataset(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load and process the ISL dataset from image and video files"""
        logger.info("Loading ISL dataset...")
        logger.info(f"Target classes from config: {len(ISL_CLASSES)} classes")
        
        if not self.data_dir.exists():
            logger.error(f"Dataset directory not found: {self.data_dir}")
            return [], []
        
        gesture_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        logger.info(f"Found {len(gesture_folders)} gesture folders in dataset")
        
        total_samples = 0
        failed_extractions = 0
        class_counts = {}
        
        for gesture_folder in gesture_folders:
            gesture_name = gesture_folder.name.upper()
            
            mapped_class = self._map_gesture_name(gesture_name)
            
            if mapped_class is None:
                logger.warning(f"Skipping unmappable gesture: {gesture_name}")
                continue
                
            logger.info(f"Processing gesture: {gesture_name} -> {mapped_class}")
            
            image_files = list(gesture_folder.glob("*.jpg")) + list(gesture_folder.glob("*.png"))
            video_files = list(gesture_folder.glob("*.mp4")) + list(gesture_folder.glob("*.avi")) + list(gesture_folder.glob("*.mov"))
            
            class_samples = 0
            
            for image_file in image_files:
                try:
                    image = cv2.imread(str(image_file))
                    if image is None:
                        logger.warning(f"Could not load image: {image_file}")
                        continue
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    landmarks = self.landmark_extractor.extract_landmarks(image_rgb)
                    
                    if landmarks is not None:
                        self.processed_data.append(landmarks)
                        self.labels.append(mapped_class)
                        total_samples += 1
                        class_samples += 1
                    else:
                        failed_extractions += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {e}")
                    failed_extractions += 1
            
            for video_file in video_files:
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    if not cap.isOpened():
                        logger.warning(f"Could not open video: {video_file}")
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    frames_to_extract = min(30, frame_count)
                    frame_interval = max(1, frame_count // frames_to_extract)
                    
                    frame_idx = 0
                    extracted_count = 0
                    
                    while cap.isOpened() and extracted_count < frames_to_extract:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_idx % frame_interval == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            landmarks = self.landmark_extractor.extract_landmarks(frame_rgb)
                            
                            if landmarks is not None:
                                self.processed_data.append(landmarks)
                                self.labels.append(mapped_class)
                                total_samples += 1
                                class_samples += 1
                                extracted_count += 1
                            else:
                                failed_extractions += 1
                        
                        frame_idx += 1
                    
                    cap.release()
                    logger.info(f"  Extracted {extracted_count} frames from video: {video_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing video {video_file}: {e}")
                    failed_extractions += 1
            
            class_counts[mapped_class] = class_samples
        
        logger.info(f"Dataset processing complete:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Failed extractions: {failed_extractions}")
        logger.info(f"  Success rate: {total_samples/(total_samples+failed_extractions)*100:.1f}%" if (total_samples+failed_extractions) > 0 else "0%")
        logger.info(f"  Class distribution: {class_counts}")
        
        return self.processed_data, self.labels
    
    def _map_gesture_name(self, gesture_name: str) -> Optional[str]:
        """
        Map dataset gesture names to standardized class names.
        Enables dynamic support for any gesture type (letters, numbers, words, sentences).
        Simply add folders to data/raw/Frames_Word_Level/ and they will be automatically included.
        
        Args:
            gesture_name: Original gesture name from dataset folder
            
        Returns:
            Mapped class name or None if invalid
        """
        gesture_name = gesture_name.upper().strip()
        
        if not gesture_name:
            return None
        
        name_mappings = {
            "HI": "HELLO",
            "THANKS": "THANK_YOU",
            "TOILET": "BATHROOM",
        }
        
        if gesture_name in name_mappings:
            return name_mappings[gesture_name]
        
        number_words = {
            "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
            "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"
        }
        
        if gesture_name in number_words:
            return number_words[gesture_name]
        
        return gesture_name
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def save_processed_data(self, filepath: str):
        """Save processed dataset for future use"""
        if not self.processed_data:
            logger.error("No processed data to save!")
            return
        
        if not self.labels:
            logger.error("No labels to save!")
            return
        
        try:
            feature_dim = self.processed_data[0].shape[0]
            data = {
                'features': [feat.tolist() for feat in self.processed_data],
                'labels': self.labels,
                'feature_dim': feature_dim,
                'num_samples': len(self.processed_data),
                'num_classes': len(set(self.labels))
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Processed data saved to {filepath}")
            logger.info(f"  Samples: {data['num_samples']}, Classes: {data['num_classes']}, Feature dim: {data['feature_dim']}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}", exc_info=True)
    
    def load_processed_data(self, filepath: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load previously processed dataset"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate loaded data
            if 'features' not in data or 'labels' not in data:
                raise ValueError("Invalid processed data file: missing 'features' or 'labels'")
            
            if len(data['features']) != len(data['labels']):
                raise ValueError(f"Data mismatch: {len(data['features'])} features but {len(data['labels'])} labels")
            
            features = [np.array(feat) for feat in data['features']]
            labels = data['labels']
            
            logger.info(f"Loaded {len(features)} samples from {filepath}")
            logger.info(f"  Feature dimension: {data.get('feature_dim', 'unknown')}")
            logger.info(f"  Number of classes: {data.get('num_classes', len(set(labels)))}")
            
            return features, labels
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}", exc_info=True)
            raise

class ISLTrainer:
    """Train ISL recognition model"""
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def prepare_data(self, features: List[np.ndarray], labels: List[str], 
                    test_size: float = None, val_size: float = None, sequence_length: int = None,
                    max_samples_per_class: int = None) -> Dict:
        logger.info("Preparing data for training...")
        
        # Use config defaults if not provided
        test_size = test_size if test_size is not None else TRAINING_CONFIG.get("test_size", 0.2)
        val_size = val_size if val_size is not None else TRAINING_CONFIG.get("val_size", 0.1)
        sequence_length = sequence_length if sequence_length is not None else TRAINING_CONFIG.get("sequence_length", 30)
        max_samples_per_class = max_samples_per_class if max_samples_per_class is not None else TRAINING_CONFIG.get("max_samples_per_class", 1500)
        
        X = np.array(features)
        y = np.array(labels)
        
        if max_samples_per_class:
            logger.info(f"Limiting to max {max_samples_per_class} samples per class for stability...")
            from collections import defaultdict
            class_indices = defaultdict(list)
            
            for idx, label in enumerate(y):
                class_indices[label].append(idx)
            
            selected_indices = []
            for label, indices in class_indices.items():
                if len(indices) > max_samples_per_class:
                    selected = np.random.choice(indices, max_samples_per_class, replace=False)
                    selected_indices.extend(selected)
                    logger.info(f"Class {label}: {len(indices)} → {max_samples_per_class} samples")
                else:
                    selected_indices.extend(indices)
                    logger.info(f"Class {label}: {len(indices)} samples (kept all)")
            
            X = X[selected_indices]
            y = y[selected_indices]
            logger.info(f"Reduced dataset: {len(X)} total samples")
        
        logger.info(f"Original data shape: {X.shape}")
        logger.info(f"Feature dimension: {X.shape[1]}")
        logger.info(f"Number of classes: {len(set(y))}")
        
        X_sequences = []
        y_sequences = []
        
        for i, (feature, label) in enumerate(zip(X, y)):
            sequence = []
            for seq_idx in range(sequence_length):
                noise = np.random.normal(0, 0.01, feature.shape)
                noisy_feature = feature + noise
                sequence.append(noisy_feature)
            
            X_sequences.append(np.array(sequence))
            y_sequences.append(label)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Sequence data shape: {X_sequences.shape}")
        logger.info(f"Sequence length: {sequence_length}")
        
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_sequences, y_sequences, test_size=test_size, random_state=42, stratify=y_sequences
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
            )
        except ValueError as e:
            logger.warning("=" * 80)
            logger.warning("⚠️  STRATIFIED SPLIT FAILED")
            logger.warning(f"Reason: {e}")
            logger.warning("This usually happens when:")
            logger.warning("  - Some classes have very few samples")
            logger.warning("  - Class distribution is highly imbalanced")
            logger.warning("Falling back to random split (may affect validation accuracy)")
            logger.warning("=" * 80)
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_sequences, y_sequences, test_size=test_size, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
            )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples") 
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        from collections import Counter
        logger.info(f"Train class distribution: {dict(Counter(y_train))}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train_model(self, data_splits: Dict, epochs: int = 100) -> Dict:
        """
        Train the ISL model
        
        Args:
            data_splits: Dictionary with train/val/test data
            epochs: Number of training epochs (default 100)
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        num_classes = len(set(data_splits['y_train']))
        self.model = ISLModel(num_classes=num_classes)
        
        self.history = self.model.train(
            data_splits['X_train'], 
            data_splits['y_train'].tolist(),
            validation_data=(data_splits['X_val'], data_splits['y_val'])
        )
        
        logger.info("Training completed!")
        return self.history
    
    def evaluate_model(self, data_splits: Dict) -> Dict:
        """
        Evaluate trained model
        
        Args:
            data_splits: Dictionary with train/val/test data
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if self.model is None:
            logger.error("No trained model found!")
            return {}
        
        test_predictions = []
        test_confidences = []
        
        for sample in data_splits['X_test']:
            prediction, confidence = self.model.predict_class(sample.reshape(1, sample.shape[0], -1))
            test_predictions.append(prediction)
            test_confidences.append(confidence)
        
        true_labels = data_splits['y_test']
        
        report = classification_report(true_labels, test_predictions, output_dict=True)
        
        cm = confusion_matrix(true_labels, test_predictions)
        
        accuracy = np.mean(np.array(test_predictions) == true_labels)
        
        logger.info(f"Test Accuracy: {accuracy:.3f}")
        logger.info(f"Average Confidence: {np.mean(test_confidences):.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': test_predictions,
            'confidences': test_confidences,
            'true_labels': true_labels.tolist()
        }
    
    def save_model(self, model_path: Optional[str] = None):
        """Save trained model"""
        if self.model is None:
            logger.error("No model to save!")
            return
        
        if model_path is None:
            model_path = str(MODELS_DIR / "isl_trained_model")
        
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if self.history is None:
            logger.error("No training history found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['loss'], label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history['accuracy'], label='Training Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training plots saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        labels = class_names if len(class_names) <= 20 else class_names[:20]
        
        plt.imshow(cm[:len(labels), :len(labels)], interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()

def main():
    """Main training function with GPU optimization"""
    logger.info("🤟 ISL Bridge - Model Training Started")
    
    # Display GPU/CPU info
    device_info = print_device_info()
    device = optimize_device()
    training_config = get_training_config()
    
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    processor = ISLDatasetProcessor()
    trainer = ISLTrainer()
    
    processed_data_path = str(RAW_DATA_DIR / "processed_dataset.json")
    
    if os.path.exists(processed_data_path):
        logger.info("Loading previously processed dataset...")
        try:
            features, labels = processor.load_processed_data(processed_data_path)
        except Exception as e:
            logger.error(f"Failed to load processed data, reprocessing: {e}")
            features, labels = processor.load_dataset()
            if features:
                processor.save_processed_data(processed_data_path)
    else:
        logger.info("Processing raw dataset...")
        features, labels = processor.load_dataset()
        
        if not features:
            logger.error("No data loaded! Please check your dataset.")
            return
        
        processor.save_processed_data(processed_data_path)
    
    class_dist = processor.get_class_distribution()
    logger.info(f"Class distribution: {class_dist}")
    
    data_splits = trainer.prepare_data(features, labels)
    
    # Use epochs from config
    epochs = MODEL_CONFIG.get('epochs', 100)
    logger.info(f"Training for {epochs} epochs (from config)")
    history = trainer.train_model(data_splits, epochs=epochs)
    
    eval_results = trainer.evaluate_model(data_splits)
    
    trainer.save_model()
    
    # Cleanup old checkpoints
    try:
        checkpoint_path = MODELS_DIR / "training_checkpoint.pth"
        if checkpoint_path.exists():
            logger.info("Removing training checkpoint (final model saved)")
            checkpoint_path.unlink()
    except Exception as e:
        logger.warning(f"Could not remove training checkpoint: {e}")
    
    plots_dir = LOGS_DIR / "training_plots"
    plots_dir.mkdir(exist_ok=True)
    
    trainer.plot_training_history(str(plots_dir / "training_history.png"))
    
    if 'confusion_matrix' in eval_results:
        unique_labels = sorted(list(set(labels)))
        trainer.plot_confusion_matrix(
            eval_results['confusion_matrix'], 
            unique_labels,
            str(plots_dir / "confusion_matrix.png")
        )
    
    eval_results_path = str(LOGS_DIR / "evaluation_results.json")
    
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to native Python types"""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    eval_results_json = convert_to_json_serializable(eval_results)
    
    try:
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results_json, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_results_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}", exc_info=True)
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"Final test accuracy: {eval_results['accuracy']:.3f}")

if __name__ == "__main__":
    main()
