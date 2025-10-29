"""
ISL Model Training Script
Trains LSTM model on A-Z, 0-9 gesture dataset
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

from model import ISLModel
from landmark_extractor import LandmarkExtractor
from config import RAW_DATA_DIR, MODELS_DIR, LOGS_DIR, ISL_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISLDatasetProcessor:
    
    def __init__(self):
        self.landmark_extractor = LandmarkExtractor()
        self.data_dir = RAW_DATA_DIR / "Frames_Word_Level"
        self.processed_data = []
        self.labels = []
        
    def load_dataset(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load and process the ISL dataset from image files
        Only include A-Z, 0-9, and basic words defined in ISL_CLASSES
        
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Loading ISL dataset from images...")
        logger.info(f"Target classes: {ISL_CLASSES}")
        
        if not self.data_dir.exists():
            logger.error(f"Dataset directory not found: {self.data_dir}")
            return [], []
        
        # Get all gesture folders
        gesture_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        logger.info(f"Found {len(gesture_folders)} total gesture folders")
        
        # Filter to only include our target classes
        target_folders = []
        for folder in gesture_folders:
            gesture_name = folder.name.upper()
            
            # Check if this gesture is in our target classes
            if gesture_name in ISL_CLASSES:
                target_folders.append(folder)
            # Handle alternative naming conventions
            elif any(gesture_name.startswith(cls) for cls in ISL_CLASSES):
                target_folders.append(folder)
        
        logger.info(f"Using {len(target_folders)} folders matching target classes")
        
        total_samples = 0
        failed_extractions = 0
        class_counts = {}
        
        for gesture_folder in target_folders:
            gesture_name = gesture_folder.name.upper()
            
            # Map gesture name to our standard class names
            mapped_class = self._map_gesture_name(gesture_name)
            
            if mapped_class not in ISL_CLASSES:
                logger.warning(f"Skipping unmapped gesture: {gesture_name}")
                continue
                
            logger.info(f"Processing gesture: {gesture_name} -> {mapped_class}")
            
            # Get all image files in the folder
            image_files = list(gesture_folder.glob("*.jpg")) + list(gesture_folder.glob("*.png"))
            
            class_samples = 0
            for image_file in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        logger.warning(f"Could not load image: {image_file}")
                        continue
                    
                    # Convert BGR to RGB for MediaPipe
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Extract landmarks
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
            
            class_counts[mapped_class] = class_samples
        
        logger.info(f"Dataset processing complete:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Failed extractions: {failed_extractions}")
        logger.info(f"  Success rate: {total_samples/(total_samples+failed_extractions)*100:.1f}%" if (total_samples+failed_extractions) > 0 else "0%")
        logger.info(f"  Class distribution: {class_counts}")
        
        return self.processed_data, self.labels
    
    def _map_gesture_name(self, gesture_name: str) -> str:
        """
        Map dataset gesture names to our standard class names
        
        Args:
            gesture_name: Original gesture name from dataset
            
        Returns:
            Mapped class name
        """
        gesture_name = gesture_name.upper().strip()
        
        # Direct mappings
        name_mappings = {
            "HELLO": "HELLO",
            "HI": "HELLO", 
            "THANK_YOU": "THANK_YOU",
            "THANKS": "THANK_YOU",
            "PLEASE": "PLEASE",
            "SORRY": "SORRY",
            "YES": "YES",
            "NO": "NO",
            "HELP": "HELP",
            "WATER": "WATER",
            "FOOD": "FOOD",
            "BATHROOM": "BATHROOM",
            "TOILET": "BATHROOM",
            "STOP": "STOP",
            "GO": "GO",
        }
        
        # Check direct mappings first
        if gesture_name in name_mappings:
            return name_mappings[gesture_name]
        
        # Check if it's already in our target classes
        if gesture_name in ISL_CLASSES:
            return gesture_name
        
        # Check for single letters A-Z
        if len(gesture_name) == 1 and gesture_name.isalpha():
            return gesture_name
        
        # Check for single digits 0-9
        if len(gesture_name) == 1 and gesture_name.isdigit():
            return gesture_name
        
        # Check for number words
        number_words = {
            "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
            "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"
        }
        
        if gesture_name in number_words:
            return number_words[gesture_name]
        
        # Return original if no mapping found
        return gesture_name
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def save_processed_data(self, filepath: str):
        """Save processed dataset for future use"""
        data = {
            'features': [feat.tolist() for feat in self.processed_data],
            'labels': self.labels,
            'feature_dim': self.processed_data[0].shape[0] if self.processed_data else 0,
            'num_samples': len(self.processed_data),
            'num_classes': len(set(self.labels))
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load previously processed dataset"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        features = [np.array(feat) for feat in data['features']]
        labels = data['labels']
        
        logger.info(f"Loaded {len(features)} samples from {filepath}")
        return features, labels

class ISLTrainer:
    """Train ISL recognition model"""
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def prepare_data(self, features: List[np.ndarray], labels: List[str], 
                    test_size: float = 0.2, val_size: float = 0.1, sequence_length: int = 30) -> Dict:
        logger.info("Preparing data for training...")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        logger.info(f"Original data shape: {X.shape}")
        logger.info(f"Feature dimension: {X.shape[1]}")
        logger.info(f"Number of classes: {len(set(y))}")
        
        # Create sequences by repeating each sample to create temporal sequences
        # This helps the LSTM learn temporal patterns
        X_sequences = []
        y_sequences = []
        
        for i, (feature, label) in enumerate(zip(X, y)):
            # Create a sequence by slightly varying the feature vector
            sequence = []
            for seq_idx in range(sequence_length):
                # Add small random noise to create variation in sequence
                noise = np.random.normal(0, 0.01, feature.shape)
                noisy_feature = feature + noise
                sequence.append(noisy_feature)
            
            X_sequences.append(np.array(sequence))
            y_sequences.append(label)
        
        X_sequences = np.array(X_sequences)  # Shape: (samples, sequence_length, features)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Sequence data shape: {X_sequences.shape}")
        logger.info(f"Sequence length: {sequence_length}")
        
        # Split data with stratification to ensure balanced classes
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_sequences, y_sequences, test_size=test_size, random_state=42, stratify=y_sequences
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            # Fallback to random split if stratification fails
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_sequences, y_sequences, test_size=test_size, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
            )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples") 
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Show class distribution
        from collections import Counter
        logger.info(f"Train class distribution: {dict(Counter(y_train))}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train_model(self, data_splits: Dict, epochs: int = 50) -> Dict:
        """
        Train the ISL model
        
        Args:
            data_splits: Dictionary with train/val/test data
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Initialize model
        num_classes = len(set(data_splits['y_train']))
        self.model = ISLModel(num_classes=num_classes)
        
        # Train model
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
        
        # Get test predictions
        test_predictions = []
        test_confidences = []
        
        for sample in data_splits['X_test']:
            prediction, confidence = self.model.predict_class(sample.reshape(1, 1, -1))
            test_predictions.append(prediction)
            test_confidences.append(confidence)
        
        # Calculate metrics
        true_labels = data_splits['y_test']
        
        # Classification report
        report = classification_report(true_labels, test_predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, test_predictions)
        
        # Calculate accuracy
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
        
        # Plot loss
        ax1.plot(self.history['loss'], label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
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
        # Limit labels for readability
        labels = class_names if len(class_names) <= 20 else class_names[:20]
        
        # Use matplotlib imshow to avoid seaborn type issues
        plt.imshow(cm[:len(labels), :len(labels)], interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
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
    """Main training function"""
    logger.info("🤟 ISL Bridge - Model Training Started")
    
    # Create necessary directories
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Initialize processor and trainer
    processor = ISLDatasetProcessor()
    trainer = ISLTrainer()
    
    # Check if processed data exists
    processed_data_path = str(RAW_DATA_DIR / "processed_dataset.json")
    
    if os.path.exists(processed_data_path):
        logger.info("Loading previously processed dataset...")
        features, labels = processor.load_processed_data(processed_data_path)
    else:
        logger.info("Processing raw dataset...")
        features, labels = processor.load_dataset()
        
        if not features:
            logger.error("No data loaded! Please check your dataset.")
            return
        
        # Save processed data for future use
        processor.save_processed_data(processed_data_path)
    
    # Show dataset statistics
    class_dist = processor.get_class_distribution()
    logger.info(f"Class distribution: {class_dist}")
    
    # Prepare data for training (ensuring enough samples for stratified splits)
    data_splits = trainer.prepare_data(features, labels, test_size=0.20, val_size=0.10)
    
    # Train model
    history = trainer.train_model(data_splits, epochs=50)
    
    # Evaluate model
    eval_results = trainer.evaluate_model(data_splits)
    
    # Save model
    trainer.save_model()
    
    # Generate plots
    plots_dir = LOGS_DIR / "training_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot training history
    trainer.plot_training_history(str(plots_dir / "training_history.png"))
    
    # Plot confusion matrix
    if 'confusion_matrix' in eval_results:
        unique_labels = sorted(list(set(labels)))
        trainer.plot_confusion_matrix(
            eval_results['confusion_matrix'], 
            unique_labels,
            str(plots_dir / "confusion_matrix.png")
        )
    
    # Save evaluation results
    eval_results_path = str(LOGS_DIR / "evaluation_results.json")
    
    # Convert numpy arrays and float32 values to JSON-serializable types
    eval_results_json = eval_results.copy()
    for key, value in eval_results_json.items():
        if hasattr(value, 'tolist'):  # numpy array
            eval_results_json[key] = value.tolist()
        elif hasattr(value, 'item'):  # numpy scalar
            eval_results_json[key] = value.item()
        elif str(type(value).__name__) in ['float32', 'float64']:
            eval_results_json[key] = float(value)
        elif str(type(value).__name__) in ['int32', 'int64']:
            eval_results_json[key] = int(value)
    
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results_json, f, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_results_path}")
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"Final test accuracy: {eval_results['accuracy']:.3f}")

if __name__ == "__main__":
    main()