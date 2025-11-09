"""
LSTM Model for ISL Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def get_config():
    from config import MODELS_DIR, MODEL_CONFIG, ISL_CLASSES
    return MODELS_DIR, MODEL_CONFIG, ISL_CLASSES

class GestureRecognitionLSTM(nn.Module):
    """LSTM neural network for gesture recognition"""
    
    def __init__(self, input_dim: int = 166, hidden_dim: int = 128, num_classes: int = 36, num_layers: int = 2):
        super(GestureRecognitionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        
        x = hidden[-1]
        x = self.batch_norm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        
        return output

class ISLModel:
    
    def __init__(self, num_classes: Optional[int] = None):
        _, MODEL_CONFIG, ISL_CLASSES = get_config()
        
        self.num_classes = num_classes or len(ISL_CLASSES)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.history = None
        self.model = None
        self._build_model()
        
    def _build_model(self):
        _, MODEL_CONFIG, _ = get_config()
        
        self.model = GestureRecognitionLSTM(
            input_dim=MODEL_CONFIG.get('input_dim', 166),
            hidden_dim=MODEL_CONFIG.get('hidden_dim', 128),
            num_classes=self.num_classes,
            num_layers=MODEL_CONFIG.get('num_layers', 2)
        ).to(self.device)
        
    def build_model(self, input_shape: Optional[Tuple] = None):
        return self.model
        
    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not initialized"
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model: GestureRecognitionLSTM\nTotal Parameters: {total_params:,}\nTrainable Parameters: {trainable_params:,}"
        
    def train(self, X: np.ndarray, y: List[str], validation_data: Optional[Tuple] = None) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Use smaller batch size for large datasets to prevent memory issues
        batch_size = 16 if len(X) > 50000 else 32
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)
        
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        best_accuracy = 0.0
        
        print(f"Training with {len(X)} samples, batch size: {batch_size}")
        
        for epoch in range(100):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total
            avg_loss = epoch_loss / len(dataloader)
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            scheduler.step(avg_loss)
            
            # Save checkpoint if this is the best model so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                try:
                    from pathlib import Path
                    checkpoint_dir = Path("models")
                    checkpoint_dir.mkdir(exist_ok=True)
                    checkpoint_path = checkpoint_dir / "training_checkpoint.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy,
                        'loss': avg_loss,
                    }, checkpoint_path)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f})')
            
            # GPU memory cleanup during training
            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()
            
            if accuracy > 0.85:
                print(f'Early stopping at epoch {epoch} with accuracy {accuracy:.4f}')
                break
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.history = history
        print(f'Training complete! Best accuracy: {best_accuracy:.4f}')
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # GPU memory cleanup after prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return probabilities.cpu().numpy()
    
    def predict_class(self, X: np.ndarray, threshold: Optional[float] = None) -> Tuple[str, float]:
        try:
            probabilities = self.predict(X)
            
            if len(probabilities.shape) > 1:
                probabilities = probabilities[0]
                
            max_prob_idx = np.argmax(probabilities)
            max_prob = probabilities[max_prob_idx]
            
            if threshold and max_prob < threshold:
                return "uncertain", max_prob
            
            if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                predicted_class = self.label_encoder.inverse_transform([max_prob_idx])[0]
            else:
                _, _, ISL_CLASSES = get_config()
                if max_prob_idx < len(ISL_CLASSES):
                    predicted_class = ISL_CLASSES[max_prob_idx]
                else:
                    predicted_class = f"class_{max_prob_idx}"
            
            return predicted_class, max_prob
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Prediction error: {e}", exc_info=True)
            return "unknown", 0.0
    
    def save_model(self, filepath: Optional[str] = None):
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        MODELS_DIR, _, _ = get_config()
        
        if filepath is None:
            filepath = str(MODELS_DIR / "isl_trained_model")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes,
                'num_layers': self.model.num_layers
            },
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
        }, f"{filepath}.pth")
        
        print(f"Model saved to {filepath}.pth")
    
    def load_model(self, filepath: Optional[str] = None) -> Optional[List[str]]:
        """
        Load model from checkpoint file
        
        Returns:
            List of label classes if available, None otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        MODELS_DIR, _, _ = get_config()
        
        if filepath is None:
            filepath = str(MODELS_DIR / "isl_trained_model")
        
        try:
            checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
            
            # Validate checkpoint structure
            if 'model_config' not in checkpoint:
                raise ValueError("Invalid checkpoint: missing 'model_config'")
            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint: missing 'model_state_dict'")
            
            config = checkpoint['model_config']
            self.model = GestureRecognitionLSTM(**config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            label_classes = None
            if checkpoint.get('label_encoder_classes'):
                label_classes = checkpoint['label_encoder_classes']
                self.label_encoder.classes_ = np.array(label_classes)
                logger.info(f"Loaded {len(label_classes)} label classes")
            else:
                logger.warning("No label classes found in checkpoint")
            
            logger.info(f"Model loaded from {filepath}.pth")
            return label_classes
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}.pth")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def get_model_size(self) -> Dict[str, float]:
        if self.model is None:
            return {'total_mb': 0.0, 'parameters': 0.0, 'trainable_parameters': 0.0}
            
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = param_size + buffer_size
        return {
            'total_mb': model_size / 1024 / 1024,
            'parameters': float(sum(p.numel() for p in self.model.parameters())),
            'trainable_parameters': float(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        }
