"""
Landmark Extraction for ISL Recognition
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mediapipe_config():
    try:
        from config import MEDIAPIPE_CONFIG, FEATURE_CONFIG
        return MEDIAPIPE_CONFIG, FEATURE_CONFIG
    except ImportError:
        return {
            "model_complexity": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        }, {
            "buffer_size": 30,
            "feature_dim": 166
        }

class LandmarkExtractor:
    """Extract hand, pose, and face landmarks using MediaPipe"""
    
    def __init__(self):
        import mediapipe.python.solutions.holistic as mp_holistic
        self.mp_holistic = mp_holistic
        
        mediapipe_config, feature_config = get_mediapipe_config()
        
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=mediapipe_config.get("model_complexity", 1),
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=mediapipe_config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=mediapipe_config.get("min_tracking_confidence", 0.5)
        )
        
        self.feature_dim = 166
        self.buffer_size = feature_config.get("buffer_size", 30)
        self.feature_buffer = np.zeros((self.buffer_size, self.feature_dim), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_filled = False
        
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract optimized landmarks from frame
        
        Args:
            frame: Input frame (RGB format)
            
        Returns:
            Optimized landmark features or None
        """
        try:
            results = self.holistic.process(frame)
            
            landmarks = self._extract_essential_features(results)
            
            if landmarks is not None:
                self.feature_buffer[self.buffer_index] = landmarks
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                if not self.buffer_filled and self.buffer_index == 0:
                    self.buffer_filled = True
                
                return landmarks
            
            return None
            
        except Exception as e:
            logger.error(f"Landmark extraction error: {e}")
            return None
    
    def _extract_essential_features(self, results) -> Optional[np.ndarray]:
        """Extract only essential features for efficiency"""
        features = []
        
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            features.append(left_hand)
        else:
            features.append(np.zeros(63, dtype=np.float32))
        
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            features.append(right_hand)
        else:
            features.append(np.zeros(63, dtype=np.float32))
        
        if results.pose_landmarks:
            essential_pose_indices = list(range(11, 17)) + list(range(23, 25))
            pose_features = []
            for idx in essential_pose_indices:
                if idx < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[idx]
                    pose_features.extend([lm.x, lm.y, lm.z])
                else:
                    pose_features.extend([0.0, 0.0, 0.0])
            features.append(np.array(pose_features, dtype=np.float32))
        else:
            features.append(np.zeros(24, dtype=np.float32))
        
        if results.face_landmarks:
            key_face_indices = [1, 2, 5, 10, 151, 195, 197, 246]
            face_features = []
            for idx in key_face_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    face_features.extend([lm.x, lm.y])
                else:
                    face_features.extend([0.0, 0.0])
            features.append(np.array(face_features, dtype=np.float32))
        else:
            features.append(np.zeros(16, dtype=np.float32))
        
        if features:
            combined_features = np.concatenate(features)
            
            if len(combined_features) > self.feature_dim:
                combined_features = combined_features[:self.feature_dim]
            elif len(combined_features) < self.feature_dim:
                padding = np.zeros(self.feature_dim - len(combined_features), dtype=np.float32)
                combined_features = np.concatenate([combined_features, padding])
            
            return combined_features
        
        return None
    
    def get_sequence_features(self, sequence_length: int = 10) -> Optional[np.ndarray]:
        """Get recent sequence features efficiently"""
        if not self.buffer_filled and self.buffer_index < sequence_length:
            return None
        
        if self.buffer_filled:
            if self.buffer_index >= sequence_length:
                return self.feature_buffer[self.buffer_index - sequence_length:self.buffer_index].copy()
            else:
                part1 = self.feature_buffer[self.buffer_size - (sequence_length - self.buffer_index):]
                part2 = self.feature_buffer[:self.buffer_index]
                return np.vstack([part1, part2])
        else:
            return self.feature_buffer[:self.buffer_index][-sequence_length:].copy()
    
    def get_memory_usage(self) -> dict:
        """Get memory usage information"""
        buffer_size_mb = self.feature_buffer.nbytes / (1024 * 1024)
        return {
            'buffer_size_mb': buffer_size_mb,
            'feature_dimension': self.feature_dim,
            'buffer_length': self.buffer_size,
            'current_index': self.buffer_index
        }
    
    def reset_buffer(self):
        """Reset the feature buffer"""
        self.feature_buffer.fill(0)
        self.buffer_index = 0
        self.buffer_filled = False
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()