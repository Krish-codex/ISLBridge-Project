"""
Landmark Extraction for ISL Recognition
Extracts hand, pose, and face landmarks using MediaPipe
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
            "feature_dim": 258
        }

class LandmarkExtractor:
    
    def __init__(self):
        # Import mediapipe with proper attribute access
        import mediapipe.python.solutions.holistic as mp_holistic
        self.mp_holistic = mp_holistic
        
        # Get config
        mediapipe_config, feature_config = get_mediapipe_config()
        
        # Initialize holistic with optimal settings
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=mediapipe_config.get("model_complexity", 1),
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=mediapipe_config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=mediapipe_config.get("min_tracking_confidence", 0.5)
        )
        
        # Force reduced feature dimension
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
            # Process frame with MediaPipe
            results = self.holistic.process(frame)
            
            # Extract only essential landmarks
            landmarks = self._extract_essential_features(results)
            
            if landmarks is not None:
                # Add to ring buffer
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
        
        # Hand landmarks (most important for sign language)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            features.append(left_hand)  # 21 * 3 = 63 features
        else:
            features.append(np.zeros(63, dtype=np.float32))
        
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            features.append(right_hand)  # 21 * 3 = 63 features
        else:
            features.append(np.zeros(63, dtype=np.float32))
        
        # Pose landmarks (essential upper body points only)
        if results.pose_landmarks:
            # Extract only upper body landmarks (0-10, 11-16, 23-24)
            essential_pose_indices = list(range(11, 17)) + list(range(23, 25))  # Shoulders, arms, hips
            pose_features = []
            for idx in essential_pose_indices:
                if idx < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[idx]
                    pose_features.extend([lm.x, lm.y, lm.z])
                else:
                    pose_features.extend([0.0, 0.0, 0.0])
            features.append(np.array(pose_features, dtype=np.float32))  # 8 * 3 = 24 features
        else:
            features.append(np.zeros(24, dtype=np.float32))
        
        # Face landmarks (minimal - just mouth and eyes centers)
        if results.face_landmarks:
            # Extract only key facial points for expressions
            key_face_indices = [1, 2, 5, 10, 151, 195, 197, 246]  # Mouth corners, eye centers
            face_features = []
            for idx in key_face_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    face_features.extend([lm.x, lm.y])  # Only x, y for face
                else:
                    face_features.extend([0.0, 0.0])
            features.append(np.array(face_features, dtype=np.float32))  # 8 * 2 = 16 features
        else:
            features.append(np.zeros(16, dtype=np.float32))
        
        if features:
            # Concatenate all features efficiently
            combined_features = np.concatenate(features)
            
            # Ensure consistent feature dimension
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
        
        # Get last N features from ring buffer
        if self.buffer_filled:
            if self.buffer_index >= sequence_length:
                return self.feature_buffer[self.buffer_index - sequence_length:self.buffer_index].copy()
            else:
                # Wrap around the buffer
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