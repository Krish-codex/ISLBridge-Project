"""
ISL Bridge Desktop Application
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import logging

from model import ISLModel
from landmark_extractor import LandmarkExtractor
from enhanced_translation import MultiLanguageTranslator
from config import MODEL_CONFIG, APP_CONFIG
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ISLBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 ISL Bridge - Sign Language Translator [RUNNING]")
        self.root.withdraw()
        
        # Configure window properties
        self.root.configure(bg='#f5f5f5')
        
        # Make window always on top and prominent
        self.root.attributes('-topmost', True)
        
        # Set as a tool window to make it independent
        try:
            self.root.attributes('-toolwindow', False)  # Ensure it's a normal window
        except:
            pass
        
        # Center window on screen with specific size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1200
        window_height = 700
        x = (screen_width - window_width) // 2
        y = max(50, (screen_height - window_height) // 2 - 100)  # Position slightly higher
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Core components
        self.settings = MODEL_CONFIG
        self.extractor = None
        self.recognition_model = None
        self.model_ready = False
        self.model_load_lock = threading.Lock()  # Thread-safe model loading
        self.translator = None
        self.available_classes = []  # Will be loaded from trained model
        
        # Camera
        self.camera = None
        self.recording = False
        
        # Recognition buffers
        self.gesture_buffer = []
        self.previous_sign = ""
        self.sentence_words = []
        self.selected_language = "en"
        
        # Performance optimization: frame skipping
        self.frame_counter = 0
        self.prediction_interval = APP_CONFIG.get("prediction_interval", 5)  # From config
        
        # Previous sign tracking with timeout
        self.last_prediction_time = 0
        self.prediction_timeout = APP_CONFIG.get("prediction_timeout", 2.0)  # From config
        
        # Build interface
        self.setup_modern_ui()
        
        # Load AI model
        self.load_model_async()
        
        # Ensure window is visible after a moment
        self.root.after(200, self.ensure_window_visible)
    
    def ensure_window_visible(self):
        """Make window visible with smooth transition (no flashing)"""
        try:
            # Show the window smoothly
            self.root.deiconify()  # Make visible
            self.root.lift()
            self.root.focus_force()
            self.root.attributes('-topmost', True)
            
            # Play system sound to notify user
            self.root.bell()
            
            # Show a small notification
            self.root.after(500, self.show_startup_message)
        except Exception as e:
            logger.error(f"Window visibility error: {e}", exc_info=True)
    
    def show_startup_message(self):
        """Show startup notification"""
        try:
            # Create a temporary label overlay
            splash = tk.Label(
                self.root,
                text="🎉 ISL Bridge is READY!\n\nClick 'Start Camera' to begin",
                font=('Arial', 16, 'bold'),
                bg='#2ecc71',
                fg='white',
                padx=30,
                pady=20
            )
            splash.place(relx=0.5, rely=0.5, anchor='center')
            
            # Remove after 3 seconds
            self.root.after(3000, splash.destroy)
        except:
            pass
    
    def setup_modern_ui(self):
        # Header section
        header = tk.Frame(self.root, bg='#2c3e50', height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header, 
            text="🤟 ISL Bridge",
            font=('Arial', 28, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=30, pady=20)
        
        subtitle_label = tk.Label(
            header,
            text="Indian Sign Language Translator",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack(side=tk.LEFT, padx=10)
        
        # Language selector and status
        controls = tk.Frame(header, bg='#2c3e50')
        controls.pack(side=tk.RIGHT, padx=30)
        
        # Language selection
        lang_label = tk.Label(
            controls,
            text="Language:",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        lang_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.language_var = tk.StringVar(value="en")
        self.language_combo = ttk.Combobox(
            controls,
            textvariable=self.language_var,
            values=["en", "hi", "ta", "te"],
            state="readonly",
            width=8,
            font=('Arial', 9)
        )
        self.language_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.language_combo.bind('<<ComboboxSelected>>', self.on_language_change)
        
        self.status_indicator = tk.Label(
            controls,
            text="● Ready",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='#2ecc71'
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 15))
        
        # Always on top toggle
        self.always_on_top_var = tk.BooleanVar(value=True)
        always_on_top_check = tk.Checkbutton(
            controls,
            text="📌 Always on Top",
            variable=self.always_on_top_var,
            command=self.toggle_always_on_top,
            font=('Arial', 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            selectcolor='#34495e',
            activebackground='#2c3e50',
            activeforeground='#ecf0f1'
        )
        always_on_top_check.pack(side=tk.LEFT)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#f5f5f5')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left: Camera feed
        left_frame = tk.Frame(content_frame, bg='#f5f5f5')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        camera_label = tk.Label(
            left_frame,
            text="📹 Camera Feed",
            font=('Arial', 14, 'bold'),
            bg='#f5f5f5',
            fg='#2c3e50'
        )
        camera_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Camera display with border
        camera_container = tk.Frame(left_frame, bg='#34495e', bd=2, relief=tk.SOLID)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(
            camera_container,
            text="📷\n\nClick 'Start Camera' to begin\n\nShow your ISL signs here",
            font=('Arial', 16),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera controls
        controls_frame = tk.Frame(left_frame, bg='#f5f5f5')
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = tk.Button(
            controls_frame,
            text="🎥 Start Camera",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            command=self.toggle_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_button = tk.Button(
            controls_frame,
            text="🗑️ Clear",
            font=('Arial', 12),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            relief=tk.FLAT,
            padx=15,
            pady=10,
            cursor='hand2',
            command=self.clear_sentence
        )
        clear_button.pack(side=tk.LEFT)
        
        # Right: Results panel
        right_frame = tk.Frame(content_frame, bg='white', width=400, relief=tk.RAISED, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Current sign
        result_label = tk.Label(
            right_frame,
            text="Current Sign",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        result_label.pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        self.prediction_frame = tk.Frame(right_frame, bg='#ecf0f1', height=100)
        self.prediction_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        self.prediction_frame.pack_propagate(False)
        
        self.prediction_var = tk.StringVar(value="—")
        self.prediction_label = tk.Label(
            self.prediction_frame,
            textvariable=self.prediction_var,
            font=('Arial', 48, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.prediction_label.pack(expand=True)
        
        # Confidence
        confidence_label = tk.Label(
            right_frame,
            text="Confidence",
            font=('Arial', 11),
            bg='white',
            fg='#7f8c8d'
        )
        confidence_label.pack(anchor=tk.W, padx=20)
        
        self.confidence_var = tk.StringVar(value="0%")
        confidence_value = tk.Label(
            right_frame,
            textvariable=self.confidence_var,
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#27ae60'
        )
        confidence_value.pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        self.confidence_bar = ttk.Progressbar(
            right_frame,
            mode='determinate',
            length=360
        )
        self.confidence_bar.pack(padx=20, pady=(0, 20))
        
        # Sentence builder
        sentence_label = tk.Label(
            right_frame,
            text="📝 Your Message",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        sentence_label.pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        sentence_container = tk.Frame(right_frame, bg='#f8f9fa', relief=tk.SUNKEN, bd=1)
        sentence_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.sentence_text = tk.Text(
            sentence_container,
            font=('Arial', 14),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.sentence_text.pack(fill=tk.BOTH, expand=True)
        
        # Action buttons
        action_frame = tk.Frame(right_frame, bg='white')
        action_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Add Sign button (new - for manual control)
        add_sign_button = tk.Button(
            action_frame,
            text="➕ Add Sign",
            font=('Arial', 10, 'bold'),
            bg='#27ae60',  # Green to stand out
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self.add_current_sign_to_sentence
        )
        add_sign_button.pack(side=tk.LEFT, padx=(0, 5))
        
        space_button = tk.Button(
            action_frame,
            text="Space",
            font=('Arial', 10),
            bg='#34495e',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self.add_space
        )
        space_button.pack(side=tk.LEFT, padx=(0, 5))
        
        backspace_button = tk.Button(
            action_frame,
            text="⌫ Backspace",
            font=('Arial', 10),
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self.backspace
        )
        backspace_button.pack(side=tk.LEFT)
        
        speak_button = tk.Button(
            action_frame,
            text="🔊 Speak",
            font=('Arial', 10),
            bg='#27ae60',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self.speak_sentence
        )
        speak_button.pack(side=tk.LEFT, padx=(10, 0))
        
        translate_button = tk.Button(
            action_frame,
            text="🌐 Translate",
            font=('Arial', 10),
            bg='#9b59b6',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self.translate_sentence
        )
        translate_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Supported signs info
        info_frame = tk.Frame(right_frame, bg='#e8f5e9', relief=tk.FLAT)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        info_text = "⏳ Loading model...\nSupported classes will appear here."
        self.info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 9),
            bg='#e8f5e9',
            fg='#2e7d32',
            justify=tk.LEFT
        )
        self.info_label.pack(padx=10, pady=10)
    
    def load_model_async(self):
        def load():
            with self.model_load_lock:  # Thread-safe model loading
                try:
                    self.root.after(0, lambda: self.status_indicator.config(text="● Loading...", fg='#f39c12'))
                    
                    self.extractor = LandmarkExtractor()
                    self.recognition_model = ISLModel()
                    self.translator = MultiLanguageTranslator()
                    
                    # Notify user if translation is unavailable
                    if not self.translator.translation_available:
                        logger.warning("Translation service unavailable. Using local dictionary only.")
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Translation Limited",
                            "Online translation service is not available.\n\n"
                            "Install deep-translator for full multi-language support:\n"
                            "pip install deep-translator\n\n"
                            "Local gesture translations will still work."
                        ))
                    
                    model_path = Path("models/isl_trained_model.pth")
                    if not model_path.exists():
                        self.root.after(0, lambda: messagebox.showerror("Error", 
                            "No trained model found!\n\nPlease train the model first:\npython train_model.py"))
                        self.root.after(0, lambda: self.status_indicator.config(text="● No Model", fg='#e74c3c'))
                        return
                    
                    # Validate checkpoint file
                    try:
                        # Single canonical load - returns label classes
                        label_classes = self.recognition_model.load_model("models/isl_trained_model")
                        
                        if label_classes:
                            self.available_classes = label_classes
                            self.root.after(0, lambda: self.update_info_text())
                        else:
                            # Fallback: use config classes
                            from config import ISL_CLASSES
                            self.available_classes = ISL_CLASSES
                            logger.warning("Using fallback classes from config")
                        
                        self.model_ready = True
                        self.root.after(0, lambda: self.status_indicator.config(text="● Ready", fg='#2ecc71'))
                        logger.info(f"Model loaded successfully with {len(self.available_classes)} classes: {self.available_classes}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load model checkpoint: {e}", exc_info=True)
                        self.root.after(0, lambda: messagebox.showerror("Error", 
                            f"Failed to load model checkpoint!\n\nThe model file may be corrupted.\n\n"
                            f"Error: {str(e)}\n\nPlease retrain the model:\npython train_model.py"))
                        self.root.after(0, lambda: self.status_indicator.config(text="● Error", fg='#e74c3c'))
                        
                except Exception as e:
                    logger.error(f"Failed to load model: {e}", exc_info=True)
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
                    self.root.after(0, lambda: self.status_indicator.config(text="● Error", fg='#e74c3c'))
        
        threading.Thread(target=load, daemon=True).start()
    
    def update_info_text(self):
        """Update the info label with dynamically loaded classes"""
        if not self.available_classes:
            return
        
        # Categorize classes
        numbers = [c for c in self.available_classes if c.isdigit()]
        letters = [c for c in self.available_classes if len(c) == 1 and c.isalpha()]
        words = [c for c in self.available_classes if len(c) > 1]
        
        # Build info text
        parts = []
        if numbers:
            parts.append(f"Numbers: {', '.join(sorted(numbers, key=int))}")
        if letters:
            # Sort letters to ensure proper range display
            letters_sorted = sorted(letters)
            parts.append(f"Letters: {letters_sorted[0]}-{letters_sorted[-1]}")
        if words:
            word_preview = ', '.join(words[:5])
            if len(words) > 5:
                word_preview += f", +{len(words)-5} more"
            parts.append(f"Words: {word_preview}")
        
        info_text = f"✅ Supports {len(self.available_classes)} gestures:\n"
        info_text += "\n".join(f"  • {part}" for part in parts)
        info_text += "\n🌐 Multi-language: English, Hindi, Tamil, Telugu"
        
        self.info_label.config(text=info_text)
    
    def toggle_always_on_top(self):
        """Toggle window always on top state"""
        is_on_top = self.always_on_top_var.get()
        self.root.attributes('-topmost', is_on_top)
        if is_on_top:
            self.root.lift()
    
    def toggle_camera(self):
        if not self.recording:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        if not self.model_ready:
            messagebox.showwarning("Warning", "Model is still loading. Please wait...")
            return
        
        try:
            # Use configurable camera indices
            camera_indices = APP_CONFIG.get("camera_retry_indices", [1, 2, 0])
            camera_opened = False
            
            for cam_index in camera_indices:
                logger.info(f"Trying camera index {cam_index}...")
                self.camera = cv2.VideoCapture(cam_index)
                
                if self.camera.isOpened():
                    # Test if camera actually works by reading a frame
                    ret, test_frame = self.camera.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        logger.info(f"✓ Camera {cam_index} opened and tested successfully!")
                        camera_opened = True
                        break
                    else:
                        logger.warning(f"✗ Camera {cam_index} opened but cannot read frames")
                        self.camera.release()
                else:
                    logger.debug(f"✗ Camera {cam_index} not available")
                    if self.camera:
                        self.camera.release()
            
            if not camera_opened:
                error_message = (
                    "Cannot open camera!\n\n"
                    f"Tried camera indices: {camera_indices}\n\n"
                    "Troubleshooting:\n"
                    "1. Check if another app is using the camera\n"
                    "2. Verify camera permissions are enabled\n"
                    "3. Ensure camera drivers are installed\n"
                    "4. Try disconnecting/reconnecting the camera"
                )
                messagebox.showerror("Camera Error", error_message)
                logger.error("Failed to open any camera")
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.recording = True
            self.start_button.config(text="⏹️ Stop Camera", bg='#e74c3c', activebackground='#c0392b')
            self.status_indicator.config(text="● Recording", fg='#e74c3c')
            self.capture_frames()
            
        except Exception as e:
            error_msg = f"Camera error: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg, exc_info=True)
    
    def stop_camera(self):
        self.recording = False
        if self.camera:
            self.camera.release()
            self.camera = None  # Clear reference for memory cleanup
        self.start_button.config(text="🎥 Start Camera", bg='#3498db', activebackground='#2980b9')
        self.status_indicator.config(text="● Ready", fg='#2ecc71')
        self.video_label.config(image='', text="📷\n\nClick 'Start Camera' to begin\n\nShow your ISL signs here")
        
        # Clear video frame reference for memory cleanup
        if hasattr(self, '_video_image_ref'):
            self._video_image_ref = None
        
        self.gesture_buffer.clear()
        
        # GPU memory cleanup if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    
    def capture_frames(self):
        if not self.recording or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()  # frame is BGR from OpenCV
            if ret:
                # Convert to RGB once here for both display and processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pass RGB frame to display
                self.display_frame(frame_rgb)
                
                # Pass RGB frame to processing (MediaPipe requires RGB)
                if self.model_ready and self.extractor and self.recognition_model:
                    self.process_frame(frame_rgb)
            
            # Use configurable frame capture interval
            interval = APP_CONFIG.get("frame_capture_interval", 30)
            self.root.after(interval, self.capture_frames)
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}", exc_info=True)
    
    def display_frame(self, frame_rgb):
        """Display RGB frame in the video label"""
        try:
            # frame_rgb is already RGB, no conversion needed
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            self.video_label.config(image=photo, text="")
            self._video_image_ref = photo  
        except Exception as e:
            logger.error(f"Display error: {e}")
    
    def process_frame(self, frame_rgb):
        """Process RGB frame for gesture recognition"""
        try:
            if not self.extractor or not self.recognition_model or not self.model_ready:
                return
                
            # Extract landmarks from RGB frame (MediaPipe requires RGB)
            landmarks = self.extractor.extract_landmarks(frame_rgb)
            
            if landmarks is not None:
                self.gesture_buffer.append(landmarks)
                
                seq_length = self.settings.get("sequence_length", 30)
                if len(self.gesture_buffer) > seq_length:
                    self.gesture_buffer.pop(0)
                
                # Performance optimization: Only predict every Nth frame
                self.frame_counter += 1
                if len(self.gesture_buffer) == seq_length and (self.frame_counter % self.prediction_interval == 0):
                    try:
                        sequence_array = np.array(self.gesture_buffer).reshape(1, seq_length, -1)
                        result = self.recognition_model.predict_class(sequence_array, threshold=0.55)
                        
                        if result and isinstance(result, (tuple, list)) and len(result) >= 2:
                            predicted_sign, confidence_score = result[0], result[1]
                            
                            if predicted_sign not in ["uncertain", "unknown"]:
                                self.update_prediction(predicted_sign, confidence_score)
                    
                    except Exception as e:
                        logger.error(f"Prediction error: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
    
    def update_prediction(self, predicted_sign, confidence_score):
        """Update prediction display without automatically adding to sentence"""
        import time
        current_time = time.time()
        
        # Reset previous_sign if enough time has passed (allows re-recognition of same sign)
        if current_time - self.last_prediction_time > self.prediction_timeout:
            self.previous_sign = ""
        
        if predicted_sign != self.previous_sign and confidence_score > 0.55:
            self.previous_sign = predicted_sign
            self.last_prediction_time = current_time
            self.prediction_var.set(predicted_sign)
            self.confidence_var.set(f"{float(confidence_score):.0%}")
            self.confidence_bar['value'] = float(confidence_score) * 100
    
    def add_current_sign_to_sentence(self):
        """Adds the sign currently in the prediction box to the sentence"""
        sign = self.prediction_var.get()
        if sign and sign != "—":
            self.add_to_sentence(sign)
            self.add_space()  # Automatically add space after adding sign
    
    def add_to_sentence(self, sign):
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        
        if self.translator and hasattr(self.translator, 'translate_gesture'):
            translated_text = self.translator.translate_gesture(sign, self.selected_language)
        else:
            translated_text = sign
        
        # Multi-character words (like HELLO, THANK_YOU) get spaces around them
        # Single characters (letters/numbers) get concatenated directly
        if len(sign) > 1:  # It's a word, not a letter or digit
            new_text = current_text + " " + translated_text if current_text else translated_text
        else:
            new_text = current_text + translated_text
        
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert("1.0", new_text)
    
    def add_space(self):
        """Add space to sentence"""
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert("1.0", current_text + " ")
    
    def backspace(self):
        """Remove last character"""
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if current_text:
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", current_text[:-1])
    
    def safe_speak_text(self, text):
        try:
            if self.translator and hasattr(self.translator, 'speak_text'):
                # Use translate_sentence for full sentences, translate_gesture for single gestures
                if len(text.split()) > 1:
                    translated_text = self.translator.translate_sentence(text, self.selected_language)
                else:
                    translated_text = self.translator.translate_gesture(text, self.selected_language)
                self.translator.speak_text(translated_text, self.selected_language)
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
    
    def _do_speak(self, text):
        self.safe_speak_text(text)
    
    def on_language_change(self, event=None):
        new_lang = self.language_var.get()
        self.selected_language = new_lang
        
        if self.translator and hasattr(self.translator, 'set_language'):
            self.translator.set_language(new_lang)
        
        if self.sentence_text:
            current_text = self.sentence_text.get("1.0", tk.END).strip()
            if current_text and self.translator:
                translated_sentence = self.translator.translate_sentence(current_text, new_lang)
                self.sentence_text.delete("1.0", tk.END)
                self.sentence_text.insert("1.0", translated_sentence)
    
    def speak_sentence(self):
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        text_to_speak = current_text if current_text else "No text to speak"
        self.safe_speak_text(text_to_speak)
    
    def translate_sentence(self):
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if current_text and self.translator and hasattr(self.translator, 'translate_sentence'):
            translated = self.translator.translate_sentence(current_text, self.selected_language)
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", translated)
    
    def clear_sentence(self):
        self.sentence_text.delete("1.0", tk.END)
        self.prediction_var.set("—")
        self.confidence_var.set("0%")
        self.confidence_bar['value'] = 0
        self.previous_sign = ""
        self.gesture_buffer.clear()
    
    def on_closing(self):
        """Handle app closing with proper resource cleanup"""
        try:
            self.stop_camera()
            
            # Cleanup MediaPipe resources
            if self.extractor:
                try:
                    self.extractor.close()
                    logger.info("MediaPipe resources cleaned up")
                except Exception as e:
                    logger.warning(f"Error closing extractor: {e}")
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
            
            self.root.destroy()
            logger.info("Application closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            self.root.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()
    app = ISLBridgeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
