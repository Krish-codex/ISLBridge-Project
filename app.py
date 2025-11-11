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
from gpu_utils import cleanup_gpu_memory
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
        
        # Set as a normal window (not always on top)
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
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header with gradient effect
        header = tk.Frame(self.root, bg='#1a252f', height=90)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header, 
            text="🤟 ISL Bridge",
            font=('Segoe UI', 32, 'bold'),
            bg='#1a252f',
            fg='#ffffff'
        )
        title_label.pack(side=tk.LEFT, padx=40, pady=25)
        
        subtitle_label = tk.Label(
            header,
            text="Real-Time Sign Language Translation",
            font=('Segoe UI', 11),
            bg='#1a252f',
            fg='#95a5a6'
        )
        subtitle_label.pack(side=tk.LEFT, padx=5)
        
        # Language and status controls
        controls = tk.Frame(header, bg='#1a252f')
        controls.pack(side=tk.RIGHT, padx=40)
        
        lang_label = tk.Label(
            controls,
            text="Language:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a252f',
            fg='#ecf0f1'
        )
        lang_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.language_var = tk.StringVar(value="en")
        self.language_combo = ttk.Combobox(
            controls,
            textvariable=self.language_var,
            values=["en", "hi", "ta", "te", "bn", "gu", "mr", "pa"],
            state="readonly",
            width=10,
            font=('Segoe UI', 9)
        )
        self.language_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.language_combo.bind('<<ComboboxSelected>>', self.on_language_change)
        
        self.status_indicator = tk.Label(
            controls,
            text="● Ready",
            font=('Segoe UI', 12, 'bold'),
            bg='#1a252f',
            fg='#2ecc71'
        )
        self.status_indicator.pack(side=tk.LEFT)
        
        # Main content with better layout
        content_frame = tk.Frame(self.root, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        # Left: Camera section
        left_frame = tk.Frame(content_frame, bg='#ecf0f1')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        camera_header = tk.Frame(left_frame, bg='#34495e', height=45)
        camera_header.pack(fill=tk.X)
        camera_header.pack_propagate(False)
        
        camera_label = tk.Label(
            camera_header,
            text="📹  Live Camera Feed",
            font=('Segoe UI', 13, 'bold'),
            bg='#34495e',
            fg='white'
        )
        camera_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Camera display with modern border
        camera_container = tk.Frame(left_frame, bg='#2c3e50', bd=3, relief=tk.RAISED)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(
            camera_container,
            text="📷\n\nClick 'Start Camera' to begin\n\nPosition your hands in view for sign recognition",
            font=('Segoe UI', 14),
            bg='#ffffff',
            fg='#7f8c8d',
            justify=tk.CENTER
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Camera controls with better spacing
        controls_frame = tk.Frame(left_frame, bg='#ecf0f1')
        controls_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.start_button = tk.Button(
            controls_frame,
            text="🎥 Start Camera",
            font=('Segoe UI', 13, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            padx=25,
            pady=12,
            cursor='hand2',
            command=self.toggle_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 12))
        
        clear_button = tk.Button(
            controls_frame,
            text="🗑️ Clear All",
            font=('Segoe UI', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            command=self.clear_sentence
        )
        clear_button.pack(side=tk.LEFT)
        
        # Right: Recognition panel with card design
        right_frame = tk.Frame(content_frame, bg='#ffffff', width=450, relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_frame.pack_propagate(False)
        
        # Prediction section with modern card
        pred_header = tk.Frame(right_frame, bg='#3498db', height=40)
        pred_header.pack(fill=tk.X)
        pred_header.pack_propagate(False)
        
        pred_title = tk.Label(
            pred_header,
            text="🎯 Current Sign",
            font=('Segoe UI', 12, 'bold'),
            bg='#3498db',
            fg='white'
        )
        pred_title.pack(side=tk.LEFT, padx=20, pady=8)
        
        pred_card = tk.Frame(right_frame, bg='#ffffff')
        pred_card.pack(fill=tk.X, padx=20, pady=15)
        
        self.prediction_var = tk.StringVar(value="—")
        self.prediction_label = tk.Label(
            pred_card,
            textvariable=self.prediction_var,
            font=('Segoe UI', 48, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            height=2
        )
        self.prediction_label.pack()
        
        # Store for backward compatibility
        self.prediction_frame = pred_card
        
        # Confidence display
        conf_frame = tk.Frame(pred_card, bg='#ffffff')
        conf_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(
            conf_frame,
            text="Confidence:",
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#7f8c8d'
        ).pack(side=tk.LEFT, padx=(10, 5))
        
        self.confidence_var = tk.StringVar(value="0%")
        tk.Label(
            conf_frame,
            textvariable=self.confidence_var,
            font=('Segoe UI', 10, 'bold'),
            bg='#ffffff',
            fg='#27ae60'
        ).pack(side=tk.LEFT)
        
        # Progress bar
        self.confidence_bar = ttk.Progressbar(
            pred_card,
            mode='determinate',
            length=400,
            style='green.Horizontal.TProgressbar'
        )
        self.confidence_bar.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Configure progressbar style
        style.configure('green.Horizontal.TProgressbar', background='#27ae60', thickness=8)
        
        # Message section with card design
        msg_header = tk.Frame(right_frame, bg='#9b59b6', height=40)
        msg_header.pack(fill=tk.X, pady=(10, 0))
        msg_header.pack_propagate(False)
        
        msg_title = tk.Label(
            msg_header,
            text="📝 Your Message",
            font=('Segoe UI', 12, 'bold'),
            bg='#9b59b6',
            fg='white'
        )
        msg_title.pack(side=tk.LEFT, padx=20, pady=8)
        
        sentence_container = tk.Frame(right_frame, bg='#f8f9fa', relief=tk.SUNKEN, bd=2)
        sentence_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.sentence_text = tk.Text(
            sentence_container,
            font=('Segoe UI', 13),
            bg='#ffffff',
            fg='#2c3e50',
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            height=5,
            borderwidth=0
        )
        self.sentence_text.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder
        self.sentence_text.insert("1.0", "Start signing and click '➕ Add Sign' to build your message...\n\n✨ Tips:\n• Hold signs steady\n• Click Add Sign button\n• Use Auto for instant speech")
        self.sentence_text.config(fg='#95a5a6')
        
        # Bind events to handle placeholder
        self.sentence_text.bind("<FocusIn>", self.clear_placeholder)
        self.sentence_text.bind("<FocusOut>", self.restore_placeholder)
        self.is_placeholder = True
        
        # Action buttons - Row 1
        action_frame1 = tk.Frame(right_frame, bg='#ffffff')
        action_frame1.pack(fill=tk.X, padx=20, pady=(0, 8))
        
        add_sign_button = tk.Button(
            action_frame1,
            text="➕ Add Sign",
            font=('Segoe UI', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=10,
            cursor='hand2',
            command=self.add_current_sign_to_sentence
        )
        add_sign_button.pack(side=tk.LEFT, padx=(0, 6))
        
        space_button = tk.Button(
            action_frame1,
            text="⎵ Space",
            font=('Segoe UI', 11),
            bg='#34495e',
            fg='white',
            activebackground='#2c3e50',
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=10,
            cursor='hand2',
            command=self.add_space
        )
        space_button.pack(side=tk.LEFT, padx=(0, 6))
        
        backspace_button = tk.Button(
            action_frame1,
            text="⌫ Delete",
            font=('Segoe UI', 11),
            bg='#e67e22',
            fg='white',
            activebackground='#d35400',
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=10,
            cursor='hand2',
            command=self.backspace
        )
        backspace_button.pack(side=tk.LEFT, padx=(0, 6))
        
        self.auto_speak_var = tk.BooleanVar(value=False)
        auto_speak_check = tk.Checkbutton(
            action_frame1,
            text="🔊 Auto",
            variable=self.auto_speak_var,
            font=('Segoe UI', 10, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            selectcolor='#ecf0f1',
            activebackground='#ffffff',
            cursor='hand2'
        )
        auto_speak_check.pack(side=tk.LEFT, padx=(8, 0))
        
        # Action buttons - Row 2
        action_frame2 = tk.Frame(right_frame, bg='#ffffff')
        action_frame2.pack(fill=tk.X, padx=20, pady=(0, 8))
        
        self.speak_button = tk.Button(
            action_frame2,
            text="🔊 SPEAK",
            font=('Segoe UI', 13, 'bold'),
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60',
            relief=tk.FLAT,
            bd=0,
            padx=25,
            pady=12,
            cursor='hand2',
            command=self.speak_sentence
        )
        self.speak_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        
        translate_button = tk.Button(
            action_frame2,
            text="🌐 Translate",
            font=('Segoe UI', 11),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=12,
            cursor='hand2',
            command=self.translate_sentence
        )
        translate_button.pack(side=tk.LEFT, padx=(0, 0))
        
        # Action buttons - Row 3
        action_frame3 = tk.Frame(right_frame, bg='#ffffff')
        action_frame3.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        copy_button = tk.Button(
            action_frame3,
            text="📋 Copy",
            font=('Segoe UI', 10),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=8,
            cursor='hand2',
            command=self.copy_to_clipboard
        )
        copy_button.pack(side=tk.LEFT, padx=(0, 6))
        
        save_button = tk.Button(
            action_frame3,
            text="💾 Save",
            font=('Segoe UI', 10),
            bg='#16a085',
            fg='white',
            activebackground='#138d75',
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=8,
            cursor='hand2',
            command=self.save_to_file
        )
        save_button.pack(side=tk.LEFT, padx=(0, 0))
        
        self.speech_status = tk.Label(
            action_frame3,
            text="",
            font=('Segoe UI', 9),
            bg='#ffffff',
            fg='#7f8c8d'
        )
        self.speech_status.pack(side=tk.LEFT, padx=(15, 0))
        
        # Info section
        info_frame = tk.Frame(right_frame, bg='#e8f5e9', relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        info_text = "⏳ Loading model...\nSupported classes will appear here."
        self.info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Segoe UI', 9),
            bg='#e8f5e9',
            fg='#2e7d32',
            justify=tk.LEFT
        )
        self.info_label.pack(padx=12, pady=12)
    
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
                            "Install googletrans for full multi-language support:\n"
                            "pip install googletrans==4.0.0rc1\n\n"
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
        info_text += "\n⚠️ Hold gesture steady for 2-3 seconds"
        
        self.info_label.config(text=info_text)
    
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
        
        # GPU memory cleanup using gpu_utils
        cleanup_gpu_memory()
    
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
                        # Increased threshold from 0.55 to 0.75 for better accuracy
                        result = self.recognition_model.predict_class(sequence_array, threshold=0.75)
                        
                        if result and isinstance(result, (tuple, list)) and len(result) >= 2:
                            predicted_sign, confidence_score = result[0], result[1]
                            
                            # More strict filtering: require higher confidence for specific problematic gestures
                            min_confidence = 0.75
                            if predicted_sign in ['V', 'K', 'N', 'G']:
                                # These gestures need even higher confidence due to model bias
                                min_confidence = 0.85
                            
                            if predicted_sign not in ["uncertain", "unknown"] and confidence_score >= min_confidence:
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
            
            # Auto-speak if enabled
            if self.auto_speak_var.get():
                self.speak_single_word(sign)
    
    def clear_placeholder(self, event=None):
        """Clear placeholder text when user focuses on the text widget"""
        if self.is_placeholder:
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.config(fg='#2c3e50')  # Normal text color
            self.is_placeholder = False
    
    def restore_placeholder(self, event=None):
        """Restore placeholder if text widget is empty"""
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if not current_text:
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", "Your message will appear here...\n\n✨ Tips:\n• Hold a sign steady to detect it\n• Click '➕ Add Sign' to add to message\n• Enable '🔊 Auto' for instant speech\n• Use '🔊 SPEAK TEXT' to read full message")
            self.sentence_text.config(fg='#95a5a6')  # Gray for placeholder
            self.is_placeholder = True
    
    def add_to_sentence(self, sign):
        # Clear placeholder if present
        if self.is_placeholder:
            self.clear_placeholder()
        
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
        if self.is_placeholder:
            self.clear_placeholder()
        
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert("1.0", current_text + " ")
    
    def backspace(self):
        """Remove last character"""
        if self.is_placeholder:
            return  # Don't modify placeholder
        
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if current_text:
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", current_text[:-1])
            
            # Restore placeholder if text becomes empty
            if len(current_text[:-1]) == 0:
                self.restore_placeholder()
    
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
        
        # Visual feedback
        self.speak_button.config(bg='#f39c12', text='🔊 SPEAKING...')
        self.speech_status.config(text="🗣️ Speaking...", fg='#f39c12')
        
        def speak_and_reset():
            self.safe_speak_text(text_to_speak)
            # Reset button after 2 seconds
            import time
            time.sleep(2)
            self.root.after(0, lambda: self.speak_button.config(bg='#2ecc71', text='🔊 SPEAK TEXT'))
            self.root.after(0, lambda: self.speech_status.config(text="✅ Done", fg='#27ae60'))
            self.root.after(2000, lambda: self.speech_status.config(text=""))
        
        import threading
        threading.Thread(target=speak_and_reset, daemon=True).start()
    
    def speak_single_word(self, word: str):
        """Speak a single word/sign (used for auto-speak)"""
        self.speech_status.config(text=f"🗣️ {word}", fg='#3498db')
        
        def speak_and_clear():
            self.safe_speak_text(word)
            import time
            time.sleep(1)
            self.root.after(0, lambda: self.speech_status.config(text=""))
        
        import threading
        threading.Thread(target=speak_and_clear, daemon=True).start()
    
    def copy_to_clipboard(self):
        """Copy message text to clipboard"""
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if current_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(current_text)
            self.speech_status.config(text="📋 Copied!", fg='#3498db')
            self.root.after(2000, lambda: self.speech_status.config(text=""))
        else:
            self.speech_status.config(text="⚠️ No text to copy", fg='#e74c3c')
            self.root.after(2000, lambda: self.speech_status.config(text=""))
    
    def save_to_file(self):
        """Save message text to file"""
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if not current_text:
            self.speech_status.config(text="⚠️ No text to save", fg='#e74c3c')
            self.root.after(2000, lambda: self.speech_status.config(text=""))
            return
        
        try:
            from tkinter import filedialog
            from datetime import datetime
            
            default_name = f"ISL_Message_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                initialfile=default_name,
                filetypes=[
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(current_text)
                self.speech_status.config(text="💾 Saved!", fg='#16a085')
                self.root.after(2000, lambda: self.speech_status.config(text=""))
                logger.info(f"Message saved to {filepath}")
        except Exception as e:
            logger.error(f"Save error: {e}")
            self.speech_status.config(text="❌ Save failed", fg='#e74c3c')
            self.root.after(2000, lambda: self.speech_status.config(text=""))
    
    def translate_sentence(self):
        if self.is_placeholder:
            return  # Don't translate placeholder
        
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if current_text and self.translator and hasattr(self.translator, 'translate_sentence'):
            translated = self.translator.translate_sentence(current_text, self.selected_language)
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", translated)
            self.speech_status.config(text=f"🌐 Translated to {self.selected_language.upper()}", fg='#9b59b6')
            self.root.after(2000, lambda: self.speech_status.config(text=""))
    
    def clear_sentence(self):
        self.sentence_text.delete("1.0", tk.END)
        self.restore_placeholder()  # Show placeholder after clearing
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
            
            # GPU memory cleanup using gpu_utils
            cleanup_gpu_memory()
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
