# 🤟 ISL Bridge - Indian Sign Language Translator

**An Engineering Clinics Project**

A standalone desktop application for real-time Indian Sign Language (ISL) recognition using PyTorch, MediaPipe Holistic, and tkinter. This project aims to bridge the communication gap for the Deaf and Hard of Hearing (DHH) community in India.

## 🎯 Project Overview

ISL Bridge provides a technological solution to help DHH individuals communicate effectively with the hearing community by translating ISL gestures in real-time using computer vision and deep learning.

## ✨ What It Does

* **Real-time Recognition:** Translates ISL gestures from your webcam into text.
* **Dynamic Gesture Support:** Automatically recognizes any gestures you add to the dataset - no code changes needed!
* **Currently Trained:** 36 gestures (A-Z alphabet + 0-9 numbers)
* **Expandable:** Add folders for new gestures (HELLO, THANK_YOU, etc.) and retrain
* **Lightweight AI:** Uses an efficient PyTorch-based LSTM model with only ~340KB size.
* **Computer Vision:** Powered by MediaPipe Holistic for landmark extraction (166 features: hands, pose, and face).
* **Desktop App:** A simple and clean GUI built with tkinter.
* **Multi-Language Support:** Translates to 8 Indian languages (English, Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi) with offline Text-to-Speech using pyttsx3.
* **Optimized Performance:** Frame skipping reduces lag for smooth real-time recognition.

## 🌟 Key Features

- **Webcam-Based Recognition**: Real-time gesture capture and analysis
- **MediaPipe Holistic Integration**: Accurate hand, pose, and face landmark detection
- **LSTM Deep Learning Model**: Sequential gesture pattern recognition
- **Offline Operation**: Works without internet (except translation API)
- **User-Friendly Interface**: Intuitive tkinter-based desktop GUI
- **Cross-Platform**: Windows, macOS, and Linux compatible

## 🎯 Project Scope & Impact

### Problem Statement
The Deaf and Hard of Hearing (DHH) community in India faces significant communication barriers when interacting with the hearing population. Traditional methods like interpreters are not always available, creating challenges in education, healthcare, and daily communication.

### Our Solution
ISL Bridge provides a technological bridge by:
- **Real-time Translation**: Converts ISL gestures to text instantly
- **Multi-Language Output**: Supports multiple Indian languages for broader accessibility
- **Offline Capability**: Works without constant internet (TTS is fully offline)
- **Lightweight Design**: Runs on standard laptops/desktops without special hardware
- **User-Friendly**: Simple interface designed for all age groups

### Target Audience
- DHH individuals using Indian Sign Language
- Educational institutions
- Healthcare facilities
- Government service centers
- Family members learning to communicate with DHH relatives

---

## 🚀 Quick Start (3 Steps!)

### **Step 1: Setup**

1.  Clone this repository.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
3.  Activate it (Windows):
    ```bash
    .venv\Scripts\activate
    ```
4.  Install all required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### **Step 2: Get a Dataset**

ISL Bridge supports **both images and videos** for training:

1.  Create the following folder structure:
    ```
    data/raw/Frames_Word_Level/
    ├── A/
    │   ├── A1.jpg
    │   ├── A2.jpg
    │   └── A_video1.mp4
    ├── B/
    │   ├── B1.jpg
    │   └── B_video.mp4
    └── HELLO/
        ├── HELLO_1.jpg
        ├── HELLO_2.png
        └── HELLO_video.mp4
    ```

2.  **Supported formats:**
    - **Images:** `.jpg`, `.png` - For static gestures
    - **Videos:** `.mp4`, `.avi`, `.mov` - For dynamic gestures (system auto-extracts ~30 frames)

3.  Populate these folders with your data. The system automatically detects any folders you add - no need to edit `config.py`!

**Note:** Videos are automatically processed - the system extracts approximately 30 frames uniformly from each video file.

### **Step 3: Train & Run**

1.  **Train the model:**
    ```bash
    python train_model.py
    ```
    This will process images and videos, extract landmarks, train the LSTM, and save the final model to `models/isl_trained_model.pth`.

2.  **Run the application:**
    ```bash
    python app.py
    ```
    ...or double-click the `run_isl_bridge.bat` file.

---

## � Data Format & Training

### Supported Data Types

ISL Bridge training script automatically handles:

✅ **Images** (`.jpg`, `.png`):
- Best for: Static gesture poses
- Processing: Direct landmark extraction from each image
- Use case: Alphabet letters, numbers, static words

✅ **Videos** (`.mp4`, `.avi`, `.mov`):
- Best for: Dynamic gestures, continuous movements
- Processing: Extracts ~30 frames uniformly from each video
- Use case: Action words, phrases, sentences

### Training Process

The LSTM model is trained on **sequences** of landmarks:
1. **From Images:** Creates 30-frame sequences with synthetic noise for data augmentation
2. **From Videos:** Extracts real 30-frame sequences from video files
3. **Mixed Data:** Can train on both images and videos simultaneously!

**Recommendation:** For best real-world performance, include video data in your training set, as it better represents the dynamic nature of sign language.

---

## 📖 How to Use the App

**Note:** This app now includes performance optimizations with frame skipping and manual sign addition control.

1.  **Start Camera:** Click the "🎥 Start Camera" button to turn on your webcam.
2.  **Show a Sign:** Hold your hand clearly in view. The "Current Sign" box will update as you sign (predictions run every 5 frames for smooth performance).
3.  **➕ Add Sign:** When the correct sign is shown, click the green **"➕ Add Sign"** button. This adds the recognized word to your sentence and automatically adds a space.
4.  **Space:** Click the **"Space"** button if you need to add an extra space manually.
5.  **⌫ Backspace:** Click to remove the last character from your message.
6.  **🗑️ Clear:** Click to clear the entire message.
7.  **Language Selection:** Choose your target language from the dropdown (Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi, or English).
8.  **🔊 Speak:** Click to hear the translation spoken aloud.

**Workflow Example:**
- Show sign "H" → Click "Add Sign" → Show "E" → Click "Add Sign" → Show "L" → Click "Add Sign" → Show "L" → Click "Add Sign" → Show "O" → Click "Add Sign"
- Result: "H E L L O " in your message box

---

## 📁 Project Files

### Core Application Files
* `app.py`: Main tkinter desktop application with optimized frame processing
* `model.py`: PyTorch LSTM model architecture (GestureRecognitionLSTM class)
* `landmark_extractor.py`: MediaPipe-based landmark extraction (166 features)
* `config.py`: Central configuration (dynamic class detection, model settings, paths)

### Supporting Modules
* `enhanced_translation.py`: Multi-language translation with thread-safe TTS
* `train_model.py`: Training script ⚠️ **(Uses static images - see accuracy note)**
* `preprocess_data.py`: Data preprocessing utilities
* `error_handling.py`: Error management system

### Utilities & Launchers
* `run_isl_bridge.bat`: Windows launcher (double-click to run)
* `requirements.txt`: All Python dependencies
* `analyze_dataset.py`: Dataset analysis helper script

### Data Directories
* `data/raw/`: Place your training images and videos here
* `data/processed/`: Auto-generated processed features
* `models/`: Trained model saved here (`isl_trained_model.pth`)
* `logs/`: Training logs and plots

---

## 🆘 Troubleshooting

* **"Camera shows a static image / tutorial screen"**
    * Your computer has a virtual camera (like BYOM). The app automatically tries camera indices 0, 1, and 2. It should find your real webcam. If it doesn't, check the terminal output to see which camera opened successfully.

* **"No model found" Error**
    * You must run `python train_model.py` first to create the `isl_trained_model.pth` file.

* **"Predictions are inaccurate"**
    * **For Image-only training:** Static images are less effective than videos. Consider adding video data to your dataset for better accuracy.
    * **Solution:** Add `.mp4`/`.avi`/`.mov` files to your gesture folders and retrain. Videos capture the dynamic nature of sign language better than static images.
    * **Recommended:** Mix of 20-50% videos with images for optimal performance.

* **"Camera feed is laggy"**
    * The app uses frame skipping (predicts every 5 frames) to reduce lag. You can adjust `self.prediction_interval` in `app.py` (line 73). Lower = more responsive but slower, Higher = faster but less responsive.

* **"App is slow or freezing"**
    * Close other camera applications
    * Ensure good lighting (less processing needed)
    * Check if your CPU is overloaded

* **"Signs are being added automatically!"**
    * This has been fixed! You now need to manually click the "➕ Add Sign" button to add recognized signs to your sentence.

---

## 📊 Technical Details

### Model Architecture
* **Type:** 2-Layer LSTM (Long Short-Term Memory) neural network
* **Framework:** PyTorch
* **Size:** ~340KB (lightweight and portable)
* **Input:** Sequential landmark data (30 frames @ 166 features per frame)
* **Output:** Dynamic gesture classes (currently 36: A-Z + 0-9) with confidence scores

### Feature Extraction (MediaPipe Holistic)
* **Total Features:** 166 landmarks per frame
  - **Left hand:** 21 landmarks (x, y, z) = 63 features
  - **Right hand:** 21 landmarks (x, y, z) = 63 features  
  - **Pose:** 8 keypoints (x, y, z) = 24 features
  - **Face:** 16 keypoints (x, y) = 16 features
  - **Total:** 166 features per frame

### Recognition System
* **Sequence Length:** 30 frames (~1 second at 30 FPS)
* **Current Gesture Classes:** 36 (automatically detected from dataset)
  - **Alphabets:** A-Z (26 letters)
  - **Numbers:** 0-9 (10 digits)
  - **Words:** Expandable - add any word by creating a folder in the dataset!
* **Performance:** Predicts every 5 frames (80% CPU reduction)
* **Confidence Threshold:** 55% minimum for recognition (optimized for real-time webcam use)

### 🎯 Adding New Gestures (No Code Changes Required!)

Want to add HELLO, THANK_YOU, or any custom gesture?

1. **Create a folder** in `data/raw/Frames_Word_Level/YOUR_GESTURE/`
2. **Add data** (20-50 samples recommended):
   - Images: `.jpg`, `.png` files
   - Videos: `.mp4`, `.avi`, `.mov` files
   - Or mix both!
3. **Retrain**: `python train_model.py`
4. **Run app**: `python app.py`

The system automatically detects and learns any new gestures you add!

**Pro Tip:** For dynamic gestures (actions, phrases), use videos for better accuracy!



### Software Stack
- **Programming Language:** Python 3.8+
- **Deep Learning:** PyTorch 2.0+
- **Computer Vision:** OpenCV, MediaPipe Holistic
- **GUI Framework:** tkinter (built-in)
- **Translation:** Google Translate API (googletrans)
- **Text-to-Speech:** pyttsx3 (offline)
- **Data Processing:** NumPy, scikit-learn

### Language Support
**Primary Languages (as per project scope):**
- English (en)
- Hindi (hi)  
- Tamil (ta)

**Extended Support (implementation):**
- Telugu (te)
- Bengali (bn)
- Gujarati (gu)
- Marathi (mr)
- Punjabi (pa)

## ⚡ Performance Optimizations

The app includes several optimizations for smooth real-time operation:

1. **Frame Skipping:** AI predictions run every 5th frame instead of every frame, reducing CPU load by 80%
2. **Thread-Safe TTS:** Text-to-speech uses mutex locks to prevent concurrent execution errors
3. **Manual Sign Addition:** Users control when to add signs, preventing unwanted auto-additions
4. **BGR→RGB Conversion:** Optimized to convert once per frame, not multiple times
5. **Multi-Camera Detection:** Automatically tries camera indices 0, 1, 2 to find real webcam

## 🙏 Credits

### Development Team
* **Ishika Sehgal** - Team Leader, UI Developer
* **Krish Nagpal** - Team Member, Developer, Project Lead 
* **Jiya Choudhary** - Team Member, Documentation Specialist
* **Isha Patial** - Team Member, Dataset Curator
* **Project:** Engineering Clinics Course Project

### Technologies & Frameworks
* **MediaPipe** by Google - Holistic landmark detection framework
* **PyTorch** - Deep learning framework for LSTM model
* **OpenCV** - Computer vision and camera processing library
* **Google Translate API** - Multi-language translation support
* **pyttsx3** - Offline text-to-speech functionality
* **ISL Community** - Gesture datasets and domain knowledge

### Special Thanks
* Engineering Clinics faculty and instructors
* Indian Sign Language research community
* Open-source contributors

---

## 📝 License

This project is licensed under the **MIT License** - feel free to use and modify!

---

## 🎓 Project Information

**Course:** Engineering Clinics  
**Project:** ISL Bridge - Indian Sign Language Translator  
**Timeline:** October 2025  
**Objective:** Bridge communication gap for DHH community in India  
**Team:** Krish Nagpal, Jiya Choudhary, Ishika Sehgal, Isha Patial

**ISL Bridge** - Making sign language communication accessible to everyone! 🤟

**Repository:** [ISLBridge-Project](https://github.com/Krish-codex/ISLBridge-Project)

---

**⭐ Star this repository if you find it helpful!**