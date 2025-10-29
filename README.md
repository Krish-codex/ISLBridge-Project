# 🤟 ISL Bridge - Indian Sign Language Translator

**An Engineering Clinics Project**

A standalone desktop application for real-time Indian Sign Language (ISL) recognition using PyTorch, MediaPipe Holistic, and tkinter. This project aims to bridge the communication gap for the Deaf and Hard of Hearing (DHH) community in India.

## 🎯 Project Overview

ISL Bridge provides a technological solution to help DHH individuals communicate effectively with the hearing community by translating ISL gestures in real-time using computer vision and deep learning.

## ✨ What It Does

* **Real-time Recognition:** Translates ISL gestures from your webcam into text.
* **48 Gestures:** Recognizes the complete alphabet (A-Z), numbers (0-9), and 12 essential words (HELLO, THANK_YOU, PLEASE, SORRY, YES, NO, HELP, WATER, FOOD, BATHROOM, STOP, GO).
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

The current `train_model.py` script is designed to work with **static images** (one picture per sign).

1.  Create the following folder structure:
    ```
    data/raw/Frames_Word_Level/
    ├── A/
    │   ├── A1.jpg
    │   └── A2.jpg
    ├── B/
    │   ├── B1.jpg
    │   └── B2.jpg
    └── HELLO/
        ├── HELLO_1.jpg
        └── ...
    ```
2.  Populate these folders with images for all 48 classes listed in `config.py`.

### **Step 3: Train & Run**

1.  **Train the model:**
    ```bash
    python train_model.py
    ```
    This will process the images, train the LSTM, and save the final model to `models/isl_trained_model.pth`.

2.  **Run the application:**
    ```bash
    python app.py
    ```
    ...or double-click the `run_isl_bridge.bat` file.

---

## 🚨 Important: How to Improve Accuracy (Critical Fix)

You will likely notice that the model's accuracy is low. This is because there is a **mismatch between your training data and your real-world use.**

* **Training (`train_model.py`):** You are training an LSTM (a sequence model) on *static images*. The script copies the landmarks from one image 30 times and adds "noise." The model learns to recognize this *one noisy pose*.
* **Running (`app.py`):** You are feeding the model a *real, dynamic sequence* of 30 different frames from your webcam. The model has never seen this type of real, moving data.

**How to Fix This:**
You must train the model on data that looks like the data it will see in real life.
1.  **Get a Video Dataset:** Download a video-based ISL dataset, like the **INCLUDE dataset**.
2.  **Modify `train_model.py`:** You must rewrite this script to:
    * Load video files instead of images.
    * Extract landmarks from *every frame* of each video.
    * Create your training sequences (X, y) from these *real* video frames.
3.  **Retrain your model.** An LSTM trained on real video sequences will be *significantly* more accurate than one trained on static images.

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
* `config.py`: Central configuration (48 classes, model settings, paths)

### Supporting Modules
* `enhanced_translation.py`: Multi-language translation with thread-safe TTS
* `train_model.py`: Training script ⚠️ **(Uses static images - see accuracy note)**
* `preprocess_data.py`: Data preprocessing utilities
* `error_handling.py`: Error management system

### Utilities & Launchers
* `run_isl_bridge.bat`: Windows launcher (double-click to run)
* `requirements.txt`: All Python dependencies
* `dataset_analyzer.py`: Dataset analysis helper script

### Data Directories
* `data/raw/`: Place your training images here
* `data/processed/`: Auto-generated processed features
* `models/`: Trained model saved here (`isl_trained_model.pth`)
* `logs/`: Training logs and plots

---

## 🆘 Troubleshooting

* **"Camera shows a static image / tutorial screen"**
    * Your computer has a virtual camera (like BYOM). The app automatically tries camera indices 0, 1, and 2. It should find your real webcam. If it doesn't, check the terminal output to see which camera opened successfully.

* **"No model found" Error**
    * You must run `python train_model.py` first to create the `isl_trained_model.pth` file.

* **"Predictions are very inaccurate"**
    * This is expected. See the "🚨 Important: How to Improve Accuracy" section above. Your training method (static images) does not match your runtime use (dynamic video).

* **"Camera feed is laggy"**
    * The app uses frame skipping (predicts every 5 frames) to reduce lag. You can adjust `self.prediction_interval` in `app.py` (line 69). Lower = more responsive but slower, Higher = faster but less responsive.

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
* **Output:** 48 gesture classes with confidence scores

### Feature Extraction (MediaPipe Holistic)
* **Total Features:** 166 landmarks per frame
  - **Left hand:** 21 landmarks (x, y, z) = 63 features
  - **Right hand:** 21 landmarks (x, y, z) = 63 features  
  - **Pose:** 8 keypoints (x, y, z) = 24 features
  - **Face:** 16 keypoints (x, y) = 16 features
  - **Total:** 166 features per frame

### Recognition System
* **Sequence Length:** 30 frames (~1 second at 30 FPS)
* **Gesture Classes:** 48 total
  - **Alphabets:** A-Z (26 letters)
  - **Numbers:** 0-9 (10 digits)
  - **Words:** HELLO, THANK_YOU, PLEASE, SORRY, YES, NO, HELP, WATER, FOOD, BATHROOM, STOP, GO (12 words)
* **Performance:** Predicts every 5 frames (80% CPU reduction)
* **Confidence Threshold:** 70% minimum for recognition

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
