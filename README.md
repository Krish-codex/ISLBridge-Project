# 🤟 ISL Bridge - Indian Sign Language Translator# 🤟 ISL Bridge - Indian Sign Language Translator



**Real-time ISL recognition using PyTorch, MediaPipe Holistic, and tkinter****An Engineering Clinics Project**



A desktop application that translates Indian Sign Language gestures into text and speech, helping bridge communication between the DHH community and hearing individuals.A standalone desktop application for real-time Indian Sign Language (ISL) recognition using PyTorch, MediaPipe Holistic, and tkinter. This project aims to bridge the communication gap for the Deaf and Hard of Hearing (DHH) community in India.



---## 🎯 Project Overview



## ✨ FeaturesISL Bridge provides a technological solution to help DHH individuals communicate effectively with the hearing community by translating ISL gestures in real-time using computer vision and deep learning.



- **Real-time Recognition:** Webcam-based gesture translation with MediaPipe Holistic## ✨ What It Does

- **Expandable Dataset:** Add new gestures by creating folders - no code changes needed

- **Multi-Language Support:** 8 Indian languages (English, Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi)* **Real-time Recognition:** Translates ISL gestures from your webcam into text.

- **Hybrid Text-to-Speech:** gTTS (online) with pyttsx3 (offline) fallback* **Dynamic Gesture Support:** Automatically recognizes any gestures you add to the dataset - no code changes needed!

- **GPU Acceleration:** 6-10x faster training with NVIDIA GPU* **Currently Trained:** 36 gestures (A-Z alphabet + 0-9 numbers)

- **Lightweight:** ~340KB LSTM model with 166 landmark features* **Expandable:** Add folders for new gestures (HELLO, THANK_YOU, etc.) and retrain

- **Offline-First:** Works without internet (except translation and online TTS)* **Lightweight AI:** Uses an efficient PyTorch-based LSTM model with only ~340KB size.

* **Computer Vision:** Powered by MediaPipe Holistic for landmark extraction (166 features: hands, pose, and face).

---* **Desktop App:** A simple and clean GUI built with tkinter.

* **Multi-Language Support:** Translates to 8 Indian languages (English, Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi) with hybrid Text-to-Speech using gTTS (online, natural voice) and pyttsx3 (offline fallback).

## 🚀 Quick Start* **Optimized Performance:** Frame skipping reduces lag for smooth real-time recognition.



### 1. Setup Environment## 🌟 Key Features



```bash- **Webcam-Based Recognition**: Real-time gesture capture and analysis

# Clone repository- **MediaPipe Holistic Integration**: Accurate hand, pose, and face landmark detection

git clone <your-repo-url>- **LSTM Deep Learning Model**: Sequential gesture pattern recognition

cd ISLBridge-Project- **Offline Operation**: Works without internet (except translation API and optional online TTS)

- **User-Friendly Interface**: Intuitive tkinter-based desktop GUI

# Create virtual environment- **Cross-Platform**: Windows, macOS, and Linux compatible

python -m venv .venv

## 🎯 Project Scope & Impact

# Activate (Windows)

.venv\Scripts\activate### Problem Statement

The Deaf and Hard of Hearing (DHH) community in India faces significant communication barriers when interacting with the hearing population. Traditional methods like interpreters are not always available, creating challenges in education, healthcare, and daily communication.

# Install dependencies

pip install -r requirements.txt### Our Solution

```ISL Bridge provides a technological bridge by:

- **Real-time Translation**: Converts ISL gestures to text instantly

### 2. Prepare Dataset- **Multi-Language Output**: Supports multiple Indian languages for broader accessibility

- **Offline Capability**: Works without constant internet (TTS supports both online and offline modes)

Create this folder structure:- **Lightweight Design**: Runs on standard laptops/desktops without special hardware

- **User-Friendly**: Simple interface designed for all age groups

```

data/raw/Frames_Word_Level/### Target Audience

├── A/- DHH individuals using Indian Sign Language

│   ├── A1.jpg- Educational institutions

│   └── A_video.mp4- Healthcare facilities

├── B/- Government service centers

│   ├── B1.jpg- Family members learning to communicate with DHH relatives

│   └── B_video.mp4

├── HELLO/---

    └── HELLO_1.jpg

```## 🚀 Quick Start (3 Steps!)



**Supported Formats:**### **Step 1: Setup**

- Images: `.jpg`, `.png`

- Videos: `.mp4`, `.avi`, `.mov` (auto-extracts ~30 frames)1.  Clone this repository.

2.  Create a virtual environment:

The system automatically detects any folders you add!    ```bash

    python -m venv .venv

### 3. Train & Run    ```

3.  Activate it (Windows):

```bash    ```bash

# Train model (auto-uses GPU if available)    .venv\Scripts\activate

python train_model.py    ```

4.  Install all required libraries:

# Run application    ```bash

python app.py    pip install -r requirements.txt

# OR double-click: run_isl_bridge.bat    ```

```

### **Step 2: Get a Dataset**

---

ISL Bridge supports **both images and videos** for training:

## ⚡ GPU Acceleration (Optional)

1.  Create the following folder structure:

### Quick GPU Setup    ```

    data/raw/Frames_Word_Level/

```powershell    ├── A/

# 1. Check GPU    │   ├── A1.jpg

nvidia-smi    │   ├── A2.jpg

    │   └── A_video1.mp4

# 2. Install PyTorch with CUDA    ├── B/

pip uninstall torch torchvision torchaudio -y    │   ├── B1.jpg

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121    │   └── B_video.mp4

    └── HELLO/

# 3. Verify        ├── HELLO_1.jpg

python verify_gpu.py        ├── HELLO_2.png

```        └── HELLO_video.mp4

    ```

### Performance

2.  **Supported formats:**

| Operation | CPU | GPU (RTX 3050) | Speedup |    - **Images:** `.jpg`, `.png` - For static gestures

|-----------|-----|----------------|---------|    - **Videos:** `.mp4`, `.avi`, `.mov` - For dynamic gestures (system auto-extracts ~30 frames)

| Training | 45-60 min | 5-10 min | **6-10x** |

| Inference | 50-80 ms | 8-15 ms | **5-7x** |3.  Populate these folders with your data. The system automatically detects any folders you add - no need to edit `config.py`!



**Note:** GPU is auto-detected - no code changes needed!**Note:** Videos are automatically processed - the system extracts approximately 30 frames uniformly from each video file.



---### **Step 3: Train & Run**



## 📁 Project Structure1.  **Train the model:**

    ```bash

```    python train_model.py

ISLBridge-Project/    ```

├── app.py                      # Main desktop application    This will process images and videos, extract landmarks, train the LSTM, and save the final model to `models/isl_trained_model.pth`.

├── train_model.py              # Model training script    

├── model.py                    # LSTM model definition    **GPU Acceleration (RECOMMENDED):**

├── landmark_extractor.py       # MediaPipe landmark extraction    - **6-10x faster training** with NVIDIA GPU

├── enhanced_translation.py     # Multi-language translation & TTS    - See [GPU Setup Guide](#-gpu-acceleration-setup) below

├── config.py                   # Configuration settings    - Your system: `RTX 3050 6GB` detected ✅

├── gpu_utils.py                # GPU optimization utilities

├── verify_gpu.py               # GPU testing tool2.  **Run the application:**

├── requirements.txt            # Python dependencies    ```bash

├── run_isl_bridge.bat          # Quick launch script    python app.py

├── data/    ```

│   └── raw/    ...or double-click the `run_isl_bridge.bat` file.

│       ├── processed_dataset.json

│       └── Frames_Word_Level/  # Your gesture folders---

├── models/

│   └── isl_trained_model.pth   # Trained model## ⚡ GPU Acceleration Setup

└── logs/

    ├── training_plots/### Why Use GPU?

    └── evaluation_results.json- **Training:** 6-10x faster (5-10 min instead of 45-60 min)

```- **Inference:** 5-7x faster real-time recognition

- **Recommended:** Any NVIDIA GPU with 4GB+ VRAM

---

### Your System

## 🎮 GUI Features✅ **GPU Detected:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  

✅ **Drivers:** v581.32 with CUDA 13.0  

- **Camera Controls:** Start/Stop camera with live feed

- **Sign Recognition:** Real-time gesture detection with confidence display### Quick Setup (3 commands)

- **Message Builder:** Add signs, spaces, delete characters

- **Auto-Speak:** Instant speech output (toggle on/off)```powershell

- **Translation:** Convert to selected Indian language# 1. Check your GPU

- **Export:** Copy to clipboard or save to filenvidia-smi



---# 2. Install PyTorch with CUDA support

pip uninstall torch torchvision torchaudio -y

## 🔧 Configurationpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121



Edit `config.py` to customize:# 3. Verify GPU is working

python verify_gpu.py

```python```

MODEL_CONFIG = {

    "input_size": 166,       # MediaPipe landmarks### Expected Output

    "hidden_size": 128,      # LSTM hidden units```

    "num_layers": 2,         # LSTM layers✓ CUDA Available: True

    "sequence_length": 30,   # Frames per gesture✓ GPU Name: NVIDIA GeForce RTX 3050 6GB Laptop GPU

}✓ GPU Memory: 6.00 GB

🚀 GPU Speedup: 8-15x faster!

GPU_CONFIG = {```

    "enable_gpu": True,              # Auto-use GPU

    "enable_mixed_precision": True,  # FP16 for 2x speedup### Performance Comparison

}

```| Operation | CPU | GPU (RTX 3050) | Speedup |

|-----------|-----|----------------|---------|

---| Training (100 epochs) | 45-60 min | 5-10 min | **6-10x** ⚡ |

| Per-epoch | 30-40 sec | 3-5 sec | **8-10x** ⚡ |

## 📊 Model Details| Inference | 50-80 ms | 8-15 ms | **5-7x** ⚡ |



- **Architecture:** 2-layer LSTM (166 → 128 → 128 → Classes)**Note:** Your code automatically detects and uses GPU if available - no code changes needed!

- **Input:** 166 MediaPipe landmarks (21 × 2 hands + 33 pose + 468 face)

- **Training:** Class-balanced dataset (1500 samples/class), data augmentation### GPU Features Included

- **Optimization:** Adam optimizer, learning rate: 0.0005, dropout: 0.3

✅ **Automatic Device Selection**

---- Code automatically detects GPU/CPU

- No manual configuration needed

## 🛠️ Troubleshooting- Seamless fallback to CPU if no GPU



### Camera Issues✅ **Mixed Precision Training (FP16)**

```bash- 2x additional speedup on GPU

# Test camera- Reduces memory usage

python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"- Maintains model accuracy

```

✅ **Smart Memory Management**

### Model Not Loading- Automatic GPU memory cleanup

```bash- Prevents memory leaks

# Check if model exists- Optimized batch sizes

dir models\isl_trained_model.pth

✅ **Performance Monitoring**

# Retrain if needed- Real-time GPU memory tracking

python train_model.py- Training speed benchmarks

```- Detailed device information



### GPU Not Detected### GPU Configuration

```bash

# Verify CUDA installationAll GPU settings are in `config.py` under `GPU_CONFIG`:

python verify_gpu.py

```python

# Reinstall PyTorch with CUDAGPU_CONFIG = {

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121    "enable_gpu": True,              # Auto-use GPU if available

```    "enable_mixed_precision": True,  # FP16 for 2x speedup

    "enable_cudnn_benchmark": True,  # Optimize algorithms

### Low Accuracy    "gpu_memory_fraction": 0.8,      # Use max 80% GPU memory

- Ensure good lighting and hand visibility    "num_workers": 4,                # DataLoader workers

- Hold gestures steady for 1-2 seconds    "pin_memory": True,              # Fast GPU transfer

- Add more training samples per gesture (500-1500 recommended)}

- Check confidence score (aim for >70%)```



---**Adjusting for your GPU:**

- If you get "Out of Memory" errors, reduce `batch_size` to 16 or 8

## 📦 Dependencies- If training is slow, check `nvidia-smi` for GPU usage

- For 4GB GPUs, set `gpu_memory_fraction: 0.6`

Core libraries:

- **PyTorch** (2.0+): Deep learning frameworkFor detailed setup and troubleshooting, see: [`GPU_SETUP_GUIDE.md`](GPU_SETUP_GUIDE.md)

- **MediaPipe** (0.10+): Landmark extraction

- **OpenCV** (4.8+): Video capture---

- **NumPy**: Numerical operations

- **tkinter**: GUI (pre-installed with Python)## � Data Format & Training

- **gTTS/pyttsx3**: Text-to-speech

- **googletrans**: Translation### Supported Data Types



For GPU:ISL Bridge training script automatically handles:

- **CUDA Toolkit** (12.1+)

- **cuDNN** (8.9+)✅ **Images** (`.jpg`, `.png`):

- **NVIDIA Drivers** (latest)- Best for: Static gesture poses

- Processing: Direct landmark extraction from each image

Install all: `pip install -r requirements.txt`- Use case: Alphabet letters, numbers, static words



---✅ **Videos** (`.mp4`, `.avi`, `.mov`):

- Best for: Dynamic gestures, continuous movements

## 📝 Adding New Gestures- Processing: Extracts ~30 frames uniformly from each video

- Use case: Action words, phrases, sentences

1. Create folder: `data/raw/Frames_Word_Level/NEW_GESTURE/`

2. Add images/videos to folder### Training Process

3. Retrain: `python train_model.py`

4. Done! The gesture is now recognized automaticallyThe LSTM model is trained on **sequences** of landmarks:

1. **From Images:** Creates 30-frame sequences with synthetic noise for data augmentation

---2. **From Videos:** Extracts real 30-frame sequences from video files

3. **Mixed Data:** Can train on both images and videos simultaneously!

## 🔍 Advanced Usage

**Recommendation:** For best real-world performance, include video data in your training set, as it better represents the dynamic nature of sign language.

### Analyze Dataset

```bash---

python analyze_dataset.py

# Shows: classes, samples per class, statistics## 📖 How to Use the App

```

**Note:** This app now includes performance optimizations with frame skipping and manual sign addition control.

### Custom Training Parameters

Edit `config.py` under `MODEL_CONFIG`:1.  **Start Camera:** Click the "🎥 Start Camera" button to turn on your webcam.

- Adjust `sequence_length` for longer/shorter gestures2.  **Show a Sign:** Hold your hand clearly in view. The "Current Sign" box will update as you sign (predictions run every 5 frames for smooth performance).

- Increase `hidden_size` for more complex patterns3.  **➕ Add Sign:** When the correct sign is shown, click the green **"➕ Add Sign"** button. This adds the recognized word to your sentence and automatically adds a space.

- Modify `learning_rate` for training stability4.  **Space:** Click the **"Space"** button if you need to add an extra space manually.

5.  **⌫ Backspace:** Click to remove the last character from your message.

---6.  **🗑️ Clear:** Click to clear the entire message.

7.  **Language Selection:** Choose your target language from the dropdown (Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi, or English).

## 📄 License8.  **🔊 Speak:** Click to hear the translation spoken aloud.



This project is an Engineering Clinics initiative aimed at improving accessibility for the Deaf and Hard of Hearing community in India.**Workflow Example:**

- Show sign "H" → Click "Add Sign" → Show "E" → Click "Add Sign" → Show "L" → Click "Add Sign" → Show "L" → Click "Add Sign" → Show "O" → Click "Add Sign"

---- Result: "H E L L O " in your message box



## 🤝 Contributing---



Contributions welcome! Focus areas:## 📁 Project Files

- Adding more ISL gestures to dataset

- Improving recognition accuracy### Core Application Files

- Optimizing performance* **`app.py`** (39.4 KB) - Main tkinter desktop application

- UI/UX enhancements  - Real-time webcam capture and processing

- Multi-platform testing  - Gesture recognition with frame skipping optimization

  - Multi-language translation interface

---  - GPU memory management

  

## 🎯 Project Goals* **`model.py`** (13.4 KB) - PyTorch LSTM model

  - GestureRecognitionLSTM architecture (2-layer LSTM)

**Primary:** Bridge communication gap for DHH community in India    - Automatic GPU/CPU detection

**Technical:** Real-time, accurate ISL recognition with minimal latency    - Mixed precision inference support

**Social Impact:** Accessible technology for education, healthcare, and daily communication  - Class weight balancing for better accuracy



---* **`landmark_extractor.py`** (8.2 KB) - MediaPipe integration

  - Extracts 166 features per frame (hands, pose, face)

## 📞 Support  - Optimized for real-time processing

  - Handles missing landmarks gracefully

For issues or questions:

1. Check troubleshooting section above* **`config.py`** (3.8 KB) - Central configuration

2. Review logs in `logs/` directory  - Dynamic gesture class detection

3. Test GPU with `verify_gpu.py`  - Model hyperparameters

4. Verify camera with OpenCV test command  - GPU configuration settings

  - Translation and TTS settings

---  - All paths and constants



**Made with ❤️ for the DHH Community**### Training & Data Processing

* **`train_model.py`** (22.5 KB) - Unified training script
  - **GPU-optimized training** (6-10x faster with NVIDIA GPU)
  - Processes both images and videos
  - Class balancing (max 1500 samples/class)
  - Data augmentation with noise
  - Automatic checkpoint saving
  - Generates training plots and confusion matrix
  - Mixed precision training (FP16) support

### GPU & Performance Utilities
* **`gpu_utils.py`** (7.0 KB) - GPU management utilities
  - Automatic device detection (GPU/CPU)
  - GPU memory monitoring and cleanup
  - Mixed precision support checking
  - Optimal batch size calculation
  - Performance benchmarking tools
  - cuDNN and TF32 optimization

* **`verify_gpu.py`** (3.3 KB) - GPU verification tool
  - Tests CUDA availability
  - Benchmarks GPU vs CPU performance
  - Verifies mixed precision support
  - Displays GPU properties and memory

### Translation & Audio
* **`enhanced_translation.py`** (20.8 KB) - Multi-language support
  - Hybrid TTS system (gTTS + pyttsx3)
  - Thread-safe audio playback
  - 8 Indian languages supported
  - Automatic online/offline fallback

### Analysis & Tools
* **`analyze_dataset.py`** (4.4 KB) - Dataset analysis
  - Shows gesture class distribution
  - Counts samples per class
  - Identifies low-data classes
  - Visualizes dataset statistics

### Launchers & Config
* **`run_isl_bridge.bat`** - Windows launcher (double-click to run)
* **`requirements.txt`** - Python dependencies with CUDA instructions
* **`README.md`** - This comprehensive guide
* **`GPU_SETUP_GUIDE.md`** - Detailed GPU setup instructions
* **`UPGRADE_SUMMARY.md`** - Technical upgrade details

### Data Directories
* **`data/raw/`** - Place your training images and videos here
  - `Frames_Word_Level/` - Gesture folders (A-Z, 0-9, custom words)
  - `processed_dataset.json` - Cached processed features
  
* **`models/`** - Trained model storage
  - `isl_trained_model.pth` - Final trained model (~340KB)
  - `training_checkpoint.pth` - Temporary checkpoint (auto-deleted)
  
* **`logs/`** - Training logs and visualizations
  - `evaluation_results.json` - Accuracy metrics
  - `training_plots/` - Loss and accuracy graphs, confusion matrix
  
* **`exports/`** - Exported translations and recordings

---

## 🆘 Troubleshooting

### Camera Issues

* **"Camera shows a static image / tutorial screen"**
  - Your computer has a virtual camera (like BYOM). The app automatically tries camera indices 0, 1, and 2. It should find your real webcam. If it doesn't, check the terminal output to see which camera opened successfully.
  
* **"Camera feed is laggy"**
  - The app uses frame skipping (predicts every 5 frames) to reduce lag. You can adjust `prediction_interval` in `config.py` under `APP_CONFIG`.
  - Lower = more responsive but slower, Higher = faster but less responsive.
  - Close other camera applications that might be using the webcam.

* **"Camera not detected"**
  - Ensure webcam is connected and enabled in device manager
  - Try running: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
  - Check camera permissions in Windows settings

### Model & Training Issues

* **"No model found" Error**
  - You must run `python train_model.py` first to create the `isl_trained_model.pth` file.
  - Check that `models/` directory exists and is not empty.

* **"Predictions are inaccurate"**
  - **For Image-only training:** Static images are less effective than videos. Consider adding video data to your dataset for better accuracy.
  - **Solution:** Add `.mp4`/`.avi`/`.mov` files to your gesture folders and retrain. Videos capture the dynamic nature of sign language better than static images.
  - **Recommended:** Mix of 20-50% videos with images for optimal performance.
  - Ensure good lighting and clear hand visibility during recognition.
  - Model requires at least 20-30 samples per gesture class for good accuracy.

* **"Training is very slow"**
  - **Best solution:** Use GPU acceleration (see GPU Setup section above)
  - With GPU: Training takes 5-10 minutes
  - Without GPU: Training takes 45-60 minutes
  - Reduce `max_samples_per_class` in `config.py` to speed up training

* **"Out of Memory during training"**
  - **If using GPU:** Reduce batch size in `config.py`: `"batch_size": 16` or `8`
  - Reduce `max_samples_per_class` to `1000` instead of `1500`
  - Close other GPU-intensive applications
  - If using CPU: Close background applications to free RAM

### GPU Issues

* **"CUDA not available" after PyTorch installation**
  - Restart your terminal/IDE
  - Verify installation: `python -c "import torch; print(torch.cuda.is_available())"`
  - Reinstall PyTorch with CUDA:
    ```powershell
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
  - Run `python verify_gpu.py` for detailed diagnostics

* **"GPU Out of Memory"**
  - Reduce batch size: Edit `config.py` → `"batch_size": 16` or `8`
  - Reduce samples: `"max_samples_per_class": 1000`
  - Lower GPU memory fraction: `"gpu_memory_fraction": 0.6`
  - Close other GPU applications (games, video editing software)

* **"Training slower than expected with GPU"**
  - Check if GPU is actually being used: Run `nvidia-smi -l 1` in another terminal
  - Ensure laptop is plugged in (not on battery saver mode)
  - Check for thermal throttling (keep laptop well-ventilated)
  - Background apps might be using GPU - close unnecessary programs

### Application Performance

* **"App is slow or freezing"**
  - Close other camera applications
  - Ensure good lighting (less processing needed)
  - Check if your CPU is overloaded (Task Manager)
  - Adjust `prediction_interval` to higher value (e.g., 10) in `config.py`

* **"Signs are being added automatically!"**
  - This has been fixed! You now need to manually click the "➕ Add Sign" button to add recognized signs to your sentence.

* **"Text-to-Speech not working"**
  - **Online mode (gTTS):** Requires internet connection
  - **Offline mode (pyttsx3):** Should work without internet
  - Check TTS mode in `config.py`: `"tts_mode": "hybrid"` for automatic fallback
  - Verify audio output device is connected and not muted

### Dataset Issues

* **"No data loaded! Please check your dataset"**
  - Ensure `data/raw/Frames_Word_Level/` exists
  - Check that gesture folders (A, B, C, etc.) contain image/video files
  - Verify file formats: `.jpg`, `.png`, `.mp4`, `.avi`, `.mov`
  - Run `python analyze_dataset.py` to check dataset structure

* **"Class X has very few samples"**
  - Add more images/videos for that gesture (aim for 20-50 per class)
  - Training will still work but accuracy may be lower for under-represented classes

### Installation Issues

* **"ModuleNotFoundError: No module named X"**
  - Ensure virtual environment is activated: `.venv\Scripts\activate`
  - Reinstall requirements: `pip install -r requirements.txt`
  - For GPU support: See GPU Setup section

* **"ImportError: DLL load failed"**
  - Install Visual C++ Redistributable from Microsoft
  - Reinstall PyTorch: `pip install torch --force-reinstall`

### Getting Help

If you encounter issues not listed here:

1. **Check logs:** Look at terminal output for error messages
2. **Run diagnostics:**
   - `python verify_gpu.py` - Test GPU setup
   - `python analyze_dataset.py` - Check dataset
3. **Consult documentation:**
   - [`GPU_SETUP_GUIDE.md`](GPU_SETUP_GUIDE.md) - GPU troubleshooting
   - [`UPGRADE_SUMMARY.md`](UPGRADE_SUMMARY.md) - Technical details
4. **GitHub Issues:** Report bugs on the repository issues page

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
- **Translation:** googletrans (Google Translate API)
- **Text-to-Speech:** Hybrid system with gTTS (online, natural voice) and pyttsx3 (offline fallback)
- **Audio Processing:** pydub (for audio playback and manipulation)
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
2. **Hybrid TTS:** Uses gTTS (online, natural voice) with automatic fallback to pyttsx3 (offline) for reliable audio output
3. **Thread-Safe TTS:** Text-to-speech uses mutex locks to prevent concurrent execution errors
4. **Manual Sign Addition:** Users control when to add signs, preventing unwanted auto-additions
5. **BGR→RGB Conversion:** Optimized to convert once per frame, not multiple times
6. **Multi-Camera Detection:** Automatically tries camera indices 0, 1, 2 to find real webcam

### TTS Configuration

The text-to-speech system can be configured in `config.py` under `TRANSLATION_CONFIG`:

```python
"tts_mode": "hybrid"  # Options: "hybrid", "online", "offline"
```

- **hybrid** (default): Tries gTTS (online, natural-sounding) first, automatically falls back to pyttsx3 (offline) if internet is unavailable
- **online**: Uses only gTTS (requires internet connection)
- **offline**: Uses only pyttsx3 (works without internet)

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
* **googletrans** - Multi-language translation support (Google Translate API)
* **gTTS** - Online text-to-speech with natural voice quality
* **pyttsx3** - Offline text-to-speech functionality
* **pydub** - Audio processing and playback
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

---

## 📚 Complete Feature List

### Core Functionality
✅ Real-time ISL gesture recognition from webcam  
✅ 36+ gesture classes (A-Z, 0-9, custom words)  
✅ Sequential pattern analysis (30-frame LSTM)  
✅ Confidence-based filtering (55-85% thresholds)  
✅ Manual sentence building with "Add Sign" button  
✅ Multi-language translation (8 Indian languages)  
✅ Hybrid text-to-speech (online + offline)  

### GPU Acceleration ⚡
✅ Automatic GPU/CPU detection  
✅ 6-10x faster training with NVIDIA GPU  
✅ Mixed precision (FP16) for 2x additional speedup  
✅ cuDNN optimization for faster convolutions  
✅ Smart memory management and cleanup  
✅ Real-time performance monitoring  

### Training System
✅ Supports images (.jpg, .png) and videos (.mp4, .avi, .mov)  
✅ Automatic video frame extraction (~30 frames/video)  
✅ Class balancing (max 1500 samples/class)  
✅ Data augmentation with synthetic noise  
✅ Stratified train/val/test splits (70/10/20)  
✅ Comprehensive evaluation metrics  
✅ Training plots and confusion matrices  
✅ Automatic model checkpointing  

### Performance Optimizations
✅ Frame skipping (predict every 5th frame)  
✅ Prediction caching and timeout  
✅ Multi-camera detection (indices 0, 1, 2)  
✅ Thread-safe TTS with mutex locks  
✅ GPU memory auto-cleanup  
✅ Efficient batch processing  

### Utilities & Tools
✅ `verify_gpu.py` - GPU testing and benchmarks  
✅ `gpu_utils.py` - Centralized GPU management  
✅ `analyze_dataset.py` - Dataset analysis tool  
✅ `GPU_SETUP_GUIDE.md` - Complete GPU setup docs  
✅ `UPGRADE_SUMMARY.md` - Technical upgrade details  

---

## 🎯 Quick Command Reference

```powershell
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# GPU Setup (Recommended)
nvidia-smi  # Check GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python verify_gpu.py  # Verify GPU works

# Training
python train_model.py  # Train model (5-10 min with GPU, 45-60 min CPU)

# Run Application  
python app.py  # Launch GUI
# OR
run_isl_bridge.bat  # Windows launcher

# Analysis
python analyze_dataset.py  # Check dataset structure

# GPU Monitoring (during training)
nvidia-smi -l 1  # Real-time GPU usage (updates every second)
```

---

## 📊 Performance Benchmarks

### Training Performance (100 epochs, 53K samples)

| Hardware | Time | Per Epoch | Speedup |
|----------|------|-----------|---------|
| **CPU** (i5-8250U) | 45-60 min | 30-40 sec | Baseline |
| **GPU** (RTX 3050 6GB) | 5-10 min | 3-5 sec | **6-10x** ⚡ |
| **GPU** (RTX 4060 8GB) | 3-5 min | 2-3 sec | **12-15x** ⚡ |

### Inference Performance (per frame)

| Hardware | Latency | FPS | Speedup |
|----------|---------|-----|---------|
| **CPU** (i5-8250U) | 50-80 ms | 12-20 FPS | Baseline |
| **GPU** (RTX 3050) | 8-15 ms | 65-125 FPS | **5-7x** ⚡ |
| **GPU** (RTX 4060) | 5-10 ms | 100-200 FPS | **8-12x** ⚡ |

### Memory Usage

| Component | CPU Mode | GPU Mode |
|-----------|----------|----------|
| **Model Size** | 340 KB | 340 KB |
| **System RAM** | 2-3 GB | 1-2 GB |
| **GPU VRAM** | N/A | 500-800 MB |
| **Storage** | 2 GB | 5 GB (with CUDA) |

---

## 🔧 Advanced Configuration

### Adjusting GPU Memory (config.py)

```python
GPU_CONFIG = {
    "gpu_memory_fraction": 0.8,  # Use 80% of GPU memory
    # For 4GB GPUs: set to 0.6
    # For 6GB GPUs: set to 0.8
    # For 8GB+ GPUs: set to 0.9
}
```

### Adjusting Batch Size

```python
TRAINING_CONFIG = {
    "batch_size": 32,  # Default
    # For OOM errors: try 16 or 8
    # For fast GPUs: try 64 or 128
}
```

### Adjusting Frame Prediction Rate

```python
APP_CONFIG = {
    "prediction_interval": 5,  # Predict every 5th frame
    # Lower (3): More responsive, higher CPU
    # Higher (10): Faster, less responsive
}
```

### Adjusting Confidence Thresholds (model.py)

```python
# Line ~290 in model.py
if confidence_score < 0.55:  # Base threshold
    return "—", 0.0

# For stricter filtering: increase to 0.65 or 0.70
# For more lenient: decrease to 0.45 or 0.50
```

---

## 🌐 Extending to New Gestures

### Example: Adding "HELLO" Gesture

1. **Create folder:**
   ```
   data/raw/Frames_Word_Level/HELLO/
   ```

2. **Add data** (20-50 samples recommended):
   ```
   HELLO/
   ├── hello_1.jpg
   ├── hello_2.jpg
   ├── hello_video1.mp4
   ├── hello_video2.mp4
   └── ...
   ```

3. **Train:**
   ```powershell
   python train_model.py
   ```

4. **Use in app:**
   - Model automatically recognizes "HELLO"
   - No code changes needed!

### Best Practices for Custom Gestures

✅ **Use Videos**: Better for dynamic gestures  
✅ **Mix Data**: Combine images + videos  
✅ **Sufficient Samples**: 20-50 per gesture minimum  
✅ **Good Lighting**: Consistent across all samples  
✅ **Clear Hands**: Ensure hands are visible  
✅ **Varied Angles**: Multiple perspectives  
✅ **Natural Speed**: Perform gestures at normal pace  

---

## 📞 Support & Contribution

### Getting Help
- 📧 **Issues**: [GitHub Issues](https://github.com/Krish-codex/ISLBridge-Project/issues)
- 📖 **Documentation**: See README.md, GPU_SETUP_GUIDE.md, UPGRADE_SUMMARY.md
- 🔧 **Diagnostics**: Run `python verify_gpu.py` and `python analyze_dataset.py`

### Contributing
Contributions are welcome! Areas for improvement:
- Additional gesture datasets
- Improved model architectures
- Better UI/UX designs
- Multi-platform testing
- Documentation improvements
- Bug fixes and optimizations

### Acknowledgments
Special thanks to:
- Engineering Clinics course faculty
- Indian Sign Language community
- MediaPipe and PyTorch teams
- Open-source community

---

## 📜 Version History

### v2.0 (November 2025) - GPU Acceleration Update
- ✅ GPU support with automatic detection
- ✅ Mixed precision training (FP16)
- ✅ 6-10x training speedup
- ✅ Consolidated training scripts
- ✅ Centralized GPU utilities
- ✅ Comprehensive documentation

### v1.0 (October 2025) - Initial Release
- ✅ Real-time ISL recognition
- ✅ 36 gesture classes (A-Z, 0-9)
- ✅ Multi-language translation
- ✅ Hybrid TTS system
- ✅ Desktop GUI application

---
