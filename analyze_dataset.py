"""
Dataset Analysis Tool for ISL Bridge
Analyzes current dataset and helps prepare for INCLUDE integration
"""
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def analyze_current_dataset():
    """Analyze the current ISL Bridge dataset"""
    print("🔍 Analyzing Current ISL Bridge Dataset")
    print("="*50)
    
    # Check current data structure
    frames_dir = Path("data/raw/Frames_Word_Level")
    
    if not frames_dir.exists():
        print("❌ Frames_Word_Level directory not found")
        return
    
    # Count gestures and frames
    gesture_stats = {}
    total_frames = 0
    
    for gesture_dir in frames_dir.iterdir():
        if gesture_dir.is_dir():
            frame_count = len(list(gesture_dir.glob("*.jpg"))) + len(list(gesture_dir.glob("*.png")))
            gesture_stats[gesture_dir.name] = frame_count
            total_frames += frame_count
    
    print(f"📊 Current Dataset Statistics:")
    print(f"   Gesture Classes: {len(gesture_stats)}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Average Frames per Gesture: {total_frames/len(gesture_stats):.1f}")
    
    # Show top gestures by frame count
    print(f"\n📈 Top 10 Gestures by Frame Count:")
    sorted_gestures = sorted(gesture_stats.items(), key=lambda x: x[1], reverse=True)
    for i, (gesture, count) in enumerate(sorted_gestures[:10], 1):
        print(f"   {i:2d}. {gesture:<20} : {count:3d} frames")
    
    # Check for low-frame gestures
    low_frame_gestures = [(g, c) for g, c in gesture_stats.items() if c < 5]
    if low_frame_gestures:
        print(f"\n⚠️  Gestures with < 5 frames ({len(low_frame_gestures)}):")
        for gesture, count in low_frame_gestures[:5]:
            print(f"   - {gesture}: {count} frames")
        if len(low_frame_gestures) > 5:
            print(f"   ... and {len(low_frame_gestures) - 5} more")
    
    return gesture_stats

def check_include_readiness():
    """Check if system is ready for INCLUDE dataset processing"""
    print("\n🚀 INCLUDE Dataset Integration Readiness")
    print("="*50)
    
    # Check disk space (approximate)
    import shutil
    free_space_gb = shutil.disk_usage(".").free / (1024**3)
    print(f"💾 Available Disk Space: {free_space_gb:.1f} GB")
    
    if free_space_gb < 10:
        print("   ⚠️  Warning: Less than 10GB free space")
        print("   ⚠️  INCLUDE processing may require 20-50GB")
    else:
        print("   ✅ Sufficient disk space available")
    
    # Check required packages
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found - install with: pip install opencv-python")
    
    # Check current model compatibility
    model_path = Path("models/isl_trained_model.pth")
    if model_path.exists():
        print(f"✅ Current model exists: {model_path}")
        print("   📝 Note: Will need retraining after adding INCLUDE data")
    else:
        print("❌ No trained model found")
    
    # Check config
    try:
        from config import ISL_CLASSES
        print(f"✅ Current ISL_CLASSES: {len(ISL_CLASSES)} gestures")
        print("   📝 Note: Will be updated after INCLUDE processing")
    except ImportError:
        print("❌ Cannot import ISL_CLASSES from config")

def generate_download_instructions():
    """Generate instructions for downloading INCLUDE dataset"""
    print("\n📥 INCLUDE Dataset Download Instructions")
    print("="*50)
    
    instructions = """
1. 🌐 Visit the INCLUDE dataset sources:
   - Kaggle: Search for "INCLUDE Indian Sign Language"
   - GitHub: Look for official INCLUDE repositories
   - Research papers: Check for dataset links

2. 📁 Create directory structure:
   mkdir -p data/raw/INCLUDE_videos

3. 📥 Download video files to:
   data/raw/INCLUDE_videos/

4. 🔧 Run preprocessing:
   python prepare_include_dataset.py --test  # Test with 10 videos first
   python prepare_include_dataset.py         # Full processing

5. 📝 Update configuration:
   Copy generated ISL_CLASSES to config.py

6. 🚀 Retrain model:
   python train_model.py

Common INCLUDE dataset formats:
- Video files: .mp4, .avi, .mov
- Naming: Usually gesture_name.mp4 or similar
- Size: Can be 10-50GB total
"""
    
    print(instructions)

def create_test_setup():
    """Create a small test setup for INCLUDE processing"""
    print("\n🧪 Creating Test Setup")
    print("="*50)
    
    test_dir = Path("data/raw/INCLUDE_videos_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created test directory: {test_dir}")
    print("\nTo test INCLUDE processing:")
    print("1. Copy 3-5 video files to data/raw/INCLUDE_videos_test/")
    print("2. Run: python prepare_include_dataset.py --source data/raw/INCLUDE_videos_test --test")
    print("3. Check results in data/raw/Frames_Word_Level/")

def main():
    """Main analysis function"""
    print("🤟 ISL Bridge Dataset Analyzer")
    print("="*60)
    
    # Analyze current dataset
    current_stats = analyze_current_dataset()
    
    # Check readiness for INCLUDE
    check_include_readiness()
    
    # Generate download instructions
    generate_download_instructions()
    
    # Create test setup
    create_test_setup()
    
    print("\n" + "="*60)
    print("Analysis complete! Follow the instructions above to integrate INCLUDE dataset.")

if __name__ == "__main__":
    main()