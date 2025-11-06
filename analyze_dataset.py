"""
Dataset Analysis Tool for ISL Bridge
"""
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def analyze_current_dataset():
    print("🔍 Analyzing Current ISL Bridge Dataset")
    print("="*50)
    
    frames_dir = Path("data/raw/Frames_Word_Level")
    
    if not frames_dir.exists():
        print("❌ Frames_Word_Level directory not found")
        return
    
    gesture_stats = {}
    total_frames = 0
    
    for gesture_dir in frames_dir.iterdir():
        if gesture_dir.is_dir():
            image_count = len(list(gesture_dir.glob("*.jpg"))) + len(list(gesture_dir.glob("*.png")))
            video_count = len(list(gesture_dir.glob("*.mp4"))) + len(list(gesture_dir.glob("*.avi"))) + len(list(gesture_dir.glob("*.mov")))
            frame_count = image_count + video_count
            gesture_stats[gesture_dir.name] = frame_count
            total_frames += frame_count
    
    print(f"📊 Current Dataset Statistics:")
    print(f"   Gesture Classes: {len(gesture_stats)}")
    print(f"   Total Frames: {total_frames}")
    if len(gesture_stats) > 0:
        print(f"   Average Frames per Gesture: {total_frames/len(gesture_stats):.1f}")
    else:
        print(f"   Average Frames per Gesture: N/A (no gesture folders found)")
    
    print(f"\n📈 Top 10 Gestures by Frame Count:")
    sorted_gestures = sorted(gesture_stats.items(), key=lambda x: x[1], reverse=True)
    for i, (gesture, count) in enumerate(sorted_gestures[:10], 1):
        print(f"   {i:2d}. {gesture:<20} : {count:3d} frames")
    
    low_frame_gestures = [(g, c) for g, c in gesture_stats.items() if c < 5]
    if low_frame_gestures:
        print(f"\n⚠️  Gestures with < 5 frames ({len(low_frame_gestures)}):")
        for gesture, count in low_frame_gestures[:5]:
            print(f"   - {gesture}: {count} frames")
        if len(low_frame_gestures) > 5:
            print(f"   ... and {len(low_frame_gestures) - 5} more")
    
    return gesture_stats

def check_include_readiness():
    print("\n🚀 System Readiness Check")
    print("="*50)
    
    import shutil
    free_space_gb = shutil.disk_usage(".").free / (1024**3)
    print(f"💾 Available Disk Space: {free_space_gb:.1f} GB")
    
    if free_space_gb < 10:
        print("   ⚠️  Warning: Less than 10GB free space")
    else:
        print("   ✅ Sufficient disk space available")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__} (required for video processing)")
    except ImportError:
        print("❌ OpenCV not found - install with: pip install opencv-python")
    
    model_path = Path("models/isl_trained_model.pth")
    if model_path.exists():
        print(f"✅ Current model exists: {model_path}")
        print("   📝 Will be updated when you retrain with new gestures")
    else:
        print("⚠️  No trained model found - run 'python train_model.py' first")
    
    try:
        from config import ISL_CLASSES
        print(f"✅ Current ISL_CLASSES: {len(ISL_CLASSES)} gestures detected")
        print("   📝 Auto-updates when you add new gesture folders")
    except ImportError:
        print("❌ Cannot import ISL_CLASSES from config")

def generate_download_instructions():
    print("\n📥 Adding New Gestures")
    print("="*50)
    print("\nThe system is fully dynamic. Just add your data and train.")
    print("\nSteps:")
    print("1. Download your dataset (Kaggle, GitHub, etc.)")
    print("2. Create folders: data/raw/Frames_Word_Level/GESTURE_NAME/")
    print("3. Add your videos (.mp4, .avi, .mov) or images (.jpg, .png)")
    print("4. Run: python train_model.py")
    print("\nThe system handles everything else automatically.")

def create_test_setup():
    print("\n🧪 Quick Test Setup")
    print("="*50)
    print("\nTo test adding a new gesture:")
    print("1. Create folder: data\\raw\\Frames_Word_Level\\TEST_GESTURE")
    print("2. Add 3-5 images (.jpg/.png) or 1-2 videos (.mp4/.avi/.mov)")
    print("3. Run: python train_model.py")
    print("4. Check training log for 'Processing gesture: TEST_GESTURE'")

def main():
    print("🤟 ISL Bridge Dataset Analyzer")
    print("="*60)
    
    current_stats = analyze_current_dataset()
    check_include_readiness()
    generate_download_instructions()
    create_test_setup()
    
    print("\n" + "="*60)
    print("✅ ISL Bridge is fully dynamic - just add folders and train!")

if __name__ == "__main__":
    main()
