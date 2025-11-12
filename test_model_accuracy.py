"""Quick test to check model accuracy"""
import json
import numpy as np
from model import ISLModel
from config import RAW_DATA_DIR, TRAINING_CONFIG, MODELS_DIR

print("Loading model...")
model = ISLModel()

# Load trained weights
model_path = MODELS_DIR / "isl_trained_model"
try:
    label_classes = model.load_model(str(model_path))
    print(f"✓ Model loaded with {model.num_classes} classes")
    print(f"✓ Label classes: {label_classes[:10] if label_classes else 'None'}...")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load processed dataset
dataset_file = RAW_DATA_DIR / "processed_dataset.json"
if dataset_file.exists():
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n✓ Dataset loaded: {len(data['features'])} samples")
    
    # Test on a sample
    X = np.array(data['features'])
    y = np.array(data['labels'])
    
    print(f"✓ Data shape: X={X.shape}, y={y.shape}")
    
    # Get sequence length from config (must match training!)
    sequence_length = TRAINING_CONFIG.get("sequence_length", 30)
    print(f"✓ Using sequence length: {sequence_length} (matching training)")
    
    # Make predictions on first 100 samples
    print("\nTesting predictions...")
    correct = 0
    total = min(100, len(X))
    errors = 0
    
    for i in range(total):
        try:
            # Create 30-frame sequence by repeating single frame (matching training format)
            single_frame = X[i]
            sequence = np.tile(single_frame, (sequence_length, 1))  # Repeat 30 times
            sample = sequence.reshape(1, sequence_length, -1)  # Shape: (1, 30, 166)
            
            result = model.predict_class(sample)
            if isinstance(result, tuple) and len(result) >= 2:
                pred_class, confidence = result[0], result[1]
            else:
                pred_class = result
            
            if pred_class == y[i]:
                correct += 1
            
            # Print first 10 for debugging
            if i < 10:
                match = "✓" if pred_class == y[i] else "✗"
                conf_str = f"({confidence:.1%})" if isinstance(result, tuple) else ""
                print(f"  Sample {i}: True={y[i]:<3} Pred={pred_class:<3} {conf_str} {match}")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only print first 3 errors
                print(f"  Error on sample {i}: {e}")
    
    if i < total - 1:
        total = i + 1  # Adjust if interrupted
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"{'='*60}")
    
    if accuracy < 70:
        print("\n⚠️  LOW ACCURACY! Model needs retraining with better parameters.")
    elif accuracy < 90:
        print("\n⚠️  MODERATE ACCURACY. Model could be improved.")
    else:
        print("\n✓ GOOD ACCURACY! Model is working well.")
else:
    print("✗ Dataset not found!")
