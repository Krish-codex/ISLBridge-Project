"""
GPU Utility Functions for ISL Bridge
Handles GPU detection, optimization, and memory management
"""
import torch
import logging

logger = logging.getLogger(__name__)

def get_device_info():
    """Get detailed device information"""
    info = {
        'has_cuda': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'device_name': None,
        'total_memory_gb': None,
        'cuda_version': None,
        'cudnn_enabled': None,
        'num_gpus': 0,
        'cpu_threads': torch.get_num_threads()
    }
    
    if info['has_cuda']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['cuda_version'] = torch.version.cuda
        info['cudnn_enabled'] = torch.backends.cudnn.enabled
        info['num_gpus'] = torch.cuda.device_count()
    
    return info

def print_device_info():
    """Print formatted device information"""
    info = get_device_info()
    
    print("\n" + "=" * 70)
    print("COMPUTE DEVICE INFORMATION")
    print("=" * 70)
    
    if info['has_cuda']:
        print(f"✅ GPU Available: {info['device_name']}")
        print(f"   Memory: {info['total_memory_gb']:.2f} GB")
        print(f"   CUDA Version: {info['cuda_version']}")
        print(f"   cuDNN Enabled: {info['cudnn_enabled']}")
        print(f"   Number of GPUs: {info['num_gpus']}")
        print(f"   Device: {info['device']}")
        print("\n💡 GPU acceleration is ENABLED for faster training/inference")
    else:
        print(f"⚠️  No GPU detected - using CPU")
        print(f"   CPU Threads: {info['cpu_threads']}")
        print(f"   Device: {info['device']}")
        print("\n💡 To enable GPU acceleration:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA Toolkit (11.8 or 12.1)")
        print("   3. Install PyTorch with CUDA:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 70 + "\n")
    
    return info

def optimize_device():
    """Apply device-specific optimizations"""
    if torch.cuda.is_available():
        # Enable cuDNN autotuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN autotuner for GPU optimization")
        
        # Enable TF32 on Ampere GPUs for faster matmul
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("Enabled TF32 for faster matrix operations")
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for cuDNN operations")
        
        return torch.device('cuda')
    else:
        # CPU optimizations
        logger.info(f"Using CPU with {torch.get_num_threads()} threads")
        return torch.device('cpu')

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
    }

def print_gpu_memory():
    """Print GPU memory usage"""
    mem = get_gpu_memory_info()
    if mem:
        print(f"GPU Memory: {mem['allocated_mb']:.0f}MB / {mem['total_gb']*1024:.0f}MB "
              f"({mem['allocated_mb']/(mem['total_gb']*1024)*100:.1f}%)")
    else:
        print("CPU mode - no GPU memory to display")

def cleanup_gpu_memory():
    """Clean up GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU memory cache cleared")

def get_optimal_batch_size(model, input_shape, device):
    """
    Determine optimal batch size based on available memory
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        device: torch.device
    
    Returns:
        Optimal batch size
    """
    if device.type == 'cpu':
        # CPU: use moderate batch size
        return 32
    
    # GPU: test different batch sizes
    batch_sizes = [64, 32, 16, 8]
    
    for batch_size in batch_sizes:
        try:
            # Test with dummy input
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            cleanup_gpu_memory()
            logger.info(f"Optimal batch size: {batch_size}")
            return batch_size
        except RuntimeError as e:
            if 'out of memory' in str(e):
                cleanup_gpu_memory()
                continue
            raise
    
    return 8  # Fallback to smallest size

def enable_mixed_precision():
    """
    Check if mixed precision training is available
    
    Returns:
        Boolean indicating if AMP is available
    """
    if torch.cuda.is_available():
        # Check if GPU supports mixed precision
        try:
            # Test mixed precision
            with torch.cuda.amp.autocast():
                dummy = torch.randn(1, 10).cuda()
                _ = dummy * 2
            return True
        except:
            return False
    return False

def get_training_config():
    """Get optimized training configuration based on device"""
    device = optimize_device()
    info = get_device_info()
    
    config = {
        'device': device,
        'use_amp': enable_mixed_precision(),
        'num_workers': 4 if info['has_cuda'] else 2,
        'pin_memory': info['has_cuda'],
        'persistent_workers': info['has_cuda'],
    }
    
    logger.info(f"Training config: AMP={config['use_amp']}, "
                f"Workers={config['num_workers']}, "
                f"Pin Memory={config['pin_memory']}")
    
    return config

if __name__ == "__main__":
    # Test GPU utilities
    print_device_info()
    
    device = optimize_device()
    print(f"\nOptimized device: {device}")
    
    if torch.cuda.is_available():
        print("\nGPU Memory Before:")
        print_gpu_memory()
        
        # Test allocation
        test_tensor = torch.randn(1000, 1000).to(device)
        
        print("\nGPU Memory After Test Allocation:")
        print_gpu_memory()
        
        cleanup_gpu_memory()
        
        print("\nGPU Memory After Cleanup:")
        print_gpu_memory()
    
    print("\nMixed Precision Available:", enable_mixed_precision())
    
    config = get_training_config()
    print(f"\nRecommended Training Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
