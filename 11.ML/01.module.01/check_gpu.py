"""
Check GPU and CUDA availability
"""
import torch

def check_gpu():
    print("="*50)
    print("GPU DIAGNOSTICS")
    print("="*50)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("\nCUDA is not available. Please install PyTorch with CUDA support.")
        print("Run 'setup_pytorch.py' to install the correct version.")

if __name__ == "__main__":
    check_gpu()
