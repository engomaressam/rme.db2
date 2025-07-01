"""
Setup PyTorch with CUDA support
"""
import subprocess
import sys

def install_pytorch():
    print("Uninstalling current PyTorch...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    
    print("\nInstalling PyTorch with CUDA 11.8...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    
    print("\nInstallation complete!")
    print("Please restart your kernel and run the GPU check again.")

if __name__ == "__main__":
    install_pytorch()
