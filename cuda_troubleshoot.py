#!/usr/bin/env python
# coding: utf-8

"""
CUDA Troubleshooting Script for PyTorch
Automatically detects and fixes CUDA-related issues
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    if description:
        print(f"Description: {description}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print("✓ Success")
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed (exit code: {e.returncode})")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, e.stderr.strip()

def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print("\n=== CHECKING PYTORCH CUDA STATUS ===")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("CUDA is not available in PyTorch")
            return False

    except ImportError:
        print("PyTorch is not installed")
        return False

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed."""
    print("\n=== CHECKING NVIDIA DRIVERS ===")

    # Check for nvidia-smi
    success, output = run_command("nvidia-smi", "Checking NVIDIA drivers")
    if success:
        # Extract CUDA version from nvidia-smi output
        for line in output.split('\n'):
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip()
                print(f"NVIDIA CUDA version: {cuda_version}")
                return True, cuda_version
        return True, "Unknown"
    else:
        print("NVIDIA drivers not found or not working")
        return False, None

def install_cuda_pytorch():
    """Install PyTorch with CUDA support."""
    print("\n=== INSTALLING PYTORCH WITH CUDA ===")

    # First uninstall existing PyTorch
    print("Uninstalling existing PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y", "Removing CPU-only PyTorch")

    # Try different CUDA versions, starting with the latest
    cuda_versions = ["cu124", "cu121", "cu118"]
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    for cuda_ver in cuda_versions:
        print(f"\nTrying PyTorch with CUDA {cuda_ver}...")
        cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_ver}"
        success, output = run_command(cmd, f"Installing PyTorch with CUDA {cuda_ver}")

        if success:
            print(f"✓ Successfully installed PyTorch with CUDA {cuda_ver}")
            return True
        else:
            print(f"✗ Failed to install PyTorch with CUDA {cuda_ver}")

    print("Failed to install PyTorch with CUDA support")
    return False

def main():
    """Main troubleshooting function."""
    print("CUDA Troubleshooting Script")
    print("=" * 40)

    # Check if we're on Windows
    if platform.system() != "Windows":
        print("This script is designed for Windows. For other platforms, please check PyTorch installation docs.")
        return

    # Check NVIDIA drivers first
    drivers_ok, cuda_version = check_nvidia_drivers()
    if not drivers_ok:
        print("\n❌ NVIDIA drivers not detected!")
        print("Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        print("Make sure to install the Game Ready drivers, not Studio drivers.")
        return

    # Check PyTorch CUDA status
    cuda_ok = check_pytorch_cuda()
    if cuda_ok:
        print("\n✅ CUDA is working correctly!")
        return

    # Try to install CUDA PyTorch
    print("\n❌ CUDA not available. Attempting to fix...")
    install_success = install_cuda_pytorch()

    if install_success:
        # Verify the installation
        print("\nVerifying installation...")
        cuda_ok = check_pytorch_cuda()
        if cuda_ok:
            print("\n✅ CUDA installation successful!")
            print("You can now run your PyTorch scripts with GPU acceleration.")
        else:
            print("\n❌ Installation completed but CUDA still not available.")
            print("This might be due to:")
            print("- Incompatible CUDA version")
            print("- Driver/CUDA version mismatch")
            print("- Python version compatibility issues")
            print("\nTry installing manually: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("\n❌ Failed to install PyTorch with CUDA support.")
        print("Please check your internet connection and try again.")
        print("Or install manually from: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    main()