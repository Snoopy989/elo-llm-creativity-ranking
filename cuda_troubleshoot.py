#!/usr/bin/env python
# coding: utf-8

"""
CUDA Troubleshooting Script for PyTorch
Automatically detects and fixes CUDA-related issues
Improved version with better diagnostics and error handling
"""

import os
import sys
import subprocess
import platform
import struct
import importlib
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    if description:
        print(f"Description: {description}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("[SUCCESS]")
            return True, result.stdout.strip()
        else:
            print(f"[FAILED] (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False, result.stderr.strip()
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        return False, str(e)

def check_python_architecture():
    """Check Python architecture (32-bit vs 64-bit)."""
    print("\n=== CHECKING PYTHON ARCHITECTURE ===")
    bits = struct.calcsize("P") * 8
    print(f"Python architecture: {bits}-bit")
    print(f"Python version: {sys.version.split()[0]}")
    
    if bits == 32:
        print("[WARNING] You're using 32-bit Python. PyTorch with CUDA requires 64-bit Python!")
        print("Please install 64-bit Python from: https://www.python.org/downloads/")
        return False
    return True

def check_pytorch_installed():
    """Check if PyTorch is installed."""
    try:
        import torch
        return True, torch
    except ImportError:
        return False, None

def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print("\n=== CHECKING PYTORCH CUDA STATUS ===")

    torch_installed, torch = check_pytorch_installed()
    if not torch_installed:
        print("PyTorch is not installed")
        return False

    print(f"PyTorch version: {torch.__version__}")
    
    # Check if it's CPU-only by examining the version string
    is_cpu_only = "+cpu" in torch.__version__

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CPU-only build: {is_cpu_only}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("CUDA is not available in PyTorch")
        if is_cpu_only:
            print("[WARNING] PyTorch is installed as CPU-only build (detected '+cpu' in version)")
        return False

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed."""
    print("\n=== CHECKING NVIDIA DRIVERS ===")

    # Check for nvidia-smi
    success, output = run_command("nvidia-smi", "Checking NVIDIA drivers")
    if success:
        # Extract CUDA version from nvidia-smi output
        cuda_version = None
        gpu_name = None
        for line in output.split('\n'):
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
            if 'Name:' in line and gpu_name is None:
                gpu_name = line.split('Name:')[1].strip().split(' ')[0]
        
        print(f"NVIDIA CUDA version: {cuda_version if cuda_version else 'Unknown'}")
        if gpu_name:
            print(f"GPU detected: {gpu_name}")
        return True, cuda_version
    else:
        print("NVIDIA drivers not found or not working")
        return False, None

def clear_pip_cache():
    """Clear pip cache to ensure fresh downloads."""
    print("\n=== CLEARING PIP CACHE ===")
    success, _ = run_command("pip cache purge", "Clearing pip cache")
    if success:
        print("[SUCCESS] Pip cache cleared")
    return success

def uninstall_torch_completely():
    """Completely uninstall all torch packages."""
    print("\n=== UNINSTALLING TORCH COMPLETELY ===")
    packages = ["torch", "torchvision", "torchaudio", "pytorch"]
    
    for pkg in packages:
        print(f"Attempting to uninstall {pkg}...")
        run_command(f"pip uninstall {pkg} -y", f"Removing {pkg}")
    
    print("[SUCCESS] Torch packages uninstalled")
    return True

def install_cuda_pytorch():
    """Install PyTorch with CUDA support."""
    print("\n=== INSTALLING PYTORCH WITH CUDA ===")

    # Clear cache first
    clear_pip_cache()
    
    # Uninstall completely
    uninstall_torch_completely()
    
    # Map NVIDIA CUDA versions to PyTorch wheel indices
    cuda_wheel_map = {
        "13": "cu124",  # CUDA 13.x → cu124
        "12": "cu124",  # CUDA 12.4+
        "11": "cu121",  # CUDA 11.8
    }
    
    # Get NVIDIA CUDA version
    nvidia_ok, nvidia_cuda = check_nvidia_drivers()
    
    if nvidia_cuda:
        major_version = nvidia_cuda.split('.')[0]
        recommended_wheel = cuda_wheel_map.get(major_version, "cu124")
        print(f"\nNVIDIA CUDA {nvidia_cuda} detected → recommending PyTorch {recommended_wheel}")
    else:
        recommended_wheel = "cu124"
        print(f"\nCouldn't detect NVIDIA CUDA version → using default {recommended_wheel}")
    
    # Try versions: recommended first, then others
    cuda_versions = [recommended_wheel] + [v for v in ["cu124", "cu121", "cu118"] if v != recommended_wheel]

    for cuda_ver in cuda_versions:
        print(f"\n{'─'*60}")
        print(f"Attempting installation with CUDA {cuda_ver}...")
        print(f"{'─'*60}")
        
        cmd = f"pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_ver}"
        success, output = run_command(cmd, f"Installing PyTorch with CUDA {cuda_ver}")

        if success:
            print(f"[SUCCESS] Successfully installed PyTorch with CUDA {cuda_ver}")
            
            # Verify it's not CPU-only
            print("\nVerifying installation type...")
            torch_installed, torch = check_pytorch_installed()
            if torch_installed and "+cpu" not in torch.__version__:
                print(f"[SUCCESS] Verified: {torch.__version__} (not CPU-only)")
                return True, cuda_ver
            elif torch_installed and "+cpu" in torch.__version__:
                print(f"[WARNING] Installed version is CPU-only: {torch.__version__}")
                print("Trying next CUDA version...\n")
                uninstall_torch_completely()
                clear_pip_cache()
            else:
                print("[WARNING] Could not verify installation")
                return False, None
        else:
            print(f"[FAILED] Failed with CUDA {cuda_ver}")
            print("Trying next version...")

    print("\n" + "="*60)
    print("[FAILED] Failed to install PyTorch with CUDA support")
    print("="*60)
    return False, None

def main():
    """Main troubleshooting function."""
    print("\n" + "="*60)
    print("CUDA Troubleshooting Script (Enhanced Version)")
    print("="*60)

    # Check if we're on Windows
    if platform.system() != "Windows":
        print("This script is designed for Windows. For other platforms, please check PyTorch installation docs.")
        return

    # Step 1: Check Python architecture
    python_ok = check_python_architecture()
    if not python_ok:
        return

    # Step 2: Check NVIDIA drivers
    drivers_ok, cuda_version = check_nvidia_drivers()
    if not drivers_ok:
        print("\n[ERROR] NVIDIA drivers not detected!")
        print("Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        print("Make sure to install the Game Ready drivers, not Studio drivers.")
        return

    # Step 3: Check PyTorch CUDA status
    cuda_ok = check_pytorch_cuda()
    if cuda_ok:
        print("\n" + "="*60)
        print("[SUCCESS] CUDA is working correctly!")
        print("="*60)
        print("You can now run your PyTorch scripts with GPU acceleration.")
        return

    # Step 4: Try to install CUDA PyTorch
    print("\n" + "="*60)
    print("[ERROR] CUDA not available. Attempting to fix...")
    print("="*60)
    
    install_success, wheel_used = install_cuda_pytorch()

    if install_success:
        # Step 5: Verify the installation
        print("\n" + "="*60)
        print("VERIFYING INSTALLATION...")
        print("="*60)
        
        # Need to reimport torch after installation
        if 'torch' in sys.modules:
            del sys.modules['torch']
        
        cuda_ok = check_pytorch_cuda()
        if cuda_ok:
            print("\n" + "="*60)
            print("[SUCCESS] CUDA installation successful!")
            print("="*60)
            print(f"[SUCCESS] Using PyTorch wheel: {wheel_used}")
            print("[SUCCESS] You can now run your PyTorch scripts with GPU acceleration.")
            print("\nTest with: python -c \"import torch; print(torch.cuda.is_available())\"")
        else:
            print("\n" + "="*60)
            print("[WARNING] INSTALLATION ISSUE")
            print("="*60)
            print("Installation completed but CUDA still not available.")
            print("\nPossible causes:")
            print("  1. Wheel package cached locally (try: pip cache purge)")
            print("  2. NVIDIA driver version mismatch with CUDA version")
            print("  3. PyTorch wheel for your Python version not available")
            print("  4. System requires reboot after driver installation")
            print("\nTroubleshooting steps:")
            print("  1. Restart your computer")
            print("  2. Run: pip cache purge")
            print("  3. Try: pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("  4. If still failing, update NVIDIA drivers first")
    else:
        print("\n" + "="*60)
        print("[ERROR] Installation Failed")
        print("="*60)
        print("\nManual installation options:")
        print("1. Visit: https://pytorch.org/get-started/locally/")
        print("2. Select your configuration:")
        print("   - OS: Windows")
        print("   - Package: Pip")
        print("   - Language: Python")
        print("   - CUDA: 12.4 (or latest available)")
        print("\n3. Copy the command provided and run it")
        print("\nOr try manual CUDA 12.1 install:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir")

if __name__ == "__main__":
    main()