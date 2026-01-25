#!/usr/bin/env python3
"""
GPU Detection and Library Installation Script
Detects NVIDIA GPU, CUDA versions, and installs compatible ML libraries
"""

import subprocess
import sys
import re
import platform


def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"Error: {e}", 1


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("Checking for NVIDIA GPU...")
    output, returncode = run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    if returncode == 0 and output:
        print(f"Found GPU: {output}")
        return True, output
    else:
        print("No NVIDIA GPU found or nvidia-smi not available")
        return False, None


def get_compute_capability():
    """Get GPU compute capability"""
    print("\nChecking compute capability...")
    output, returncode = run_command("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
    if returncode == 0 and output:
        compute_cap = output.strip()
        print(f"Compute Capability: {compute_cap}")
        return compute_cap
    return None


def get_cuda_driver_version():
    """Get CUDA driver version from nvidia-smi"""
    print("\nChecking CUDA driver version...")
    output, returncode = run_command("nvidia-smi")
    if returncode == 0:
        # Extract CUDA version from output (appears as "CUDA Version: 12.2")
        match = re.search(r'CUDA Version:\s+(\d+\.\d+)', output)
        if match:
            version = match.group(1)
            print(f"CUDA Driver Version: {version}")
            return version
    print("Could not detect CUDA driver version")
    return None


def get_system_arch():
    """Get system architecture"""
    print("\nChecking system architecture...")
    arch = platform.machine()
    print(f"Architecture: {arch}")
    return arch


def get_python_version():
    """Get Python version"""
    print("\nChecking Python version...")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python Version: {version}")
    return version


def recommend_cuda_version(compute_cap, driver_version):
    """Recommend CUDA version based on GPU and driver"""
    print("\nDetermining recommended CUDA version...")
    
    # Parse compute capability
    if compute_cap:
        major, minor = compute_cap.split('.')
        compute_cap_float = float(compute_cap)
    else:
        print("Cannot determine compute capability, defaulting to CUDA 12.1")
        return "12.1", "cu121"
    
    # Minimum CUDA based on compute capability
    if compute_cap_float >= 9.0:  # Hopper
        min_cuda = "11.8"
        recommended = "12.1"
    elif compute_cap_float >= 8.0:  # Ampere/Ada
        min_cuda = "11.1"
        recommended = "12.1"
    elif compute_cap_float >= 7.0:  # Volta/Turing
        min_cuda = "10.0"
        recommended = "11.8"
    else:  # Older
        min_cuda = "10.0"
        recommended = "11.8"
    
    # Check if driver supports recommended version
    if driver_version:
        driver_major = int(driver_version.split('.')[0])
        if driver_major < 12:
            recommended = "11.8"
    
    # Map to PyTorch wheel suffix
    cuda_suffix_map = {
        "11.8": "cu118",
        "12.1": "cu121",
        "12.2": "cu121",  # Use 12.1 wheels
        "12.3": "cu121",
        "12.4": "cu121",
    }
    
    cuda_suffix = cuda_suffix_map.get(recommended, "cu121")
    
    print(f"Recommended CUDA: {recommended} (PyTorch: {cuda_suffix})")
    return recommended, cuda_suffix


def generate_install_commands(cuda_suffix, arch):
    """Generate pip install commands"""
    print("\nGenerating installation commands...")
    
    is_arm = arch.lower() in ['aarch64', 'arm64']
    
    commands = []
    
    # PyTorch
    pytorch_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
    commands.append(f"pip3 install torch torchvision torchaudio --index-url {pytorch_url}")
    
    # Common ML libraries
    commands.append("pip3 install transformers accelerate peft datasets")
    
    # bitsandbytes (may need special handling for ARM)
    if is_arm:
        print("Note: bitsandbytes may require special ARM build")
        commands.append("# pip3 install bitsandbytes  # May need ARM-specific version")
    else:
        commands.append("pip3 install bitsandbytes")
    
    # DeepSpeed (may not work on all platforms)
    commands.append("pip3 install deepspeed")
    
    # Flash Attention (if compute capability supports it)
    commands.append("# pip3 install flash-attn --no-build-isolation  # Requires compute capability 8.0+")
    
    return commands


def check_existing_pytorch():
    """Check if PyTorch is already installed"""
    print("\nChecking existing PyTorch installation...")
    try:
        import torch
        print(f"PyTorch {torch.__version__} already installed")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   PyTorch CUDA version: {torch.version.cuda}")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False


def main():
    print("=" * 70)
    print("GPU Detection and ML Library Installation Helper")
    print("=" * 70)
    
    # Detection phase
    gpu_exists, gpu_name = check_nvidia_gpu()
    if not gpu_exists:
        print("\nExiting: No NVIDIA GPU detected")
        sys.exit(1)
    
    compute_cap = get_compute_capability()
    driver_version = get_cuda_driver_version()
    arch = get_system_arch()
    python_version = get_python_version()
    
    pytorch_installed = check_existing_pytorch()
    
    # Recommendation phase
    recommended_cuda, cuda_suffix = recommend_cuda_version(compute_cap, driver_version)
    
    # Generate commands
    install_commands = generate_install_commands(cuda_suffix, arch)
    
    # Summary report
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    print(f"GPU:                  {gpu_name}")
    print(f"Compute Capability:   {compute_cap}")
    print(f"CUDA Driver:          {driver_version}")
    print(f"Architecture:         {arch}")
    print(f"Python:               {python_version}")
    print(f"Recommended CUDA:     {recommended_cuda}")
    print(f"PyTorch CUDA suffix:  {cuda_suffix}")
    
    # Installation options
    print("\n" + "=" * 70)
    print("INSTALLATION OPTIONS")
    print("=" * 70)
    print("\n1. Auto-install (execute commands now)")
    print("2. Show commands only (copy/paste manually)")
    print("3. Save to install.sh script")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        print("\nInstalling packages...")
        for cmd in install_commands:
            if cmd.startswith("#"):
                print(f"Skipping: {cmd}")
                continue
            print(f"\nRunning: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"Command failed with return code {result.returncode}")
        print("\nInstallation complete!")
        
    elif choice == "2":
        print("\nCopy and run these commands:")
        print("-" * 70)
        for cmd in install_commands:
            print(cmd)
        print("-" * 70)
        
    elif choice == "3":
        script_name = "install_ml_libs.sh"
        with open(script_name, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated ML library installation script\n\n")
            for cmd in install_commands:
                f.write(cmd + "\n")
        subprocess.run(f"chmod +x {script_name}", shell=True)
        print(f"\nSaved to {script_name}")
        print(f"Run with: ./{script_name}")
        
    else:
        print("\nExiting without installation")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()