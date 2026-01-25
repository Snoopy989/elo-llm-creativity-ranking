#!/bin/bash
#
# NVIDIA Docker Setup Script
# Sets up NVIDIA Container Toolkit and pulls NGC PyTorch container
# Works on any Linux system with NVIDIA GPU and Docker installed
#

set -e

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_fail()    { echo -e "${RED}[FAIL]${NC} $1"; }

header() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

# Configuration
PYTORCH_CONTAINER="nvcr.io/nvidia/pytorch:24.05-py3"
TENSORFLOW_CONTAINER="nvcr.io/nvidia/tensorflow:24.05-tf2-py3"
WORKSPACE_DIR="$HOME/nvidia-workspace"

check_root() {
    if [ "$EUID" -eq 0 ]; then
        log_warning "Running as root. Consider running as regular user with sudo access."
    fi
}

check_os() {
    header "Checking Operating System"
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_NAME="$NAME"
        OS_VERSION="$VERSION_ID"
        OS_ID="$ID"
        log_info "Detected: $OS_NAME $OS_VERSION"
    else
        log_fail "Cannot detect OS. This script requires Linux."
        exit 1
    fi
    
    case "$OS_ID" in
        ubuntu|debian)
            PKG_MANAGER="apt"
            ;;
        rhel|centos|fedora|rocky|almalinux)
            PKG_MANAGER="yum"
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            fi
            ;;
        opensuse*|sles)
            PKG_MANAGER="zypper"
            ;;
        arch)
            PKG_MANAGER="pacman"
            ;;
        *)
            log_warning "Unknown distribution: $OS_ID. Will attempt Ubuntu/Debian method."
            PKG_MANAGER="apt"
            ;;
    esac
    
    log_info "Package manager: $PKG_MANAGER"
}

check_architecture() {
    header "Checking Architecture"
    
    ARCH=$(uname -m)
    log_info "Architecture: $ARCH"
    
    case "$ARCH" in
        x86_64|amd64)
            ARCH_SUPPORTED=true
            ;;
        aarch64|arm64)
            ARCH_SUPPORTED=true
            log_info "ARM64 detected (GH200/Jetson/ARM server)"
            ;;
        *)
            log_fail "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
}

check_nvidia_driver() {
    header "Checking NVIDIA Driver"
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_fail "nvidia-smi not found. Please install NVIDIA drivers first."
        echo ""
        echo "Install drivers with:"
        echo "  Ubuntu/Debian: sudo apt install nvidia-driver-550"
        echo "  RHEL/CentOS:   sudo dnf install nvidia-driver"
        echo ""
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log_fail "nvidia-smi failed. Driver may not be loaded."
        exit 1
    fi
    
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" || echo "unknown")
    
    log_success "GPU: $GPU_NAME"
    log_info "Driver: $DRIVER_VERSION"
    log_info "CUDA: $CUDA_VERSION"
}

check_docker() {
    header "Checking Docker"
    
    if ! command -v docker &> /dev/null; then
        log_fail "Docker not found. Please install Docker first."
        echo ""
        echo "Install Docker with:"
        echo "  curl -fsSL https://get.docker.com | sh"
        echo "  sudo usermod -aG docker \$USER"
        echo "  # Log out and back in"
        echo ""
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_fail "Docker daemon not running or permission denied."
        echo ""
        echo "Try:"
        echo "  sudo systemctl start docker"
        echo "  sudo usermod -aG docker \$USER"
        echo "  # Log out and back in"
        echo ""
        exit 1
    fi
    
    DOCKER_VERSION=$(docker --version | grep -oP "[0-9]+\.[0-9]+\.[0-9]+")
    log_success "Docker $DOCKER_VERSION is running"
}

check_nvidia_container_toolkit() {
    header "Checking NVIDIA Container Toolkit"
    
    if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_success "NVIDIA Container Toolkit is already configured"
        return 0
    else
        log_info "NVIDIA Container Toolkit not configured"
        return 1
    fi
}

install_nvidia_container_toolkit_apt() {
    log_info "Installing NVIDIA Container Toolkit (apt)..."
    
    # Add GPG key
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
    
    # Add repository
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    
    # Install
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
}

install_nvidia_container_toolkit_yum() {
    log_info "Installing NVIDIA Container Toolkit (yum/dnf)..."
    
    # Add repository
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
    
    # Install
    sudo $PKG_MANAGER install -y nvidia-container-toolkit
}

install_nvidia_container_toolkit_zypper() {
    log_info "Installing NVIDIA Container Toolkit (zypper)..."
    
    # Add repository
    sudo zypper ar https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
    
    # Install
    sudo zypper --gpg-auto-import-keys install -y nvidia-container-toolkit
}

install_nvidia_container_toolkit() {
    header "Installing NVIDIA Container Toolkit"
    
    case "$PKG_MANAGER" in
        apt)
            install_nvidia_container_toolkit_apt
            ;;
        yum|dnf)
            install_nvidia_container_toolkit_yum
            ;;
        zypper)
            install_nvidia_container_toolkit_zypper
            ;;
        *)
            log_fail "Unsupported package manager: $PKG_MANAGER"
            echo "Please install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
            ;;
    esac
    
    # Configure Docker runtime
    log_info "Configuring Docker runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker
    
    # Restart Docker
    log_info "Restarting Docker..."
    sudo systemctl restart docker
    
    # Wait for Docker to be ready
    sleep 3
    
    log_success "NVIDIA Container Toolkit installed"
}

verify_gpu_access() {
    header "Verifying GPU Access in Docker"
    
    log_info "Running test container..."
    
    if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi; then
        log_success "GPU is accessible from Docker containers"
        return 0
    else
        log_fail "GPU access test failed"
        return 1
    fi
}

pull_container() {
    header "Pulling PyTorch Container"
    
    log_info "Container: $PYTORCH_CONTAINER"
    log_info "This may take several minutes (container is ~20GB)..."
    echo ""
    
    if docker pull "$PYTORCH_CONTAINER"; then
        log_success "Container pulled successfully"
    else
        log_fail "Failed to pull container"
        exit 1
    fi
}

create_workspace() {
    header "Creating Workspace"
    
    mkdir -p "$WORKSPACE_DIR"
    log_success "Workspace created: $WORKSPACE_DIR"
    log_info "Mount this directory to persist your work"
}

create_run_script() {
    header "Creating Helper Scripts"
    
    # Main run script
    cat > "$WORKSPACE_DIR/run-pytorch.sh" << 'SCRIPT'
#!/bin/bash
# Run NVIDIA PyTorch container with GPU access
# Usage: ./run-pytorch.sh [additional docker args]

CONTAINER="nvcr.io/nvidia/pytorch:24.05-py3"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"

docker run --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE:/workspace" \
    -w /workspace \
    "$@" \
    "$CONTAINER"
SCRIPT
    chmod +x "$WORKSPACE_DIR/run-pytorch.sh"
    log_success "Created: $WORKSPACE_DIR/run-pytorch.sh"
    
    # Jupyter run script
    cat > "$WORKSPACE_DIR/run-jupyter.sh" << 'SCRIPT'
#!/bin/bash
# Run NVIDIA PyTorch container with Jupyter Lab
# Access at http://localhost:8888

CONTAINER="nvcr.io/nvidia/pytorch:24.05-py3"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"

docker run --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE:/workspace" \
    -w /workspace \
    -p 8888:8888 \
    "$@" \
    "$CONTAINER" \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
SCRIPT
    chmod +x "$WORKSPACE_DIR/run-jupyter.sh"
    log_success "Created: $WORKSPACE_DIR/run-jupyter.sh"
    
    # Test script
    cat > "$WORKSPACE_DIR/test-gpu.py" << 'SCRIPT'
#!/usr/bin/env python3
"""Test GPU access in PyTorch"""

import torch

print("=" * 60)
print("PyTorch GPU Test")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:    {torch.version.cuda}")
    print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"GPU count:       {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    
    # Quick test
    print("\nRunning computation test...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print("[SUCCESS] GPU computation works!")
else:
    print("[FAIL] CUDA not available")
SCRIPT
    log_success "Created: $WORKSPACE_DIR/test-gpu.py"
}

print_summary() {
    header "SETUP COMPLETE"
    
    echo ""
    echo "Workspace: $WORKSPACE_DIR"
    echo ""
    echo "Quick Start:"
    echo "  cd $WORKSPACE_DIR"
    echo ""
    echo "  # Interactive shell with GPU"
    echo "  ./run-pytorch.sh"
    echo ""
    echo "  # Jupyter Lab (access at http://localhost:8888)"
    echo "  ./run-jupyter.sh"
    echo ""
    echo "  # Test GPU inside container"
    echo "  ./run-pytorch.sh python test-gpu.py"
    echo ""
    echo "Your files in $WORKSPACE_DIR will be available at /workspace in the container."
    echo ""
}

main() {
    header "NVIDIA Docker Setup"
    
    check_root
    check_os
    check_architecture
    check_nvidia_driver
    check_docker
    
    if ! check_nvidia_container_toolkit; then
        install_nvidia_container_toolkit
    fi
    
    verify_gpu_access
    pull_container
    create_workspace
    create_run_script
    print_summary
    
    log_success "All done!"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        echo "NVIDIA Docker Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help"
        echo "  --check        Only check system, don't install"
        echo "  --pull-only    Only pull container (toolkit already installed)"
        echo ""
        exit 0
        ;;
    --check)
        check_os
        check_architecture
        check_nvidia_driver
        check_docker
        check_nvidia_container_toolkit && log_success "System ready" || log_warning "Toolkit not installed"
        exit 0
        ;;
    --pull-only)
        pull_container
        create_workspace
        create_run_script
        print_summary
        exit 0
        ;;
    *)
        main
        ;;
esac