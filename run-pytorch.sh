#!/bin/bash
# Run NVIDIA PyTorch container with GPU access
# Usage: ./run-pytorch.sh [additional docker args]

CONTAINER="nvcr.io/nvidia/pytorch:24.05-py3"
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"

# Load token from .env file
if [ -f "$WORKSPACE/.env" ]; then
    export $(grep -v '^#' "$WORKSPACE/.env" | xargs)
fi

docker run --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE:/workspace" \
    -w /workspace \
    -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
    -e HF_TOKEN="$HUGGINGFACE_TOKEN" \
    "$CONTAINER" \
    "$@"
