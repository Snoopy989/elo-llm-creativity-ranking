"""
Simple Llama 2 7B Model Downloader
===================================

Downloads Llama 2 7B model from Hugging Face.

Requirements:
pip install transformers torch huggingface_hub
"""

import os
from pathlib import Path

def download_llama_2_7b(save_directory="./downloaded_models", auth_token=None):
    """
    Download Llama 2 7B model from Hugging Face.

    Args:
        save_directory (str): Directory to save the model
        auth_token (str): Hugging Face authentication token (required)

    Returns:
        str: Path to downloaded model directory
    """
    if not auth_token:
        raise ValueError("Hugging Face authentication token is required for Llama 2 models")

    try:
        from huggingface_hub import snapshot_download

        # Model identifier
        model_name = "meta-llama/Llama-2-7b-hf"

        # Create save directory
        save_path = Path(save_directory) / "llama-2-7b"
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading Llama 2 7B to: {save_path}")

        # Download the model
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(save_path),
            local_dir_use_symlinks=False,
            use_auth_token=auth_token
        )

        print(f"Model downloaded successfully to: {downloaded_path}")
        return str(save_path)

    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    # Replace 'your_token_here' with your actual Hugging Face token
    token = "your_token_here"

    if token == "your_token_here":
        print("Please set your Hugging Face token in the script")
        print("Get your token from: https://huggingface.co/settings/tokens")
    else:
        download_llama_2_7b(auth_token=token)