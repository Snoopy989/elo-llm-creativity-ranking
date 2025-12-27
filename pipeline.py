#!/usr/bin/env python
# coding: utf-8

"""
Automated Pipeline Script for LLM Fine-Tuning and Evaluation

This script automates the entire pipeline from data download to evaluation:
1. Download raw data
2. Parse and clean data
3. Download model (if needed)
4. Fine-tune the model
5. Run inference and evaluation

Usage:
    python pipeline.py

Requirements:
    - All dependencies from requirements.txt installed
    - For model download: Set Hugging Face token in huggingface_downloader.py if model not already downloaded
"""

import subprocess
import sys
import os
from pathlib import Path

def run_step(step_name, script_name, cwd=None):
    """Run a pipeline step and report progress."""
    print(f"\n{'='*50}")
    print(f"STARTING: {step_name}")
    print(f"{'='*50}")

    # Use the current Python executable to ensure compatibility
    command = [sys.executable, script_name]

    try:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"\n{'='*50}")
            print(f"COMPLETED: {step_name}")
            print(f"{'='*50}")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
        else:
            print(f"\n{'='*50}")
            print(f"FAILED: {step_name}")
            print(f"{'='*50}")
            print("Error output:")
            print(result.stderr)
            print(f"Return code: {result.returncode}")
            sys.exit(1)
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR: {step_name}")
        print(f"{'='*50}")
        print(f"Exception: {e}")
        sys.exit(1)

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"✓ Found {description}: {filepath}")
        return True
    else:
        print(f"✗ Missing {description}: {filepath}")
        return False

def main():
    """Main pipeline function."""
    print("LLM Fine-Tuning Pipeline Automation")
    print("===================================")

    # Get the current directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Install CUDA PyTorch first
    print("Installing CUDA PyTorch...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                               "--index-url", "https://download.pytorch.org/whl/cu121"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Warning: Failed to install CUDA PyTorch. GPU acceleration may not work.")
            print("Error:", result.stderr)
        else:
            print("✓ CUDA PyTorch installed successfully.")
    except Exception as e:
        print(f"Warning: Could not install CUDA PyTorch: {e}")

    # Verify CUDA is working
    print("Verifying CUDA installation...")
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA is available and working!")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA is not available. Running troubleshooting...")
            # Run the CUDA troubleshooting script
            troubleshoot_result = subprocess.run([sys.executable, "cuda_troubleshoot.py"], 
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Troubleshooting output:")
            print(troubleshoot_result.stdout)
            if troubleshoot_result.stderr:
                print("Troubleshooting errors:")
                print(troubleshoot_result.stderr)
            
            # Check again after troubleshooting
            import importlib
            importlib.reload(torch)
            if torch.cuda.is_available():
                print("✓ CUDA is now working after troubleshooting!")
            else:
                print("⚠ CUDA troubleshooting completed but still not available.")
                print("The pipeline will continue but GPU acceleration will not be used.")
    except ImportError:
        print("⚠ PyTorch not available for CUDA check. Continuing without GPU verification.")

    # Install requirements
    if os.path.exists("requirements.txt"):
        print("Installing Python dependencies...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print("Warning: Failed to install requirements. Please install manually.")
                print("Error:", result.stderr)
            else:
                print("✓ Dependencies installed successfully.")
        except Exception as e:
            print(f"Warning: Could not install requirements: {e}")
    else:
        print("Warning: requirements.txt not found. Please ensure dependencies are installed.")

    # Install tf-keras for compatibility with transformers
    print("Installing tf-keras and python-dotenv for environment handling...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "tf-keras", "python-dotenv"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Warning: Failed to install tf-keras and python-dotenv. Some features may fail.")
            print("Error:", result.stderr)
        else:
            print("✓ tf-keras and python-dotenv installed successfully.")
    except Exception as e:
        print(f"Warning: Could not install tf-keras and python-dotenv: {e}")

    # Check if all required scripts exist
    required_scripts = [
        "download_raw_data.py",
        "parse_data.py",
        "huggingface_downloader.py",
        "fine_tuning.py",
        "inference.py"
    ]

    missing_scripts = []
    for script in required_scripts:
        if not (script_dir / script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        print(f"ERROR: Missing required scripts: {', '.join(missing_scripts)}")
        print(f"Please ensure all scripts are in the same directory as pipeline.py: {script_dir}")
        sys.exit(1)

    print("✓ All required scripts found.")

    # Step 1: Download raw data
    if os.path.exists("all_sctt_jrt.csv"):
        print("✓ Raw data already exists. Skipping download step.")
        print("  Found: all_sctt_jrt.csv")
    else:
        run_step("Download Raw Data", "download_raw_data.py")

    # Check if data was downloaded
    if not check_file_exists("all_sctt_jrt.csv", "raw data file"):
        print("ERROR: Raw data download failed - all_sctt_jrt.csv not found.")
        sys.exit(1)

    # Step 2: Parse and clean data
    # Check if parsing has already been done
    parse_outputs_exist = (
        os.path.exists("pairs_dataset.csv") and
        os.path.exists("split_assignments.csv") and
        os.path.exists("grouped_data")
    )

    if parse_outputs_exist:
        print("✓ Data parsing outputs already exist. Skipping parse/clean step.")
        print("  Found: pairs_dataset.csv, split_assignments.csv, grouped_data/")
    else:
        run_step("Parse and Clean Data", "parse_data.py")

    # Verify parsing outputs exist after running
    if not (os.path.exists("pairs_dataset.csv") and os.path.exists("split_assignments.csv")):
        print("ERROR: Data parsing failed - required output files not found.")
        sys.exit(1)

    # Step 3: Download model (optional, skip if already exists)
    model_dir = script_dir / "downloaded_models" / "llama-2-7b"
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"✓ Model already exists: {model_dir}")
    else:
        print("\nModel not found locally. Attempting to download from Hugging Face...")
        print("Note: Make sure to set your Hugging Face token in huggingface_downloader.py")
        run_step("Download Model", "huggingface_downloader.py")
        # Verify model was downloaded
        if not (model_dir.exists() and any(model_dir.iterdir())):
            print("ERROR: Model download failed - model directory not found or empty.")
            sys.exit(1)

    # Step 4: Fine-tune the model
    if os.path.exists("fine_tuned_model"):
        print("✓ Fine-tuned model already exists. Skipping fine-tuning step.")
        print("  Found: fine_tuned_model/")
        print("  Note: Delete the directory if you want to retrain the model.")
    else:
        run_step("Fine-Tune Model", "fine_tuning.py")

    # Check if fine-tuned model exists
    if not check_file_exists("fine_tuned_model", "fine-tuned model directory"):
        print("ERROR: Model fine-tuning failed - fine_tuned_model directory not found.")
        sys.exit(1)

    # Step 5: Run inference and evaluation
    run_step("Run Inference and Evaluation", "inference.py")

    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Check the output above for results and metrics.")

if __name__ == "__main__":
    main()