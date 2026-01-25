#!/usr/bin/env python
# coding: utf-8

"""
Automated Pipeline Script for LLM Fine-Tuning and Evaluation

This script automates the entire pipeline from data download to evaluation:
1. Download raw data
2. Parse and clean data
3. Download model (if needed)
4. Fine-tune the model with curriculum learning
5. Compute epoch-wise results
6. Calculate metrics
7. Run inference

Usage:
    python pipeline.py

Requirements:
    - All dependencies from requirements.txt installed
    - CUDA-enabled GPU for training
    - Hugging Face token for model access
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple


class PipelineRunner:
    """Manages the complete LLM fine-tuning and evaluation pipeline."""
    
    def __init__(self, working_directory: Optional[Path] = None):
        """
        Initialize pipeline runner.
        
        Args:
            working_directory: Base directory for pipeline operations (defaults to script directory)
        """
        self.script_dir = working_directory or Path(__file__).parent
        os.chdir(self.script_dir)
        
        self.required_scripts = [
            "download_raw_data.py",
            "parse_data.py",
            "huggingface_downloader.py",
            "fine_tuning_curriculum.py",
            "inference.py",
            "compute_epoch_wise_results.py",
            "calculate_metrics.py"
        ]
    
    def run_step(self, step_name: str, script_name: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """
        Execute a pipeline step and report progress.
        
        Args:
            step_name: Human-readable name of the step
            script_name: Python script filename to execute
            cwd: Working directory for script execution
        
        Returns:
            CompletedProcess object with execution results
        
        Raises:
            SystemExit: If script execution fails
        """
        command = [sys.executable, script_name]
        
        try:
            result = subprocess.run(
                command, 
                cwd=cwd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            if result.returncode == 0:
                return result
            else:
                print(f"ERROR: {step_name} failed")
                print(result.stderr)
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR: {step_name} - {e}")
            sys.exit(1)
    
    def check_file_exists(self, filepath: str, description: str) -> bool:
        """
        Verify that a required file exists.
        
        Args:
            filepath: Path to file to check
            description: Human-readable description of file
        
        Returns:
            True if file exists, False otherwise
        """
        if os.path.exists(filepath):
            return True
        else:
            print(f"ERROR: Missing {description}")
            return False
    
    def setup_cuda_environment(self) -> None:
        """Install and verify CUDA PyTorch installation."""
        if os.path.exists("cuda_install.py"):
            try:
                result = subprocess.run(
                    [sys.executable, "cuda_install.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    print("WARNING: CUDA installation script encountered issues")
                    print(result.stderr)
                else:
                    print("SUCCESS: CUDA installation completed")
            except Exception as e:
                print(f"WARNING: Could not run CUDA installation script: {e}")
        else:
            print("WARNING: cuda_install.py not found")
        
        self._verify_cuda()
    
    def _verify_cuda(self) -> None:
        """Verify CUDA availability and run troubleshooting if needed."""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"SUCCESS: CUDA available - {torch.cuda.get_device_name(0)}")
            else:
                print("WARNING: CUDA not available, running troubleshooting")
                self._run_cuda_troubleshooting()
        except ImportError:
            print("WARNING: PyTorch not available for CUDA check")
    
    def _run_cuda_troubleshooting(self) -> None:
        """Execute CUDA troubleshooting script."""
        subprocess.run(
            [sys.executable, "cuda_troubleshoot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        import importlib
        import torch
        importlib.reload(torch)
        if torch.cuda.is_available():
            print("SUCCESS: CUDA working after troubleshooting")
        else:
            print("WARNING: CUDA troubleshooting failed, GPU acceleration unavailable")
    
    def install_dependencies(self) -> None:
        """Install required Python packages."""
        if os.path.exists("requirements.txt"):
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    print("WARNING: Failed to install requirements")
                    print(result.stderr)
                else:
                    print("SUCCESS: Dependencies installed")
            except Exception as e:
                print(f"WARNING: Could not install requirements: {e}")
        else:
            print("WARNING: requirements.txt not found")
        
        # Install additional packages
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "tf-keras", "python-dotenv"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                print("WARNING: Failed to install tf-keras and python-dotenv")
            else:
                print("SUCCESS: Additional packages installed")
        except Exception as e:
            print(f"WARNING: Could not install additional packages: {e}")
    
    def validate_required_scripts(self) -> None:
        """Verify all required scripts are present."""
        missing_scripts = []
        for script in self.required_scripts:
            if not (self.script_dir / script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"ERROR: Missing required scripts: {', '.join(missing_scripts)}")
            sys.exit(1)
        
        print("SUCCESS: All required scripts found")
    
    def download_raw_data(self) -> None:
        """Download or verify raw data existence."""
        if os.path.exists("all_sctt_jrt.csv"):
            print("SUCCESS: Raw data found")
        else:
            self.run_step("Download Raw Data", "download_raw_data.py")
        
        if not self.check_file_exists("all_sctt_jrt.csv", "raw data file"):
            sys.exit(1)
    
    def parse_and_clean_data(self) -> None:
        """Parse and clean data into train/val/test splits."""
        parse_outputs_exist = (
            os.path.exists("pairs_train.csv") and
            os.path.exists("pairs_val.csv") and
            os.path.exists("pairs_test.csv")
        )
        
        if parse_outputs_exist:
            print("SUCCESS: Data parsing outputs found")
        else:
            self.run_step("Parse and Clean Data", "parse_data.py")
        
        if not (os.path.exists("pairs_train.csv") and 
                os.path.exists("pairs_val.csv") and 
                os.path.exists("pairs_test.csv")):
            print("ERROR: Data parsing failed")
            sys.exit(1)
    
    def download_model(self) -> None:
        """Download or verify model existence."""
        model_dir = self.script_dir / "downloaded_models" / "llama-2-7b"
        if model_dir.exists() and any(model_dir.iterdir()):
            print(f"SUCCESS: Model found")
        else:
            self.run_step("Download Model", "huggingface_downloader.py")
            
            if not (model_dir.exists() and any(model_dir.iterdir())):
                print("ERROR: Model download failed")
                sys.exit(1)
    
    def fine_tune_model(self) -> None:
        """Fine-tune model with curriculum learning."""
        if os.path.exists("fine_tuned_model_curriculum"):
            print("SUCCESS: Fine-tuned model found (skipping training)")
        else:
            self.run_step("Fine-Tune Model", "fine_tuning_curriculum.py")
        
        if not self.check_file_exists("fine_tuned_model_curriculum", "fine-tuned model"):
            sys.exit(1)
    
    def compute_epoch_results(self) -> None:
        """Compute epoch-wise performance to identify best checkpoint."""
        self.run_step("Compute Epoch-Wise Results", "compute_epoch_wise_results.py")
        
        if not self.check_file_exists("epoch_wise_LORA_results.csv", "epoch-wise results file"):
            print("WARNING: Epoch-wise results not found. Continuing anyway.")
    
    def calculate_metrics(self) -> None:
        """Calculate final evaluation metrics."""
        self.run_step("Calculate Metrics", "calculate_metrics.py")
    
    def run_inference(self) -> None:
        """Run inference if script is available."""
        if os.path.exists("inference.py"):
            try:
                self.run_step("Run Inference", "inference.py")
            except:
                print("WARNING: Inference step failed (optional)")
    
    def print_summary(self) -> None:
        """Print pipeline completion summary."""
        print("\nSUCCESS: Pipeline completed")
        print("Results:")
        print("  - epoch_wise_LORA_results.csv")
        print("  - sctt_results_* directories")
    
    def run(self) -> None:
        """Execute the complete pipeline."""
        self.setup_cuda_environment()
        self.install_dependencies()
        self.validate_required_scripts()
        self.download_raw_data()
        self.parse_and_clean_data()
        self.download_model()
        self.fine_tune_model()
        self.compute_epoch_results()
        self.calculate_metrics()
        self.run_inference()
        self.print_summary()


if __name__ == "__main__":
    pipeline = PipelineRunner()
    pipeline.run()
