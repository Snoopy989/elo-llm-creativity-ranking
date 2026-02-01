"""
Upload all checkpoints from phase_3 to Hugging Face Hub
Each checkpoint will be uploaded as a separate revision/branch
"""
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, create_repo


class Logger:
    """Professional logging utility for upload operations"""
    
    @staticmethod
    def success(message):
        print(f"[SUCCESS] {message}")
    
    @staticmethod
    def error(message):
        print(f"[ERROR] {message}")
    
    @staticmethod
    def warning(message):
        print(f"[WARNING] {message}")
    
    @staticmethod
    def info(message):
        print(f"[INFO] {message}")


# Configuration
REPO_NAME = "PhillipGre/llama2-creativity-phase3"
CHECKPOINT_DIR = Path("sctt_results_curriculum_10_epochs_Llama-2-7b-hf/phase_3")

def create_repository():
    """Create the HuggingFace repository if it doesn't exist"""
    try:
        api = HfApi()
        create_repo(REPO_NAME, repo_type="model", exist_ok=True)
        Logger.success(f"Repository {REPO_NAME} is ready")
        return True
    except Exception as e:
        Logger.error(f"Failed to create repository: {e}")
        return False

def upload_checkpoints():
    """Upload all checkpoint directories to HuggingFace"""
    
    # Get all checkpoint directories
    checkpoints = sorted([d for d in CHECKPOINT_DIR.iterdir() 
                         if d.is_dir() and d.name.startswith("checkpoint-")])
    
    Logger.info(f"Found {len(checkpoints)} checkpoints to upload")
    
    for i, checkpoint_path in enumerate(checkpoints, 1):
        checkpoint_name = checkpoint_path.name
        revision_name = checkpoint_name  # Use checkpoint name as revision
        
        print(f"\n[{i}/{len(checkpoints)}] Uploading {checkpoint_name} as revision '{revision_name}'...")
        
        try:
            # Upload command
            cmd = [
                "huggingface-cli",
                "upload",
                REPO_NAME,
                str(checkpoint_path),
                ".",  # Upload to root of repo
                "--revision", revision_name
            ]
            
            # Run without capturing output so you can see live progress
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                Logger.success(f"Successfully uploaded {checkpoint_name}")
            else:
                Logger.error(f"Failed to upload {checkpoint_name}")
                
        except KeyboardInterrupt:
            Logger.warning("Upload interrupted by user. Stopping...")
            break
        except Exception as e:
            Logger.error(f"Error uploading {checkpoint_name}: {e}")
    
    print("\n" + "="*60)
    Logger.info("Upload complete! All checkpoints are at:")
    print(f"https://huggingface.co/{REPO_NAME}")
    print("\nTo download a specific checkpoint on Lambda:")
    print(f"huggingface-cli download {REPO_NAME} --revision checkpoint-XXXX --local-dir ./model")

if __name__ == "__main__":
    print("HuggingFace Checkpoint Uploader")
    print("="*60)
    print(f"Repository: {REPO_NAME}")
    print(f"Source directory: {CHECKPOINT_DIR}")
    print("\nMake sure you've logged in with: huggingface-cli login")
    print("="*60)
    
    response = input("\nProceed with upload? (y/n): ")
    if response.lower() == 'y':
        # Create repository first
        if not create_repository():
            Logger.error("Failed to create repository. Exiting.")
            exit(1)
        
        Logger.info("Starting checkpoint uploads...")
        print()
        upload_checkpoints()
    else:
        Logger.info("Upload cancelled")
