"""
Upload the best checkpoint to Hugging Face Hub
Based on checkpoint evaluation results, upload checkpoint-32000 (best validation accuracy)
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
REPO_NAME = "PhillipGre/llama2-7b-sctt-classification1"
CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT = "checkpoint-32000"  # Best validation accuracy: 73.00%

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
    """Upload the best checkpoint to HuggingFace"""
    
    checkpoint_path = CHECKPOINT_DIR / BEST_CHECKPOINT
    
    if not checkpoint_path.exists():
        Logger.error(f"Checkpoint directory not found: {checkpoint_path}")
        return False
    
    Logger.info(f"Uploading best checkpoint: {BEST_CHECKPOINT}")
    Logger.info(f"Validation accuracy: 73.00%, Pearson: 0.434")
    
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=REPO_NAME,
            repo_type="model",
        )
        
        Logger.success(f"Successfully uploaded {BEST_CHECKPOINT}")
        return True
            
    except KeyboardInterrupt:
        Logger.warning("Upload interrupted by user")
        return False
    except Exception as e:
        Logger.error(f"Error uploading {BEST_CHECKPOINT}: {e}")
        return False

if __name__ == "__main__":
    print("HuggingFace Best Checkpoint Uploader")
    print("="*60)
    print(f"Repository: {REPO_NAME}")
    print(f"Checkpoint: {BEST_CHECKPOINT}")
    print(f"Source: {CHECKPOINT_DIR / BEST_CHECKPOINT}")
    print("="*60)
    
    # Login to HuggingFace using token from .env
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        Logger.error("HUGGINGFACE_TOKEN not found in .env file")
        exit(1)
    
    try:
        login(token=token)
        Logger.success("Logged in to HuggingFace")
    except Exception as e:
        Logger.error(f"Failed to login: {e}")
        exit(1)
    
    response = input("\nProceed with upload? (y/n): ")
    if response.lower() == 'y':
        # Create repository first
        if not create_repository():
            Logger.error("Failed to create repository. Exiting.")
            exit(1)
        
        Logger.info("Starting upload...")
        print()
        if upload_checkpoints():
            print("\n" + "="*60)
            Logger.success("Upload complete!")
            print(f"Model available at: https://huggingface.co/{REPO_NAME}")
            print("\nTo use this model:")
            print(f"  from peft import PeftModel, PeftConfig")
            print(f"  config = PeftConfig.from_pretrained('{REPO_NAME}')")
            print(f"  model = PeftModel.from_pretrained(base_model, '{REPO_NAME}')")
        else:
            Logger.error("Upload failed")
            exit(1)
    else:
        Logger.info("Upload cancelled")
