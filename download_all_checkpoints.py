#!/usr/bin/env python3
"""
Download All Checkpoints from Hugging Face Repository
=====================================================

This script downloads all checkpoint branches from the Hugging Face model repository.
Each checkpoint is stored as a separate branch.
"""

import os
import subprocess
import re
from pathlib import Path

# Configuration
REPO_URL = "https://huggingface.co/PhillipGre/llama2-7b_sctt_classification_10epochs_phase3_curriculum"
REPO_NAME = "llama2-7b_sctt_classification_10epochs_phase3_curriculum"
CHECKPOINTS_DIR = "checkpoints"

def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error: {result.stderr}")
    return result

def get_checkpoint_branches(repo_path):
    """Get all checkpoint branches from the repository."""
    result = run_command("git branch -r", cwd=repo_path)
    branches = []
    
    for line in result.stdout.split('\n'):
        # Match checkpoint branches like "origin/checkpoint-1000"
        match = re.search(r'origin/(checkpoint-\d+)', line)
        if match:
            branches.append(match.group(1))
    
    # Sort by checkpoint number
    branches.sort(key=lambda x: int(x.split('-')[1]))
    return branches

def download_checkpoint(repo_path, branch_name, output_dir):
    """Download a specific checkpoint branch."""
    checkpoint_dir = output_dir / branch_name
    
    # Skip if already exists
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        print(f"✓ {branch_name} already exists, skipping...")
        return True
    
    print(f"\n📦 Downloading {branch_name}...")
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkout the branch
    print(f"  Checking out {branch_name}...")
    result = run_command(f"git checkout {branch_name}", cwd=repo_path)
    if result.returncode != 0:
        print(f"  ✗ Failed to checkout {branch_name}")
        return False
    
    # Pull LFS files
    print(f"  Pulling LFS files for {branch_name}...")
    result = run_command("git lfs pull", cwd=repo_path)
    if result.returncode != 0:
        print(f"  ✗ Failed to pull LFS files for {branch_name}")
        return False
    
    # Copy files to checkpoint directory (excluding .git)
    print(f"  Copying files to {checkpoint_dir}...")
    result = run_command(
        f"rsync -av --exclude='.git' {repo_path}/ {checkpoint_dir}/",
        cwd=None
    )
    
    if result.returncode != 0:
        print(f"  ✗ Failed to copy files for {branch_name}")
        return False
    
    print(f"  ✓ {branch_name} downloaded successfully")
    return True

def main():
    """Main function to download all checkpoints."""
    print("=" * 60)
    print("Downloading All Checkpoints")
    print("=" * 60)
    
    repo_path = Path(REPO_NAME)
    output_dir = Path(CHECKPOINTS_DIR)
    
    # Check if repository exists
    if not repo_path.exists():
        print(f"Error: Repository '{REPO_NAME}' not found")
        print(f"Please clone it first with:")
        print(f"  git clone {REPO_URL}")
        return
    
    # Initialize git-lfs
    print("\n🔧 Initializing git-lfs...")
    run_command("git lfs install")
    run_command("git lfs install", cwd=repo_path)
    
    # Get all checkpoint branches
    print("\n🔍 Finding checkpoint branches...")
    branches = get_checkpoint_branches(repo_path)
    
    if not branches:
        print("No checkpoint branches found!")
        return
    
    print(f"Found {len(branches)} checkpoints:")
    print(f"  First: {branches[0]}")
    print(f"  Last: {branches[-1]}")
    
    # Download each checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, branch in enumerate(branches, 1):
        print(f"\n[{i}/{len(branches)}] Processing {branch}...")
        
        if download_checkpoint(repo_path, branch, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total checkpoints: {len(branches)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"\nCheckpoints saved to: {output_dir.absolute()}")
    
    # Return to main branch
    print("\n🔄 Returning to main branch...")
    run_command("git checkout main", cwd=repo_path)

if __name__ == "__main__":
    main()
