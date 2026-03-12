#!/usr/bin/env python
# coding: utf-8
"""
Upload the fine-tuned LoRA adapter + tokenizer to HuggingFace Hub.

Combines:
  - Adapter weights from: sctt_results_curriculum_10_epochs_Llama-2-13b-hf/phase_3/checkpoint-540000
  - Tokenizer files from: fine_tuned_model_curriculum

Usage:
    python upload_model_to_hf.py --repo-name username/llama2-13b-sctt-classification
    python upload_model_to_hf.py --repo-name username/llama2-13b-sctt-classification --private
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login, whoami


ADAPTER_PATH = "sctt_results_curriculum_10_epochs_Llama-2-13b-hf/phase_3/checkpoint-540000"
TOKENIZER_PATH = "fine_tuned_model_curriculum"

# Files to include from the adapter checkpoint (inference-relevant only)
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_args.bin",
]

# Files to include from the tokenizer directory
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

MODEL_CARD = """---
language: en
license: llama2
base_model: meta-llama/Llama-2-13b-hf
tags:
  - peft
  - lora
  - text-classification
  - creativity-ranking
  - elo
---

# LLaMA-2 13B SCTT Creativity Classifier

A LoRA fine-tuned LLaMA-2 13B model for **pairwise creativity ranking** using sequence classification.

## Model Details

- **Base model:** `meta-llama/Llama-2-13b-hf`
- **Fine-tuning method:** LoRA (PEFT) via curriculum learning (SCTT — 3 phases, 10 epochs)
- **Task:** 3-class pairwise classification — predict which of two responses (A / B / Equal) is more creative
- **Labels:** `0=A`, `1=B`, `2=Equal`
- **Checkpoint:** Phase 3, step 540,000

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("username/llama2-13b-sctt-classification")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    num_labels=3,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base_model, "username/llama2-13b-sctt-classification")
model = model.merge_and_unload()
model.eval()

prompt = "experiment: testing bird's understanding of human speech\\nA: response one\\nB: response two"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=180)
with torch.no_grad():
    logits = model(**inputs).logits
label_map = {0: "A", 1: "B", 2: "Equal"}
print(label_map[logits.argmax().item()])
```
"""


def stage_files(adapter_path: str, tokenizer_path: str, staging_dir: Path, repo_name: str):
    """Copy inference-relevant files into a staging directory."""
    missing = []

    for fname in ADAPTER_FILES:
        src = Path(adapter_path) / fname
        if src.exists():
            shutil.copy2(src, staging_dir / fname)
        else:
            missing.append(str(src))

    for fname in TOKENIZER_FILES:
        src = Path(tokenizer_path) / fname
        if src.exists():
            shutil.copy2(src, staging_dir / fname)
        else:
            missing.append(str(src))

    if missing:
        print(f"WARNING: The following files were not found and will be skipped:")
        for f in missing:
            print(f"  {f}")

    # Write model card
    card_path = staging_dir / "README.md"
    card_path.write_text(MODEL_CARD.replace("username/llama2-13b-sctt-classification", repo_name))

    staged = list(staging_dir.iterdir())
    print(f"\nStaged {len(staged)} files:")
    for f in sorted(staged):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:>10.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to HuggingFace Hub")
    parser.add_argument("--repo-name", type=str, default=None,
                        help="HuggingFace repo name (e.g. username/llama2-13b-sctt-classification)")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH)
    parser.add_argument("--tokenizer-path", type=str, default=TOKENIZER_PATH)
    args = parser.parse_args()

    # Auth
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("No HuggingFace token found. Set HUGGINGFACE_TOKEN or HF_TOKEN in .env")
    login(token=hf_token, add_to_git_credential=False)

    user_info = whoami()
    username = user_info["name"]
    print(f"Logged in as: {username}")

    # Resolve repo name
    if args.repo_name is None:
        repo_name = f"{username}/llama2-13b-sctt-classification"
    elif "/" not in args.repo_name:
        repo_name = f"{username}/{args.repo_name}"
    else:
        repo_name = args.repo_name
    print(f"Target repo: {repo_name} ({'private' if args.private else 'public'})")

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_name, private=args.private, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{repo_name}")

    # Stage files
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir)
        stage_files(args.adapter_path, args.tokenizer_path, staging_dir, repo_name)

        print(f"\nUploading to {repo_name}...")
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload LoRA adapter + tokenizer (phase 3, checkpoint-540000)",
        )

    print(f"\nDone! Model available at: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    main()
