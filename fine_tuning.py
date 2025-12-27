#!/usr/bin/env python
# coding: utf-8

import os
import sys
import platform
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict, Dataset
import pandas as pd
import inspect

def setup_environment():
    """Set up the environment for fine-tuning."""
    print(f"Working dir: {Path.cwd()}")
    print(f"Python: {sys.executable}")

    # Check CUDA availability - import torch if not already imported
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. This script requires a CUDA-compatible GPU.")
            print("Please install PyTorch with CUDA support:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            sys.exit(1)

        print("CUDA is available")
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("KERAS_BACKEND", "torch")

    if platform.system() == "Windows":
        os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
        os.environ.setdefault("DISABLE_BITSANDBYTES_WARN", "1")

    try:
        import torch, transformers, datasets, peft
        print(f"torch={torch.__version__} transformers={transformers.__version__} datasets={datasets.__version__} peft={peft.__version__}")
    except Exception as e:
        print("Note: Some packages may be missing; install from requirements.txt if needed.")

    dataset_file = Path("pairs_dataset.csv")
    status = "found" if dataset_file.exists() else "missing"
    print(f"Dataset {status}: {dataset_file.resolve()}")

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    # Import torch here to ensure it's available
    import torch

    # Require CUDA - exit if not available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a CUDA-compatible GPU.")
        print("Please install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    local_model = script_dir / "downloaded_models" / "llama-2-7b"
    base_model = str(local_model) if local_model.exists() else "meta-llama/Llama-2-7b-hf"
    print(f"Base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, local_files_only=local_model.exists()
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3,
        local_files_only=local_model.exists(), torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Resize token embeddings with error handling
    try:
        print("Resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer))
        print("Token embeddings resized successfully.")
    except KeyboardInterrupt:
        print("Token embedding resizing interrupted. Using model without resizing.")
        print("Note: This may affect performance but allows training to continue.")
    except Exception as e:
        print(f"Warning: Token embedding resizing failed: {e}")
        print("Continuing without resizing. Performance may be affected.")

    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "v_proj", "score"]
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, tokenizer

def prepare_dataset(tokenizer):
    """Prepare the dataset for training."""
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    dataset_paths = [
        script_dir / "pairs_dataset.csv",
        Path("pairs_dataset.csv"),
        Path("./pairs_dataset.csv"),
    ]

    dataset_file = None
    for path in dataset_paths:
        if path.exists():
            dataset_file = path
            print(f"Found dataset at: {dataset_file}")
            break

    if dataset_file is None:
        raise FileNotFoundError("pairs_dataset.csv not found")

    df = pd.read_csv(dataset_file)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def preprocess_function(examples):
        texts = []
        labels = []
        label_map = {"A": 0, "B": 1, "Equal": 2}

        for i in range(len(examples['response1'])):
            text = (
                f"Task: {examples['task'][i]}\n"
                f"Description: {examples['prompt'][i]}\n\n"
                f"Response A: {examples['response1'][i]}\n"
                f"Response B: {examples['response2'][i]}\n\n"
                "Which response is more scientifically possible and provable with a hypothesis?"
            )
            texts.append(text)
            winner = examples['winner'][i]
            labels.append(label_map[winner])

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt" if len(texts) == 1 else None
        )
        tokenized["labels"] = labels
        return tokenized

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    for split in dataset:
        dataset[split] = dataset[split].map(preprocess_function, batched=True, remove_columns=dataset[split].column_names)

    print("\n=== DATASET PREPARATION COMPLETE ===")
    print(f"train_dataset: {len(dataset['train'])} samples")
    print(f"test_dataset: {len(dataset['test'])} samples")

    return dataset

def create_training_args():
    """Create training arguments."""
    use_cuda = torch.cuda.is_available()
    bf16_supported = use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    args_kwargs = {
        "output_dir": "./results",
        "num_train_epochs": 5,
        # "max_steps": 150000, # To use max_steps instead of epochs, uncomment this line
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "per_device_eval_batch_size": 64,
        "fp16": False,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "logging_steps": 500,
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True,
    }

    if "bf16" in sig_params:
        args_kwargs["bf16"] = bool(bf16_supported)

    if "dataloader_num_workers" in sig_params:
        args_kwargs["dataloader_num_workers"] = 8

    if "dispatch_batches" in sig_params:
        args_kwargs["dispatch_batches"] = False

    if "save_strategy" in sig_params:
        args_kwargs["save_strategy"] = "steps"
    if "save_steps" in sig_params:
        args_kwargs["save_steps"] = 10000
    if "save_total_limit" in sig_params:
        args_kwargs["save_total_limit"] = 5

    if "load_best_model_at_end" in sig_params:
        args_kwargs["load_best_model_at_end"] = False
    if "metric_for_best_model" in sig_params:
        args_kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in sig_params:
        args_kwargs["greater_is_better"] = False

    if "report_to" in sig_params:
        args_kwargs["report_to"] = []

    filtered_args = {k: v for k, v in args_kwargs.items() if k in sig_params and k != 'dispatch_batches'}

    return TrainingArguments(**filtered_args)

def train_model(model, tokenizer, dataset, training_args):
    """Train the model."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    output_dir = Path("./results/lora_adapter")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved LoRA adapter to {output_dir}")
    except Exception as e:
        print("Warning saving adapter:", e)

    merged_dir = Path("./fine_tuned_model")
    try:
        from peft import merge_and_unload
        merged = merge_and_unload(model)
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model to {merged_dir}")
    except Exception as e:
        print("Could not merge adapters; saving current model state instead:", e)
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved model (unmerged) to {merged_dir}")

def evaluate_model(model, tokenizer, dataset):
    """Evaluate the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eval_args_kwargs = {
        "output_dir": "./results",
        "per_device_eval_batch_size": 32,
        "fp16": False,
    }
    if hasattr(TrainingArguments, "bf16"):
        eval_args_kwargs["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if hasattr(TrainingArguments, "remove_unused_columns"):
        eval_args_kwargs["remove_unused_columns"] = True
    if hasattr(TrainingArguments, "dataloader_num_workers"):
        eval_args_kwargs["dataloader_num_workers"] = 8
    if hasattr(TrainingArguments, "dataloader_pin_memory"):
        eval_args_kwargs["dataloader_pin_memory"] = True
    if hasattr(TrainingArguments, "report_to"):
        eval_args_kwargs["report_to"] = []
    if hasattr(TrainingArguments, "logging_strategy"):
        eval_args_kwargs["logging_strategy"] = "no"

    eval_trainer = Trainer(
        model=model,
        args=TrainingArguments(**eval_args_kwargs),
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    print("Starting evaluation...")
    eval_results = eval_trainer.evaluate()
    print("Evaluation results:")
    print(eval_results)

def main():
    """Main function to run the fine-tuning pipeline."""

    setup_environment()
    model, tokenizer = load_model_and_tokenizer()
    dataset = prepare_dataset(tokenizer)
    training_args = create_training_args()
    train_model(model, tokenizer, dataset, training_args)
    evaluate_model(model, tokenizer, dataset)

if __name__ == "__main__":
    main()

