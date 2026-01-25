"""
Fine-Tuning with Curriculum Learning

Integrates curriculum learning into the training pipeline:
- Epochs 1-2: Easy examples (large score differences)
- Epochs 3-4: Easy + Medium examples  
- Epochs 5+: All examples including hard cases

This improves convergence and final accuracy.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Tuple
from scipy.stats import pearsonr

from dotenv import load_dotenv
from huggingface_hub import login



# Add at the top of the file after imports
load_dotenv()

# Login to HuggingFace
hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("✓ Logged in to Hugging Face")
else:
    print("⚠ Warning: No HF_TOKEN found in .env file")




class CurriculumDatasetBuilder:
    """Build curriculum-based datasets from pre-split pairs."""
    
    def __init__(self, tokenizer, max_length: int = 180):
        """
        Initialize dataset builder.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"A": 0, "B": 1, "Equal": 2}
    
    def load_datasets(self, curriculum_phase: Optional[int] = None) -> DatasetDict:
        """
        Load pre-split datasets with optional curriculum filtering.
        
        Args:
            curriculum_phase: 1 (easy only), 2 (easy+medium), 3 (all), or None (no curriculum)
        
        Returns:
            DatasetDict with train/validation/test splits
        """
        # Load pre-split files
        train_df = self._load_split_file("pairs_train.csv")
        val_df = self._load_split_file("pairs_val.csv")
        test_df = self._load_split_file("pairs_test.csv")
        
        # Verify required columns
        self._validate_columns(train_df)
        
        # Apply curriculum filtering to training set only
        if curriculum_phase is not None:
            train_df = self._apply_curriculum_filter(train_df, curriculum_phase)
        
        # Convert to HuggingFace datasets
        datasets = {
            "train": self._preprocess_dataframe(train_df),
            "validation": self._preprocess_dataframe(val_df),
            "test": self._preprocess_dataframe(test_df)
        }
        
        return DatasetDict(datasets)
    
    def _load_split_file(self, filename: str) -> pd.DataFrame:
        """Load a split CSV file."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"{filename} not found. Run parse_data.py first.")
        return pd.read_csv(filepath)
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required = ['task', 'prompt', 'response1', 'response2', 'winner', 'difficulty_level']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Run parse_data.py first.")
    
    def _apply_curriculum_filter(self, df: pd.DataFrame, phase: int) -> pd.DataFrame:
        """
        Apply cumulative curriculum filtering.
        
        Args:
            df: Training dataframe
            phase: Curriculum phase (1=easy, 2=easy+medium, 3=all)
        
        Returns:
            Filtered dataframe
        """
        original_len = len(df)
        
        if phase == 1:
            # Phase 1: Easy only
            filtered_df = df[df['difficulty_level'] == 1].copy()
        elif phase == 2:
            # Phase 2: Easy + Medium (cumulative)
            filtered_df = df[df['difficulty_level'].isin([1, 2])].copy()
        elif phase == 3:
            # Phase 3: All difficulties
            filtered_df = df.copy()
        else:
            raise ValueError(f"Invalid curriculum phase: {phase}. Must be 1, 2, or 3.")
        
        print(f"\nCurriculum Phase {phase}:")
        print(f"  Training on {len(filtered_df):,} / {original_len:,} pairs ({len(filtered_df)/original_len*100:.1f}%)")
        
        difficulty_names = {1: 'Easy only', 2: 'Easy + Medium', 3: 'All difficulties'}
        print(f"  Difficulty: {difficulty_names[phase]}")
        
        # Print difficulty distribution
        self._print_difficulty_distribution(filtered_df)
        
        return filtered_df
    
    def _print_difficulty_distribution(self, df: pd.DataFrame) -> None:
        """Print difficulty distribution statistics."""
        print(f"  Difficulty distribution:")
        level_names = {1: 'Easy', 2: 'Medium-Easy', 3: 'Medium-Hard', 4: 'Hard'}
        
        for level in sorted(df['difficulty_level'].unique()):
            count = (df['difficulty_level'] == level).sum()
            pct = count / len(df) * 100
            print(f"    {level_names.get(level, f'Level {level}')}: {count:,} ({pct:.1f}%)")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> Dataset:
        """Convert pandas DataFrame to preprocessed HuggingFace Dataset."""
        dataset = Dataset.from_pandas(df)
        return dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _preprocess_function(self, examples: Dict) -> Dict:
        """Preprocess batch of examples for model input."""
        texts = []
        labels = []
        
        for i in range(len(examples['response1'])):
            # Format: task: prompt\nA: response1\nB: response2
            text = (
                f"{examples['task'][i]}: {examples['prompt'][i]}\n"
                f"A: {examples['response1'][i]}\n"
                f"B: {examples['response2'][i]}"
            )
            texts.append(text)
            labels.append(self.label_map[examples['winner'][i]])
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt" if len(texts) == 1 else None
        )
        tokenized["labels"] = labels
        
        return tokenized


class CurriculumTrainer:
    """Handles curriculum-based training workflow."""
    
    def __init__(self, model, tokenizer, experiment_name: str, device: str = "cuda"):
        """
        Initialize curriculum trainer.
        
        Args:
            model: The model to train
            tokenizer: HuggingFace tokenizer
            experiment_name: Name for this training experiment
            device: Device to train on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.experiment_name = experiment_name
        self.device = device
        self.dataset_builder = CurriculumDatasetBuilder(tokenizer)
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics including Pearson correlation."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(predictions, labels)
        
        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        
        return {
            'pearson': pearson_corr,
            'accuracy': accuracy
        }
    
    def train(self, total_epochs: int = 10, 
              max_length: int = 180,
              batch_size: int = 32,
              gradient_accumulation_steps: int = 4) -> None:
        """
        Train model with curriculum learning.
        
        Args:
            total_epochs: Total training epochs (divisible by 3 recommended)
            max_length: Maximum sequence length
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
        """
        # Generate curriculum schedule
        schedule = self._create_curriculum_schedule(total_epochs)
        self._print_schedule(schedule)
        
        # Train each phase
        for phase_idx, curriculum_phase in enumerate(sorted(set(schedule)), 1):
            epochs_in_phase = schedule.count(curriculum_phase)
            self._train_phase(
                phase_idx, 
                curriculum_phase, 
                epochs_in_phase,
                batch_size,
                gradient_accumulation_steps
            )
    
    def _create_curriculum_schedule(self, total_epochs: int) -> list:
        """Create curriculum schedule for training."""
        schedule = []
        epochs_per_phase = total_epochs // 3
        
        # Phase 1: Easy only
        schedule.extend([1] * epochs_per_phase)
        
        # Phase 2: Easy + Medium
        schedule.extend([2] * epochs_per_phase)
        
        # Phase 3: All difficulties
        schedule.extend([3] * (total_epochs - 2 * epochs_per_phase))
        
        return schedule
    
    def _print_schedule(self, schedule: list) -> None:
        """Print curriculum training schedule."""
        print("\n" + "="*60)
        print("CURRICULUM TRAINING SCHEDULE")
        print("="*60)
        
        phase_names = {1: 'Easy only', 2: 'Easy + Medium', 3: 'All difficulties'}
        for epoch, phase in enumerate(schedule, 1):
            print(f"Epoch {epoch}: {phase_names[phase]}")
    
    def _train_phase(self, phase_idx: int, 
                     curriculum_phase: int,
                     epochs_in_phase: int,
                     batch_size: int,
                     gradient_accumulation_steps: int) -> None:
        """Train a single curriculum phase."""
        print("\n" + "="*60)
        print(f"PHASE {phase_idx}: DIFFICULTY LEVEL {curriculum_phase}")
        print(f"Training for {epochs_in_phase} epochs")
        print("="*60)
        
        # Load dataset for this phase
        dataset = self.dataset_builder.load_datasets(curriculum_phase=curriculum_phase)
        # Configure training arguments
        model_name = self.model.config.name_or_path.split('/')[-1]
        phase_output_dir = f'./sctt_results_{self.experiment_name}_{model_name}/phase_{phase_idx}'

        training_args = TrainingArguments(
            output_dir=phase_output_dir,
            num_train_epochs=epochs_in_phase,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate = 5e-5,
            warmup_steps=1000,
            # weight_decay=0.01, # Disabled weight decay to align with original settings
            logging_steps=10,
            eval_steps=1000,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16 = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print(f"\nStarting Phase {phase_idx} training...")
        trainer.train()
        print(f"✓ Phase {phase_idx} complete")


class ModelManager:
    """Manages model loading, configuration, and saving."""
    
    def __init__(self, base_model: str = "meta-llama/Llama-2-7b-hf"):
        """
        Initialize model manager.
        
        Args:
            base_model: HuggingFace model name
        """
        self.base_model = base_model
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Get available device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required for training.")
        return torch.device("cuda")
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load and configure model and tokenizer."""
        print("\nLoading model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=3,
            torch_dtype=torch.bfloat16
        )
        model.to(self.device)
        
        # Resize embeddings if needed
        try:
            model.resize_token_embeddings(len(tokenizer))
        except:
            print("Warning: Token embedding resize failed")
        
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Apply LoRA
        model = self._apply_lora(model)
        
        # Print model info
        self._print_model_info(model)
        
        return model, tokenizer
    
    def _apply_lora(self, model) -> AutoModelForSequenceClassification:
        """Apply LoRA adapters to model."""
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj", "score"]
        )
        return get_peft_model(model, lora_config)
    
    def _print_model_info(self, model) -> None:
        """Print model parameter information."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
        print(f"✓ Using device: {self.device}")
    
    def save_model(self, model, tokenizer, output_dir: str = "./fine_tuned_model_curriculum") -> None:
        """Save fine-tuned model and tokenizer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"\n✓ Training complete!")
        print(f"✓ Model saved to {output_path}")


def main():
    """Main training function with curriculum learning."""
    
    print("="*60)
    print("FINE-TUNING WITH CURRICULUM LEARNING")
    print("="*60)
    
    # Initialize model manager
    model_manager = ModelManager(base_model="meta-llama/Llama-2-7b-hf")
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Initialize curriculum trainer
    total_epochs = 1  # Test with 1 epoch
    experiment_name = f"curriculum_{total_epochs}_epochs"
    trainer = CurriculumTrainer(model, tokenizer, experiment_name=experiment_name)  # ← ADD experi # ← ADD experiment_name
    
    
    # Train with curriculum
    print("\nStarting curriculum training...")
    trainer.train(
        total_epochs=10,
        max_length=180,
        batch_size=1,
        gradient_accumulation_steps=1
    )
    
    # Save final model
    model_manager.save_model(model, tokenizer)
    



if __name__ == "__main__":
    main()