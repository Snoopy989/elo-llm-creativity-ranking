import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import PeftConfig, PeftModel
from scipy.stats import pearsonr

#  SETTINGS
np.random.seed(42)
torch.cuda.empty_cache()
model_name = 'meta-llama/Llama-2-7b-hf'
checkpoints_dirs = ['sctt_results_curriculum_10_epochs_Llama-2-7b-hf/phase_3']
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
max_length = 180
label_map = {"A": 0, "B": 1, "Equal": 2}

# Compute metrics function
def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    
    # Pearson correlation
    try:
        pearson_corr, _ = pearsonr(predictions, labels)
    except:
        pearson_corr = 0.0
    
    return {
        'pearson': pearson_corr,
        'accuracy': accuracy
    }

test_args = TrainingArguments(
  do_train = False,
  do_predict = True,
  per_device_eval_batch_size=32,
  dataloader_num_workers=0,  # Set to 0 for Windows compatibility
  dataloader_pin_memory=True,
  bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
  fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
  output_dir='./checkpoint_evaluation_results',
)

#  LOAD DATA
val_df = pd.read_csv('pairs_val.csv')
test_df = pd.read_csv('pairs_test.csv')


# STORAGE LIST
datadict = []

# LOOP THRU CHECKPOINTS
for model_type in checkpoints_dirs:
  checkpoints = os.listdir(model_type)
  
  # Filter checkpoints in range 30000-40000
  filtered_checkpoints = []
  for checkpoint in checkpoints:
    if checkpoint.startswith('checkpoint-'):
      try:
        step_num = int(checkpoint.split('-')[1])
        if 30000 <= step_num <= 40000:
          filtered_checkpoints.append(checkpoint)
      except (ValueError, IndexError):
        pass  # Skip if not a valid checkpoint format
  
  # Sort checkpoints by step number
  filtered_checkpoints.sort(key=lambda x: int(x.split('-')[1]))
  
  print(f"Found {len(filtered_checkpoints)} checkpoints in range 30000-40000")
  print(f"Checkpoints to evaluate: {filtered_checkpoints}\n")
  
  # LOOP THRU CHECKPOINTS WITHIN MODEL TYPE
  for ind, checkpoint in enumerate(filtered_checkpoints):
    print(f"\nEvaluating checkpoint: {checkpoint}")
    peft_model_id = '{}/{}'.format(model_type, checkpoint)
    config = PeftConfig.from_pretrained(peft_model_id)
    
    # Load tokenizer FIRST (matching training setup)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load base model
    inference_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, 
        num_labels=3,
        torch_dtype=torch.bfloat16
    )
    
    # Resize token embeddings to match tokenizer (pad token was added during training)
    inference_model.resize_token_embeddings(len(tokenizer))
    inference_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create datasets and add text column
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Add text column to datasets (matching training format)
    def add_text_column(example):
        # Format: task: prompt\nA: response1\nB: response2 (matches fine_tuning_curriculum.py)
        example['text'] = f"{example['task']}: {example['prompt']}\nA: {example['response1']}\nB: {example['response2']}"
        # Convert 'winner' to integer label for Trainer compatibility
        if 'winner' in example:
            example['label'] = label_map[example['winner']]
        return example
    
    val_dataset = val_dataset.map(add_text_column)
    test_dataset = test_dataset.map(add_text_column)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Remove unnecessary columns, keep only model inputs and labels
    columns_to_keep = ['input_ids', 'attention_mask', 'label']
    columns_to_remove = [col for col in val_dataset.column_names if col not in columns_to_keep]
    val_dataset = val_dataset.remove_columns(columns_to_remove)
    test_dataset = test_dataset.remove_columns(columns_to_remove)
    
    # Load model
    model = PeftModel.from_pretrained(inference_model, peft_model_id)

    trainer = Trainer(
      model=model,
      args=test_args,
      compute_metrics=compute_metrics,
      tokenizer=tokenizer,
    )
    
    epoch = trainer.state.epoch if trainer.state.epoch else 0
    steps = trainer.state.global_step
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_prediction = trainer.predict(val_dataset)
    val_predictions = np.argmax(val_prediction.predictions, axis=-1)
    val_labels = val_prediction.label_ids
    
    for i in range(len(val_predictions)):
      datadict.append({
          'peft_model_id': peft_model_id, 
          'steps': steps, 
          'epoch': epoch, 
          'split': 'validation',
          'predictions': val_predictions[i], 
          'ratings': val_labels[i]
      })
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_prediction = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_prediction.predictions, axis=-1)
    test_labels = test_prediction.label_ids
    
    for i in range(len(test_predictions)):
      datadict.append({
          'peft_model_id': peft_model_id, 
          'steps': steps, 
          'epoch': epoch, 
          'split': 'test',
          'predictions': test_predictions[i], 
          'ratings': test_labels[i]
      })
    
    del model
    del tokenizer
    del trainer
    del val_dataset
    del test_dataset
    torch.cuda.empty_cache()

out_df = pd.DataFrame.from_dict(datadict)
out_df.to_csv('epoch_wise_curriculum_results.csv', index=False)
print(f"\nResults saved to epoch_wise_curriculum_results.csv")
print(f"Total predictions: {len(out_df)}")
