# Production Code for LLM Fine-Tuning and Inference

![Python](https://img.shields.io/badge/python-3.12+-blue) 

This directory contains clean, production-ready Python scripts for fine-tuning Llama 2 models and running inference for Elo ranking of responses.

## Files

### `fine_tuning.py`
Fine-tunes a Llama 2 model using LoRA (Low-Rank Adaptation) for sequence classification.

**Functions:**
- `setup_environment()`: Sets up the environment and checks for required files
- `load_model_and_tokenizer()`: Loads the base model and tokenizer
- `prepare_dataset(tokenizer)`: Prepares the dataset for training
- `create_training_args()`: Creates training arguments
- `train_model(model, tokenizer, dataset, training_args)`: Trains the model
- `evaluate_model(model, tokenizer, dataset)`: Evaluates the trained model
- `main()`: Main function to run the complete pipeline

**Usage:**
```python
python fine_tuning.py
```

**Requirements:**
- `pairs_dataset.csv` in the same directory
- Fine-tuned model will be saved to `./fine_tuned_model/`

### `parse_data.py`
Processes raw data into training pairs for fine-tuning.

**Functions:**
- `find_input_file()`: Locates the input CSV file
- `load_data(input_file)`: Loads and preprocesses data
- `remove_duplicate_responses(df, strategy)`: Removes duplicate responses
- `create_individual_files(df, output_dir)`: Creates individual CSV files per group
- `create_pairs_dataset(df, ...)`: Creates pairs dataset for training
- `check_data_leakage(pairs_df)`: Checks for data leakage between train/test
- `main()`: Main function to run data processing

**Usage:**
```python
python parse_data.py
```

**Input:** `all_sctt_jrt.csv`
**Outputs:** `pairs_dataset.csv`, `split_assignments.csv`, individual files in `grouped_data/`

### `inference.py`
Runs inference using the fine-tuned model for Elo ranking of responses.

**Functions:**
- `load_model_and_tokenizer(...)`: Loads the fine-tuned model
- `ask_llama(prompt, model, tokenizer, device, ...)`: Gets classification from model
- `build_comparison_prompt(row_a, row_b)`: Builds prompt for comparison
- `random_pairwise_prompts(df, num_pairs, seed)`: Generates random pairs
- `elo_update(rating_a, rating_b, score_a, score_b, k)`: Updates Elo ratings
- `run_llm_elo_ranking(df, model, tokenizer, device, ...)`: Main Elo ranking function
- `calculate_accuracy(df)`, `calculate_pearson_correlation(df)`, etc.: Evaluation metrics
- `main()`: Demonstrates usage

**Usage:**
```python
# Load model
model, tokenizer, device = load_model_and_tokenizer()

# Load your test data (DataFrame with columns: response, task, prompt, item, norm_response)
df = pd.read_csv("your_test_data.csv")

# Run Elo ranking
ratings, history = run_llm_elo_ranking(df, model, tokenizer, device, num_pairs=500)

# Get final results
final_df = get_final_elo_df(df, ratings)
metrics = calculate_metrics(final_df)
```

### `huggingface_downloader.py`
Downloads models from Hugging Face (separate script for model acquisition).

### `requirements.txt`
Lists all required Python packages.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For model downloading (if needed):
```bash
python huggingface_downloader.py
```

## Workflow

1. **Data Preparation:**
   ```bash
   python parse_data.py
   ```

2. **Model Fine-Tuning:**
   ```bash
   python fine_tuning.py
   ```

3. **Inference and Ranking:**
   ```python
   from inference import *
   # Load model and run ranking as shown above
   ```

## Data Format

### Input Data (`all_sctt_jrt.csv`)
- `response`: Text response
- `item`, `task`, `prompt`: Grouping columns
- `judge_response`: Numerical score
- Other metadata columns

### Pairs Dataset (`pairs_dataset.csv`)
- `response1`, `response2`: Paired responses
- `winner`: "A", "B", or "Equal"
- `split`: "train" or "test"
- `norm_response1`, `norm_response2`: Normalized scores

### Test Data for Inference
DataFrame with columns:
- `response`: Response text
- `task`: Task description
- `prompt`: Prompt text
- `item`: Item identifier
- `norm_response`: Ground truth normalized score (0-1)

## Model Details

- **Base Model:** Llama 2 7B
- **Fine-Tuning:** LoRA with r=32, alpha=64
- **Task:** Sequence Classification (3 classes: A wins, B wins, Equal)
- **Max Length:** 128 tokens
- **Training:** ~5 Epochs with gradient accumulation

## Output Files

Fine-tuning creates:
- `results/lora_adapter/`: LoRA adapter weights
- `fine_tuned_model/`: Merged model for inference

Inference creates:
- `elo_rankings_{item}.csv`: Final rankings
- `elo_history_{item}.csv`: Comparison history
- `elo_metrics_{item}.csv`: Performance metrics

## Notes

- All scripts are designed to work in cluster environments
- Models are loaded locally
- Elo ranking uses adaptive K-factor for convergence
- Correlation with ground truth is calculated automatically