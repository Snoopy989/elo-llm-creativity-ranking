# Production Code for LLM Fine-Tuning and Inference

![Python](https://img.shields.io/badge/python-3.12+-blue)

This directory contains clean, production-ready Python scripts for fine-tuning Llama 2 models and running inference for Elo ranking of responses.

## Quick Start with Pipeline

The easiest way to run the complete pipeline is using the automated script:

```bash
python pipeline.py
```

This will automatically:
1. Install all required dependencies
2. Download raw data
3. Parse and clean data
4. Download the Llama 2 model (requires Hugging Face token)
5. Fine-tune the model
6. Run inference and evaluation

### Prerequisites for Pipeline

1. **Python 3.6+** (preferably 3.12+)
2. **CUDA-compatible GPU** (required for fine-tuning)
3. **Hugging Face Account** with access to Llama 2 models
4. **Set your Hugging Face token** in `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_actual_token_here
   ```
   Get your token from: https://huggingface.co/settings/tokens

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX 10-series or newer, RTX series, etc.)
- **VRAM**: At least 8GB (16GB+ recommended for Llama 2 7B)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for model and data

### Pipeline Output

The pipeline provides detailed progress updates and will create:
- `all_sctt_jrt.csv`: Raw dataset
- `pairs_dataset.csv`: Processed training pairs
- `fine_tuned_model/`: Fine-tuned model
- Various evaluation outputs and metrics

## Files

### `pipeline.py` - **RECOMMENDED ENTRY POINT**
Automated pipeline script that runs the complete workflow.

**Features:**
- Automatic dependency installation
- Progress tracking with status updates
- Error handling and validation
- Computer-agnostic execution
- Environment variable management

**Usage:**
```bash
python pipeline.py
```

**Requirements:**
- Internet connection for data/model downloads
- Hugging Face token in `.env` file
- Sufficient disk space (~20GB for model)

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

### `download_raw_data.py`
Downloads the source datasets required for preprocessing and training.

**Functions:**
- Fetches raw data files from configured sources (e.g., cloud storage or URLs)
- Saves datasets locally in a structured directory for subsequent parsing

**Usage:**
```bash
python download_raw_data.py
```
Downloaded files will be placed under a local `data/` directory by default (or as configured inside the script).

### `requirements.txt`
Lists all required Python packages.

## Installation & Usage

**Automated (Recommended):**
```bash
python pipeline.py
```
Handles everything: dependencies, data download, processing, model download, fine-tuning, and evaluation.

**Manual Steps** (if needed):
```bash
python download_raw_data.py    # Download data
python parse_data.py          # Process data
python fine_tuning.py         # Train model
python inference.py           # Run evaluation
```

**Programmatic Usage:**
```python
from inference import *
model, tokenizer, device = load_model_and_tokenizer()
ratings, history = run_llm_elo_ranking(df, model, tokenizer, device)
```

## Environment Setup

### Hugging Face Token
Required for downloading Llama 2 models. Set up in one of these ways:

1. **Using .env file** (recommended):
   ```
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. **Environment variable**:
   ```bash
   # Windows
   set HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   
   # Linux/Mac
   export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   ```

Get your token from: https://huggingface.co/settings/tokens

**Important:** You must accept the Llama 2 model terms on Hugging Face before downloading.

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

- **Automated Pipeline**: Use `pipeline.py` for the complete automated workflow with progress tracking
- All scripts are designed to work in cluster environments
- Models are loaded locally after download
- Elo ranking uses adaptive K-factor for convergence
- Correlation with ground truth is calculated automatically
- The pipeline handles dependency installation and environment setup automatically
- Progress is displayed in real-time during execution
- Errors are caught and reported with helpful messages

## Troubleshooting

### Pipeline Issues
- **"CUDA is not available"**: Install CUDA PyTorch as shown above, or ensure you have a CUDA-compatible GPU
- **"SyntaxError"**: Ensure you're using Python 3.6+ and the files aren't corrupted
- **"ModuleNotFoundError"**: The pipeline installs dependencies automatically; try manual installation if it fails
- **"Hugging Face token"**: Ensure your token is set in `.env` and you have access to Llama 2

### Model Download Issues
- **"403 Forbidden"**: Accept Llama 2 terms on Hugging Face and ensure token has proper permissions
- **"No space left"**: Ensure ~20GB free space for model download
- **Network timeout**: Check internet connection and try again

### Fine-tuning Issues
- **"CUDA out of memory"**: Reduce batch size or use gradient checkpointing
- **"Token embedding resize interrupted"**: The script handles this gracefully but performance may be affected
- **"tf-keras not found"**: Pipeline installs this automatically; ensure pip works

### CUDA-Specific Issues
- **"PyTorch CPU version detected"**: Pipeline installs CUDA PyTorch; restart Python session if needed
- **"cuDNN version mismatch"**: Update GPU drivers or use compatible PyTorch/CUDA versions
- **"Device-side assert triggered"**: Usually indicates data preprocessing issues

### General
- Check the pipeline output for specific error messages
- Ensure all scripts are in the same directory
- For cluster environments, ensure proper Python path and permissions

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{gregory2025elo,
  title={A Novel Approach to Creativity Assessment: Forced Pairwise Ranking with Large Language Models},
  author={Gregory, Phillip and Noh, Jiho and Grouchnikov, Sam},
  year={2025}
}
```