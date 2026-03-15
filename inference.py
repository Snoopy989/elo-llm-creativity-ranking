#!/usr/bin/env python
# coding: utf-8

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login
import json
import numpy as np
import pandas as pd
import random
import time
import logging
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _fix_score_head(model, adapter_path):
    """
    Reconstruct the classification head (score layer) from the adapter checkpoint.

    The checkpoint stores score as a LoraLinear inside ModulesToSaveWrapper
    (base_layer + lora_A/B), but PEFT ≥ 0.8 re-creates it as a plain Linear
    inside ModulesToSaveWrapper when loading.  The LoRA-specific keys silently
    fail to map, leaving the score layer randomly initialised.

    Fix: manually compute  effective_weight = base_layer + (B @ A) * (α/r)
    from the safetensors file and inject it into the loaded model.
    """
    from safetensors import safe_open
    import os, json

    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(safetensors_path):
        logging.warning("No adapter_model.safetensors found — skipping score-head fix.")
        return

    with open(config_path) as f:
        cfg = json.load(f)
    lora_alpha = cfg.get("lora_alpha", 32)
    lora_r = cfg.get("r", 4)
    scaling = lora_alpha / lora_r

    with safe_open(safetensors_path, framework="pt") as sf:
        keys = list(sf.keys())

        # Locate score weights in the checkpoint
        base_key = "base_model.model.score.base_layer.weight"
        lora_a_key = "base_model.model.score.modules_to_save.lora_A.weight"
        lora_b_key = "base_model.model.score.modules_to_save.lora_B.weight"

        # Fall back to top-level lora keys if modules_to_save variants missing
        if lora_a_key not in keys:
            lora_a_key = "base_model.model.score.lora_A.weight"
        if lora_b_key not in keys:
            lora_b_key = "base_model.model.score.lora_B.weight"

        if base_key not in keys:
            logging.warning("score.base_layer.weight not in checkpoint — skipping fix.")
            return

        base_w = sf.get_tensor(base_key).float()
        lora_A = sf.get_tensor(lora_a_key).float()
        lora_B = sf.get_tensor(lora_b_key).float()

    effective_w = base_w + (lora_B @ lora_A) * scaling

    # Inject into the loaded PeftModel's score layer
    score = model.base_model.model.score
    if hasattr(score, "modules_to_save"):
        # ModulesToSaveWrapper path
        active = list(score.modules_to_save.keys())[0]
        target = score.modules_to_save[active]
        target.weight.data.copy_(effective_w.to(target.weight.dtype))
        logging.info(f"Score head fixed via ModulesToSaveWrapper (effective_w std={effective_w.std():.4f})")
    elif hasattr(score, "weight"):
        score.weight.data.copy_(effective_w.to(score.weight.dtype))
        logging.info(f"Score head fixed via direct weight injection (effective_w std={effective_w.std():.4f})")
    else:
        logging.warning("Could not locate score weight tensor to fix.")


def load_model_and_tokenizer(
    base_model_path="meta-llama/Llama-2-13b-hf",
    adapter_path="sctt_results_curriculum_10_epochs_Llama-2-13b-hf/phase_3/checkpoint-540000",
    tokenizer_path="meta-llama/Llama-2-13b-hf",
):
    """Load the fine-tuned model and tokenizer."""
    logging.info(f"Loading model from {adapter_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Load base model as SequenceClassification with 3 labels
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=3,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Resize base model to match tokenizer vocab size and set pad token
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # Load PEFT adapter (do NOT merge_and_unload — the score/classifier
        # head is in both target_modules and modules_to_save, so merging
        # corrupts the classification head weights)
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Fix score head: PEFT silently drops LoRA weights for the score layer
        # when it's in both target_modules and modules_to_save.  Reconstruct
        # the effective weight manually from the checkpoint.
        _fix_score_head(model, adapter_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        logging.info(f"Ready on {device}")

        return model, tokenizer, device

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None



def ask_llama(prompt, model, tokenizer, device, max_tokens=128, temperature=0.1, debug=False):
    """
    Classify using fine-tuned LLaMA-2 7B sequence classification model

    Args:
        prompt (str): Your comparison prompt
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: The device to run on
        max_tokens (int): Ignored for classification
        temperature (float): Ignored for classification (argmax)
        debug (bool): Print debug info

    Returns:
        str: JSON response with winner classification
    """
    import re

    if model is None or tokenizer is None:
        return '{"winner": "Equal", "error": "Model not loaded"}'

    try:
        # Tokenize input — padding="max_length" to match training tokenization
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=180,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get classification logits
        with torch.no_grad():
            logits = model(**inputs).logits

        # Convert logits to probabilities (cast to float32 for stable softmax)
        probs = torch.softmax(logits.float(), dim=-1).squeeze()

        # Map to labels: 0=A, 1=B, 2=Equal
        label_map = {0: "A", 1: "B", 2: "Equal"}
        predicted_class = torch.argmax(logits, dim=-1).item()
        winner = label_map[predicted_class]
        confidence = round(probs[predicted_class].item(), 3)

        if debug:
            print(f"DEBUG - Logits: {logits}")
            print(f"DEBUG - Probabilities: {probs}")
            print(f"DEBUG - Predicted class: {predicted_class} ({winner})")

        response_data = {"winner": winner, "confidence": confidence}
        return json.dumps(response_data)

    except Exception as e:
        error_response = {"winner": "Equal", "error": str(e)}
        if debug:
            print(f"DEBUG - Error in ask_llama: {e}")
        return json.dumps(error_response)

def ask_llama_batch(prompts, model, tokenizer, device):
    """
    Classify a batch of prompts in a single forward pass.

    Args:
        prompts: list of prompt strings
        model, tokenizer, device: as usual

    Returns:
        list of dicts: [{"winner": str, "confidence": float}, ...]
    """
    LABEL_MAP = {0: "A", 1: "B", 2: "Equal"}
    fallback = {"winner": "Equal", "confidence": 0.0}

    if model is None or tokenizer is None or len(prompts) == 0:
        return [fallback.copy() for _ in prompts]

    try:
        inputs = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=180,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits          # (batch, 3)

        probs = torch.softmax(logits.float(), dim=-1)  # (batch, 3)
        preds = torch.argmax(logits, dim=-1)            # (batch,)

        results = []
        for idx in range(len(prompts)):
            cls = preds[idx].item()
            results.append({
                "winner":     LABEL_MAP[cls],
                "confidence": round(probs[idx, cls].item(), 3),
            })
        return results

    except Exception as e:
        logging.error(f"Batch inference error: {e}")
        return [fallback.copy() for _ in prompts]


def build_comparison_prompt(row_a, row_b):
    """
    row_a, row_b: pandas Series or dict for items A and B
    Returns a string prompt for the LLM.
    Format matches the training format used in inference_stochastic_passes.py.
    """
    task = row_a.get("task", "Unknown Task")
    prompt_description = row_a.get("prompt", "No prompt provided")
    response_a = row_a.get("response", row_a.get("response1", ""))
    response_b = row_b.get("response", row_b.get("response1", ""))

    # Match the EXACT format used during fine-tuning
    prompt = (
        f"{task}: {prompt_description}\n"
        f"A: {response_a}\n"
        f"B: {response_b}"
    )
    return prompt

def random_pairwise_prompts(df, num_pairs=5, seed=None):
    """
    Selects num_pairs random pairs from df and builds prompts for LLM comparison.
    Returns a list of prompts and the corresponding responses for each pair.
    """
    if seed is not None:
        random.seed(seed)
    n = len(df)
    pairs = set()
    while len(pairs) < num_pairs:
        i, j = random.sample(range(n), 2)
        if i != j:
            pairs.add(tuple(sorted((i, j))))
    prompts = []
    for i, j in pairs:
        row_a = df.iloc[i].to_dict()
        row_b = df.iloc[j].to_dict()
        prompt = build_comparison_prompt(row_a, row_b)
        prompts.append({
            "prompt": prompt,
            "response_a": row_a.get("response", ""),
            "response_b": row_b.get("response", ""),
            "index_a": i,
            "index_b": j
        })
    return prompts

def elo_update(rating_a, rating_b, score_a, score_b, k=6):
    """
    Update Elo ratings for two players/teams, allowing for ties.
    score_a, score_b: 1 = win, 0.5 = tie, 0 = loss
    k: K-factor (sensitivity)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)
    return new_rating_a, new_rating_b

def run_llm_elo_ranking(df, model, tokenizer, device, num_pairs=None, max_rounds=None, k=6, seed=None, adaptive_k=True, k_start=32, k_end=2, convergence_patience=20, exhaustive_pairs=None):
    """
    Main ELO convergence loop.
    If exhaustive_pairs is provided (list of pair dicts), each round shuffles and iterates
    ALL of them — guaranteed full dataset coverage.
    Otherwise, num_pairs random pairs are sampled each round.
    If max_rounds is None, runs until convergence.
    convergence_patience: stop if correlation does not improve for this many rounds.
    Returns a tuple: (final Elo ratings dict, history list of dicts)
    """
    # Initialize Elo ratings
    ratings = {i: 1500 for i in range(len(df))}
    history = []

    # For correlation patience
    best_corr = float('-inf')
    last_improved_round = 0

    round_num = 0
    while max_rounds is None or round_num < max_rounds:
        # Adaptive k: start high for fast initial convergence, decrease for stability
        if adaptive_k:
            # Decay over enough rounds for each item to accumulate ~30 comparisons
            # Formula: 30 * n_items / (2 * num_pairs), floored at 3 rounds
            # e.g. 4666 items, 50000 pairs/round → decay over ~3 rounds
            n_items = len(df)
            effective_num_pairs = num_pairs if num_pairs is not None else n_items // 2
            decay_rounds = max_rounds if max_rounds is not None else 30
            # Linear decay from k_start to k_end over decay_rounds
            current_k = k_start - (k_start - k_end) * (round_num / decay_rounds)
            current_k = max(current_k, k_end)  # Don't go below k_end
        else:
            current_k = k

        start_time = time.time()

        # Use all pre-defined pairs (shuffled) or sample random pairs
        if exhaustive_pairs is not None:
            pairs = exhaustive_pairs.copy()
            random.shuffle(pairs)
        else:
            pairs = random_pairwise_prompts(df, num_pairs=num_pairs, seed=None)

        changes = 0
        successful_comparisons = 0
        for pair_idx, pair in enumerate(pairs):
            try:
                i, j = pair.get("index_a"), pair.get("index_b")
                if i is None or j is None:
                    raise KeyError("Missing 'index_a' or 'index_b' in pair: " + str(pair))
                prompt = pair["prompt"]

                # Call the LLM to get a structured response
                response = ask_llama(prompt, model, tokenizer, device)

                try:
                    # Parse JSON response directly (since fine-tuned model returns clean JSON)
                    response_data = json.loads(response)
                    winner = response_data.get("winner", "Equal")
                    confidence = response_data.get("confidence", 0.0)
                except (json.JSONDecodeError, ValueError) as e:
                    # Fallback: try regex extraction
                    import re
                    json_match = re.search(r'\{.*\}', response)
                    if json_match:
                        response_data = json.loads(json_match.group())
                        winner = response_data.get("winner", "Equal")
                        confidence = response_data.get("confidence", 0.0)
                    else:
                        logging.error(f"JSON parsing error: {e}")
                        continue  # Skip this pair entirely

                # Parse response to scores
                if winner == "A":
                    score_a, score_b = 1, 0
                elif winner == "B":
                    score_a, score_b = 0, 1
                else:
                    score_a, score_b = 0.5, 0.5

                old_a, old_b = ratings[i], ratings[j]
                new_a, new_b = elo_update(old_a, old_b, score_a, score_b, k=current_k)
                ratings[i], ratings[j] = new_a, new_b
                changes += abs(new_a - old_a) + abs(new_b - old_b)
                successful_comparisons += 1

                # Update the DataFrame with the latest Elo scores
                df.at[i, "Elo"] = new_a
                df.at[j, "Elo"] = new_b

                history.append({
                    "round": round_num + 1,  # Store 1-based round number
                    "pair_in_round": pair_idx,
                    "index_a": i,
                    "index_b": j,
                    "response": response,
                    "winner": winner,
                    "confidence": confidence,
                    "k_used": current_k,
                    "old_a": old_a,
                    "old_b": old_b,
                    "new_a": new_a,
                    "new_b": new_b
                })

            except KeyError as e:
                logging.error(f"KeyError: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                continue

        end_time = time.time()
        winner_counts = {"A": 0, "B": 0, "Equal": 0}
        for entry in history[-successful_comparisons:]:
            winner_counts[entry["winner"]] = winner_counts.get(entry["winner"], 0) + 1

        # Calculate and display real-time correlation after each round
        current_corr = None
        corr_str = ""
        try:
            current_ratings = [ratings[i] for i in range(len(df))]
            min_elo, max_elo = min(current_ratings), max(current_ratings)
            normalized_elos = [(r - min_elo) / (max_elo - min_elo) if max_elo > min_elo else 0.0 for r in current_ratings]
            if "norm_response" in df.columns:
                valid_indices = ~(np.isnan(df["norm_response"]) | np.isnan(normalized_elos))
                if np.sum(valid_indices) >= 2:
                    correlation, p_value = pearsonr(
                        df["norm_response"][valid_indices],
                        np.array(normalized_elos)[valid_indices]
                    )
                    current_corr = correlation
                    corr_str = f" | r={correlation:.4f}"
                    if correlation > best_corr:
                        best_corr = correlation
                        last_improved_round = round_num
        except Exception as e:
            logging.warning(f"Corr error: {e}")

        k_str = f" k={round(current_k,1)}" if adaptive_k else ""
        logging.info(
            f"R{round_num+1}{k_str} {end_time-start_time:.0f}s "
            f"{successful_comparisons}/{len(pairs)} "
            f"A/B/Eq:{winner_counts['A']}/{winner_counts['B']}/{winner_counts['Equal']}"
            f"{corr_str}"
        )

        # Convergence check: if average change per item is small or most changes are <1 or correlation hasn't improved in 10 rounds
        num_items = len(ratings)
        avg_change = changes / (2 * successful_comparisons) if successful_comparisons > 0 else 0  # Per item
        small_changes = sum(1 for entry in history[-successful_comparisons:] if abs(entry['new_a'] - entry['old_a']) < 1 and abs(entry['new_b'] - entry['old_b']) < 1)
        percent_small = small_changes / (2 * successful_comparisons) if successful_comparisons > 0 else 0

        if avg_change < 0.1 or percent_small > 0.95 or (round_num - last_improved_round >= convergence_patience):
            logging.info(f"Converged at R{round_num + 1}")
            break

        round_num += 1

    return ratings, history  # Ensure only two values are returned

def calculate_accuracy(df):
    """
    Calculate the average raw deviation of Elo_normalized from norm_response.
    norm_response is considered the ground truth.

    Args:
        df (pd.DataFrame): DataFrame containing 'Elo_normalized' and 'norm_response'.

    Returns:
        float: Average raw deviation.
    """
    if "norm_response" not in df.columns or "Elo_normalized" not in df.columns:
        raise ValueError("The DataFrame must contain 'norm_response' and 'Elo_normalized' columns.")

    # Calculate raw deviation, handling cases where norm_response is zero
    def calculate_deviation(row):
        if row["norm_response"] == 0:
            return abs(row["Elo_normalized"])
        return abs(row["Elo_normalized"] - row["norm_response"])

    df["raw_deviation"] = df.apply(calculate_deviation, axis=1)

    # Calculate the mean raw deviation
    average_deviation = df["raw_deviation"].mean()

    logging.info(f"Average raw deviation: {average_deviation:.4f}")
    return average_deviation

def calculate_pearson_correlation(df):
    """
    Calculate the Pearson correlation coefficient between norm_response and Elo_normalized.

    Args:
        df (pd.DataFrame): DataFrame containing 'norm_response' and 'Elo_normalized' columns.

    Returns:
        float: Pearson correlation coefficient.
    """
    if "norm_response" not in df.columns or "Elo_normalized" not in df.columns:
        raise ValueError("The DataFrame must contain 'norm_response' and 'Elo_normalized' columns.")

    # Remove any rows with NaN values
    clean_df = df[["norm_response", "Elo_normalized"]].dropna()

    if len(clean_df) < 2:
        logging.warning("Not enough valid data points for correlation calculation.")
        return np.nan

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(clean_df["norm_response"], clean_df["Elo_normalized"])

    logging.info(f"Pearson correlation: {correlation:.4f} (p={p_value:.4f}, n={len(clean_df)})")

    return correlation

def get_final_elo_df(df, ratings):
    """
    Create a final DataFrame with Elo ratings and normalized Elo scores.

    Args:
        df (pd.DataFrame): Original DataFrame
        ratings (dict): Elo ratings dict with indices as keys

    Returns:
        pd.DataFrame: DataFrame with Elo and Elo_normalized columns added
    """
    final_df = df.copy()
    final_df["Elo"] = [ratings[i] for i in range(len(df))]

    # Min-max normalize Elo scores
    min_elo = final_df["Elo"].min()
    max_elo = final_df["Elo"].max()
    if max_elo > min_elo:
        final_df["Elo_normalized"] = (final_df["Elo"] - min_elo) / (max_elo - min_elo)
    else:
        final_df["Elo_normalized"] = 0.0

    return final_df

def calculate_metrics(df):
    """
    Calculate R², MAE, and RMSE between norm_response and Elo_normalized.

    Args:
        df (pd.DataFrame): DataFrame containing 'norm_response' and 'Elo_normalized' columns.

    Returns:
        dict: Dictionary containing r_squared, mae, and rmse values.
    """
    if "norm_response" not in df.columns or "Elo_normalized" not in df.columns:
        raise ValueError("The DataFrame must contain 'norm_response' and 'Elo_normalized' columns.")

    # Remove any rows with NaN values
    clean_df = df[["norm_response", "Elo_normalized"]].dropna()

    if len(clean_df) < 2:
        logging.warning("Not enough valid data points for metrics calculation.")
        return {"r_squared": np.nan, "mae": np.nan, "rmse": np.nan}

    y_true = clean_df["norm_response"]
    y_pred = clean_df["Elo_normalized"]

    # Calculate metrics
    r_squared = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    logging.info(f"R²={r_squared:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} (n={len(clean_df)})")

    return {"r_squared": r_squared, "mae": mae, "rmse": rmse}

def main():
    """Main function: load model, run ELO convergence loop on items from inference_pairs_val_pairs.csv."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ELO ranking with convergence loop on inference_pairs_val_pairs.csv")
    parser.add_argument("--data-path", type=str, default="inference_pairs_val_pairs.csv", help="Path to pairwise CSV")
    parser.add_argument("--base-model-path", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--adapter-path", type=str, default="sctt_results_curriculum_10_epochs_Llama-2-13b-hf/phase_3/checkpoint-540000")
    parser.add_argument("--tokenizer-path", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--num-pairs", type=int, default=10000, help="Random pairs sampled per ELO round")
    parser.add_argument("--max-rounds", type=int, default=None, help="Max ELO rounds (None = run until convergence)")
    parser.add_argument("--convergence-patience", type=int, default=20, help="Stop if correlation does not improve for this many rounds")
    parser.add_argument("--k-start", type=int, default=32, help="Initial K factor for adaptive ELO decay")
    parser.add_argument("--k-end", type=int, default=2, help="Final K factor for adaptive ELO decay")
    parser.add_argument("--item", type=str, default=None, help="Filter to a specific item (e.g. 'birds')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default=None, help="Save final results to CSV")
    args = parser.parse_args()

    # Authenticate with HuggingFace if token is available
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        tokenizer_path=args.tokenizer_path,
    )
    if model is None:
        logging.error("Failed to load model. Exiting.")
        return

    # Load pairs and extract unique items
    pairs_df = pd.read_csv(args.data_path)
    logging.info(f"Loaded {len(pairs_df):,} pairs from {args.data_path}")

    if args.item:
        pairs_df = pairs_df[pairs_df["item"] == args.item].reset_index(drop=True)
        if len(pairs_df) == 0:
            logging.error(f"No pairs for item='{args.item}'.")
            return
        logging.info(f"item='{args.item}' → {len(pairs_df):,} pairs")

    items1 = pairs_df[["task", "prompt", "response1", "norm_response1", "response1_id"]].rename(
        columns={"response1": "response", "norm_response1": "norm_response", "response1_id": "response_id"}
    )
    items2 = pairs_df[["task", "prompt", "response2", "norm_response2", "response2_id"]].rename(
        columns={"response2": "response", "norm_response2": "norm_response", "response2_id": "response_id"}
    )
    items_df = pd.concat([items1, items2], ignore_index=True)
    items_df = items_df.drop_duplicates(subset=["response_id"]).reset_index(drop=True)
    n_items = len(items_df)

    # Build index lookup: response_id -> items_df integer index
    id_to_idx = {rid: idx for idx, rid in enumerate(items_df["response_id"])}

    # Build exhaustive pairs list from all pre-defined pairs in pairs_df
    exhaustive_pairs = []
    skipped = 0
    for _, row in pairs_df.iterrows():
        idx_a = id_to_idx.get(row["response1_id"])
        idx_b = id_to_idx.get(row["response2_id"])
        if idx_a is None or idx_b is None:
            skipped += 1
            continue
        row_a = items_df.iloc[idx_a].to_dict()
        row_b = items_df.iloc[idx_b].to_dict()
        exhaustive_pairs.append({
            "prompt": build_comparison_prompt(row_a, row_b),
            "index_a": idx_a,
            "index_b": idx_b,
        })
    if skipped:
        logging.warning(f"Skipped {skipped} unresolvable pairs")
    logging.info(f"{n_items:,} items | {len(exhaustive_pairs):,} pairs/round")

    # Run ELO convergence loop
    ratings, history = run_llm_elo_ranking(
        df=items_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_rounds=args.max_rounds,
        seed=args.seed,
        convergence_patience=args.convergence_patience,
        k_start=args.k_start,
        k_end=args.k_end,
        exhaustive_pairs=exhaustive_pairs,
    )

    # Build final DataFrame with normalized ELO scores
    final_df = get_final_elo_df(items_df, ratings)

    # Report metrics
    logging.info("\n=== Final Metrics ===")
    pearson = calculate_pearson_correlation(final_df)
    metrics = calculate_metrics(final_df)
    deviation = calculate_accuracy(final_df)

    print(f"\nPearson r:     {pearson:.4f}")
    print(f"R²:            {metrics['r_squared']:.4f}")
    print(f"MAE:           {metrics['mae']:.4f}")
    print(f"RMSE:          {metrics['rmse']:.4f}")
    print(f"Avg deviation: {deviation:.4f}")

    # Show top-ranked items
    top = final_df.nlargest(10, "Elo")[["response_id", "response", "Elo", "Elo_normalized", "norm_response"]]
    print(f"\nTop 10 highest-ranked items:\n{top.to_string(index=False)}")

    # Optionally save results
    if args.output_csv:
        final_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

