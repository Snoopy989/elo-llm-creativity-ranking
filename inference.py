#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import json
import numpy as np
import pandas as pd
import random
import time
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model_and_tokenizer(base_model_path="meta-llama/Llama-2-7b-hf", adapter_path="./fine_tuned_model_merged_v6"):
    """Load the fine-tuned model and tokenizer."""
    print("Loading fine-tuned sequence classification model...")
    print(f"Base model: {base_model_path}")
    print(f"Adapter path: {adapter_path}")

    try:
        # Load tokenizer from adapter path
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")

        # Load base model as SequenceClassification with 3 labels
        print("Loading base model...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=3,  # Matches fine-tuning
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Resize base model to match tokenizer vocab size
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Base model loaded")

        # Load PEFT adapter
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"LoRA adapter loaded")

        # Merge and unload for faster inference
        print("Merging adapter into base model...")
        model = model.merge_and_unload()
        print(f"Model merged")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print(f"\nModel loaded successfully on {device}")
        print(f"Model type: SequenceClassification with 3 labels (A/B/Equal)")

        return model, tokenizer, device

    except Exception as e:
        print(f"\nError loading model: {e}")
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
        # Tokenize input
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get classification logits
        with torch.no_grad():
            logits = model(**inputs).logits

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1).squeeze()

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

def build_comparison_prompt(row_a, row_b):
    """
    row_a, row_b: pandas Series or dict for items A and B
    Returns a string prompt for the LLM.

    """
    task = row_a.get("task", "Unknown Task")
    prompt_description = row_a.get("prompt", "No prompt provided")

    # Match the EXACT format used during fine-tuning
    prompt = (
        f"Task: {task}\n"
        f"Description: {prompt_description}\n\n"
        f"Response A: {row_a['response']}\n"
        f"Response B: {row_b['response']}\n\n"
        "Which response is more scientifically possible and provable with a hypothesis?"
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

def run_llm_elo_ranking(df, model, tokenizer, device, num_pairs=10, max_rounds=None, k=6, seed=None, adaptive_k=True, k_start=32, k_end=4):
    """
    Main loop: select random pairs, get LLM ratings, update Elo, repeat until convergence or max_rounds.
    Generates NEW random pairs for each round.
    If max_rounds is None, runs until convergence.
    If adaptive_k=True, k starts at k_start and decreases to k_end over rounds for faster convergence.
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
            # Use max_rounds if provided, otherwise default to 10 rounds for decay
            decay_rounds = max_rounds if max_rounds is not None else 20
            # Linear decay from k_start to k_end over decay_rounds
            current_k = k_start - (k_start - k_end) * (round_num / decay_rounds)
            current_k = max(current_k, k_end)  # Don't go below k_end
        else:
            current_k = k

        print(f"\n=== Round {round_num + 1} ===")
        if adaptive_k:
            print(f"Using adaptive k: {current_k:.1f}")

        start_time = time.time()

        # Generate NEW pairs for each round
        pairs = random_pairwise_prompts(df, num_pairs=num_pairs, seed=None)  # Fresh randomness each round
        print(f"Generated {len(pairs)} pairs for round {round_num + 1}")

        changes = 0
        successful_comparisons = 0

        print(f"Starting LLM calls for {len(pairs)} pairs...")
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
                        print(f"JSON parsing error: {e}")
                        print(f"Response causing error: {response}")
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
                print(f"KeyError encountered: {e}")
                print("Pair data:", pair)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                print("Pair data:", pair)
                continue

        end_time = time.time()
        print(f"Round {round_num + 1} LLM processing took {end_time - start_time:.2f} seconds")

        print(f"Round {round_num + 1} completed: {successful_comparisons}/{len(pairs)} successful comparisons")
        print(f"Total rating changes: {changes:.2f}")

        # Calculate and display real-time correlation after each round
        current_corr = None
        try:
            # Update DataFrame with current Elo scores
            current_ratings = [ratings[i] for i in range(len(df))]

            # Min-max normalize current Elo scores
            min_elo = min(current_ratings)
            max_elo = max(current_ratings)
            if max_elo > min_elo:  # Avoid division by zero
                normalized_elos = [(rating - min_elo) / (max_elo - min_elo) for rating in current_ratings]
            else:
                normalized_elos = [0.0] * len(current_ratings)

            # Calculate correlation with ground truth if norm_response column exists
            if "norm_response" in df.columns:
                # Remove any NaN pairs
                valid_indices = ~(np.isnan(df["norm_response"]) | np.isnan(normalized_elos))
                if np.sum(valid_indices) >= 2:
                    correlation, p_value = pearsonr(
                        df["norm_response"][valid_indices],
                        np.array(normalized_elos)[valid_indices]
                    )
                    current_corr = correlation
                    print(f"Current correlation with ground truth: {correlation:.4f} (p={p_value:.4f})")

                    # Update best correlation
                    if correlation > best_corr:
                        best_corr = correlation
                        last_improved_round = round_num
                else:
                    print("Not enough valid data for correlation calculation")
            else:
                print("No ground truth column found for correlation calculation")

        except Exception as e:
            print(f"Error calculating real-time correlation: {e}")

        # Convergence check: if average change per item is small or most changes are <1 or correlation hasn't improved in 10 rounds
        num_items = len(ratings)
        avg_change = changes / (2 * successful_comparisons) if successful_comparisons > 0 else 0  # Per item
        small_changes = sum(1 for entry in history[-successful_comparisons:] if abs(entry['new_a'] - entry['old_a']) < 1 and abs(entry['new_b'] - entry['old_b']) < 1)
        percent_small = small_changes / (2 * successful_comparisons) if successful_comparisons > 0 else 0

        if avg_change < 0.1 or percent_small > 0.95 or (round_num - last_improved_round >= 10):
            print(f"Converged at round {round_num + 1} (avg change: {avg_change:.4f}, % small changes: {percent_small:.2%}, rounds since corr improvement: {round_num - last_improved_round})")
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

    print(f"Average raw deviation: {average_deviation:.4f}")
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
        print("Not enough valid data points for correlation calculation.")
        return np.nan

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(clean_df["norm_response"], clean_df["Elo_normalized"])

    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Number of data points used: {len(clean_df)}")

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
        print("Not enough valid data points for metrics calculation.")
        return {"r_squared": np.nan, "mae": np.nan, "rmse": np.nan}

    y_true = clean_df["norm_response"]
    y_pred = clean_df["Elo_normalized"]

    # Calculate metrics
    r_squared = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"R² (Coefficient of Determination): {r_squared:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"Number of data points used: {len(clean_df)}")

    return {"r_squared": r_squared, "mae": mae, "rmse": rmse}

def main():
    """Main function to demonstrate inference pipeline."""
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()

    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Example usage: Load test data and run Elo ranking
    # This would need actual data file paths
    print("\nModel loaded successfully. Ready for inference.")
    print("To use:")
    print("1. Load your test data into a DataFrame with columns: response, task, prompt, item, norm_response")
    print("2. Call run_llm_elo_ranking(df, model, tokenizer, device, num_pairs=500)")
    print("3. Use calculate_metrics() and other functions for evaluation")

if __name__ == "__main__":
    main()

