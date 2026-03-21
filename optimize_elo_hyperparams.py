#!/usr/bin/env python
# coding: utf-8
"""
ELO Hyperparameter Optimizer
=============================
1. Loads the fine-tuned model ONCE.
2. Runs inference on a sample of pairs (item=holes from pairs_train.csv) to cache predictions.
3. Grid searches k_start × k_end × max_rounds using cached predictions (no further model calls).
4. Reports best hyperparameters ranked by Pearson r.

Usage:
    python optimize_elo_hyperparams.py
    python optimize_elo_hyperparams.py --n-sample 3000 --item holes
"""

import os
import json
import random
import logging
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from huggingface_hub import login

from inference import (
    load_model_and_tokenizer,
    ask_llama,
    build_comparison_prompt,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ──────────────────────────────────────────────────────────────
# Step 1: Build item-level df and exhaustive pair list
# ──────────────────────────────────────────────────────────────

def build_items_and_pairs(pairs_df: pd.DataFrame):
    """
    From a pairs DataFrame, reconstruct a unique-item DataFrame
    and a flat list of (idx_a, idx_b) tuples.
    """
    items1 = pairs_df[["task", "prompt", "response1", "norm_response1", "response1_id"]].rename(
        columns={"response1": "response", "norm_response1": "norm_response", "response1_id": "response_id"}
    )
    items2 = pairs_df[["task", "prompt", "response2", "norm_response2", "response2_id"]].rename(
        columns={"response2": "response", "norm_response2": "norm_response", "response2_id": "response_id"}
    )
    items_df = pd.concat([items1, items2], ignore_index=True)
    items_df = items_df.drop_duplicates(subset=["response_id"]).reset_index(drop=True)

    id_to_idx = {rid: idx for idx, rid in enumerate(items_df["response_id"])}

    pair_list = []
    for _, row in pairs_df.iterrows():
        idx_a = id_to_idx.get(row["response1_id"])
        idx_b = id_to_idx.get(row["response2_id"])
        if idx_a is not None and idx_b is not None:
            pair_list.append((idx_a, idx_b, row))

    return items_df, pair_list, id_to_idx


# ──────────────────────────────────────────────────────────────
# Step 2: Run model inference once → cache predictions
# ──────────────────────────────────────────────────────────────

def cache_model_predictions(items_df, pair_list, model, tokenizer, device):
    """
    Runs model inference for every pair in pair_list.
    Returns a list of dicts: [{idx_a, idx_b, winner, prob_A, prob_B, prob_Equal}]
    """
    cached = []
    total = len(pair_list)
    logging.info(f"Caching predictions for {total:,} pairs...")

    for i, (idx_a, idx_b, row) in enumerate(pair_list):
        row_a = items_df.iloc[idx_a].to_dict()
        row_b = items_df.iloc[idx_b].to_dict()
        prompt = build_comparison_prompt(row_a, row_b)

        response_json = ask_llama(prompt, model, tokenizer, device)
        try:
            data = json.loads(response_json)
            winner = data.get("winner", "Equal")
            confidence = data.get("confidence", 0.0)
            # Reconstruct approximate probs from confidence + winner
            if winner == "A":
                prob_A, prob_B, prob_Equal = confidence, (1 - confidence) / 2, (1 - confidence) / 2
            elif winner == "B":
                prob_A, prob_B, prob_Equal = (1 - confidence) / 2, confidence, (1 - confidence) / 2
            else:
                prob_A, prob_B, prob_Equal = (1 - confidence) / 2, (1 - confidence) / 2, confidence
        except Exception:
            winner, prob_A, prob_B, prob_Equal = "Equal", 0.33, 0.33, 0.34

        cached.append({
            "idx_a": idx_a,
            "idx_b": idx_b,
            "winner": winner,
            "prob_A": prob_A,
            "prob_B": prob_B,
            "prob_Equal": prob_Equal,
        })

        if (i + 1) % 100 == 0 or (i + 1) == total:
            logging.info(f"  Cached {i+1:,}/{total:,} pairs")

    return cached


# ──────────────────────────────────────────────────────────────
# Step 3: ELO simulation (no model calls, uses cached predictions)
#         Mirrors run_llm_elo_ranking convergence logic exactly.
# ──────────────────────────────────────────────────────────────

def simulate_elo(n_items, cached_preds, norm_response, k_start, k_end,
                 convergence_patience=20, seed=42):
    """
    Simulate the ELO convergence loop using pre-cached predictions.
    Uses the same three convergence criteria as run_llm_elo_ranking:
      1. avg_change < 0.1
      2. percent_small > 0.95  (both sides changed < 1 point)
      3. correlation hasn't improved for convergence_patience rounds

    Returns (normalized ELO ratings array, rounds_run).
    """
    rng = random.Random(seed)
    ratings = [1500.0] * n_items

    best_corr = float('-inf')
    last_improved_round = 0
    round_num = 0

    while True:
        # Linear decay: decay over 30 rounds (same default as inference.py)
        decay_rounds = 30
        current_k = k_start - (k_start - k_end) * (round_num / decay_rounds)
        current_k = max(current_k, k_end)

        shuffled = cached_preds.copy()
        rng.shuffle(shuffled)

        changes = 0.0
        small_changes = 0
        n_pairs = len(shuffled)

        for pred in shuffled:
            i, j = pred["idx_a"], pred["idx_b"]
            winner = pred["winner"]

            if winner == "A":
                score_a, score_b = 1.0, 0.0
            elif winner == "B":
                score_a, score_b = 0.0, 1.0
            else:
                score_a, score_b = 0.5, 0.5

            expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
            expected_b = 1.0 - expected_a

            old_a, old_b = ratings[i], ratings[j]
            ratings[i] += current_k * (score_a - expected_a)
            ratings[j] += current_k * (score_b - expected_b)

            delta_a = abs(ratings[i] - old_a)
            delta_b = abs(ratings[j] - old_b)
            changes += delta_a + delta_b
            if delta_a < 1 and delta_b < 1:
                small_changes += 1

        # ── Convergence criterion 3: Pearson r patience ──
        ratings_arr = np.array(ratings)
        min_r, max_r = ratings_arr.min(), ratings_arr.max()
        if max_r > min_r:
            norm_elo = (ratings_arr - min_r) / (max_r - min_r)
        else:
            norm_elo = np.zeros(n_items)

        valid = ~np.isnan(norm_response)
        if valid.sum() >= 2:
            try:
                corr, _ = pearsonr(norm_response[valid], norm_elo[valid])
                if corr > best_corr:
                    best_corr = corr
                    last_improved_round = round_num
            except Exception:
                pass

        # ── Convergence criteria 1 & 2 ──
        avg_change = changes / (2 * n_pairs) if n_pairs > 0 else 0
        percent_small = small_changes / n_pairs if n_pairs > 0 else 0

        if (avg_change < 0.1
                or percent_small > 0.95
                or (round_num - last_improved_round >= convergence_patience)):
            break

        round_num += 1

    rounds_run = round_num + 1
    ratings_arr = np.array(ratings)
    min_r, max_r = ratings_arr.min(), ratings_arr.max()
    if max_r > min_r:
        return (ratings_arr - min_r) / (max_r - min_r), rounds_run
    return np.zeros(n_items), rounds_run


# ──────────────────────────────────────────────────────────────
# Step 4: Grid search
# ──────────────────────────────────────────────────────────────

def grid_search(items_df, cached_preds, k_starts, k_ends,
                convergence_patience=20, seed=42):
    """
    Grid search over k_start × k_end.
    Convergence is determined by the same criteria as run_llm_elo_ranking
    (no fixed max_rounds — stops when ratings stabilise or correlation plateaus).
    Returns a DataFrame of results sorted by Pearson r descending.
    """
    n_items = len(items_df)
    norm_response = items_df["norm_response"].values
    results = []

    grid = [(ks, ke) for ks, ke in itertools.product(k_starts, k_ends) if ke <= ks]
    logging.info(f"Grid search: {len(grid)} combinations (convergence_patience={convergence_patience})")

    for i, (k_start, k_end) in enumerate(grid):
        elo_norm, rounds_run = simulate_elo(
            n_items, cached_preds, norm_response,
            k_start, k_end,
            convergence_patience=convergence_patience,
            seed=seed
        )

        valid = ~np.isnan(norm_response)
        if valid.sum() < 2:
            continue

        try:
            r, _ = pearsonr(norm_response[valid], elo_norm[valid])
            r2 = r2_score(norm_response[valid], elo_norm[valid])
            mae = mean_absolute_error(norm_response[valid], elo_norm[valid])
            rmse = np.sqrt(mean_squared_error(norm_response[valid], elo_norm[valid]))
        except Exception:
            r, r2, mae, rmse = np.nan, np.nan, np.nan, np.nan

        results.append({
            "k_start": k_start,
            "k_end": k_end,
            "rounds_to_converge": rounds_run,
            "pearson_r": round(r, 5),
            "r_squared": round(r2, 5),
            "mae": round(mae, 5),
            "rmse": round(rmse, 5),
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(grid):
            logging.info(f"  {i+1}/{len(grid)} done | best r so far: "
                         f"{max(x['pearson_r'] for x in results):.4f}")

    results_df = pd.DataFrame(results).sort_values("pearson_r", ascending=False).reset_index(drop=True)
    return results_df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimize ELO k_start/k_end/max_rounds via grid search")
    parser.add_argument("--data-path", type=str, default="pairs_train.csv")
    parser.add_argument("--item", type=str, default="holes", help="Item to filter for optimization")
    parser.add_argument("--n-sample", type=int, default=2000,
                        help="Number of pairs to sample for inference caching (smaller = faster)")
    load_dotenv()
    _model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    _model_folder = _model_name.split("/")[-1]
    parser.add_argument("--base-model-path", type=str, default=_model_name)
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Adapter checkpoint path (default: auto-detect from .env)")
    parser.add_argument("--tokenizer-path", type=str, default=_model_name)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="elo_hyperparam_results.csv")
    args = parser.parse_args()

    # Auth
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    # ── Load model ──
    model, tokenizer, device = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        tokenizer_path=args.tokenizer_path,
    )
    if model is None:
        logging.error("Failed to load model. Exiting.")
        return

    # ── Load & filter pairs ──
    pairs_df = pd.read_csv(args.data_path)
    pairs_df = pairs_df[pairs_df["item"] == args.item].reset_index(drop=True)
    logging.info(f"item='{args.item}' → {len(pairs_df):,} pairs total")

    # Sample for speed
    if args.n_sample and len(pairs_df) > args.n_sample:
        pairs_df = pairs_df.sample(n=args.n_sample, random_state=args.seed).reset_index(drop=True)
        logging.info(f"Sampled {args.n_sample:,} pairs for inference caching")

    # ── Build items + pairs ──
    items_df, pair_list, _ = build_items_and_pairs(pairs_df)
    logging.info(f"{len(items_df):,} unique items | {len(pair_list):,} pairs")

    # ── Cache predictions (model runs here, once) ──
    t0 = time.time()
    cached_preds = cache_model_predictions(items_df, pair_list, model, tokenizer, device)
    logging.info(f"Inference done in {time.time()-t0:.1f}s")

    # Free GPU memory after inference
    import torch
    del model
    torch.cuda.empty_cache()
    logging.info("Model freed from GPU — running grid search in CPU memory")

    # ── Grid search ──
    k_starts = [4, 8, 16, 32, 64]
    k_ends   = [1, 2, 4, 8, 16]

    t1 = time.time()
    results_df = grid_search(
        items_df, cached_preds, k_starts, k_ends,
        convergence_patience=20, seed=args.seed
    )
    logging.info(f"Grid search done in {time.time()-t1:.1f}s")

    # ── Report ──
    print("\n" + "="*70)
    print(f"GRID SEARCH RESULTS — item='{args.item}' ({len(cached_preds):,} pairs)")
    print("="*70)
    print(results_df.head(20).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n✓ Best params:  k_start={best['k_start']}  k_end={best['k_end']}  (converged in {best['rounds_to_converge']:.0f} rounds)")
    print(f"  Pearson r={best['pearson_r']:.4f}  R²={best['r_squared']:.4f}  MAE={best['mae']:.4f}  RMSE={best['rmse']:.4f}")

    results_df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved all results to {args.output_csv}")

    print(f"\nTo use best params in inference.py, add these flags:")
    print(f"  (k_start / k_end are set inside run_llm_elo_ranking — update defaults there)")


if __name__ == "__main__":
    main()
