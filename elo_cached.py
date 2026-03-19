#!/usr/bin/env python
# coding: utf-8
"""
Cached ELO Ranking
==================
Two-phase script for empirical ELO testing with a deterministic classifier.

Phase 1 — Build cache (runs model once):
    python elo_cached.py --build-cache --data-path inference_pairs_val_pairs.csv

Phase 2 — Run ELO from cache (instant, no GPU needed):
    python elo_cached.py --run-elo --cache predictions_cache_val.csv
    python elo_cached.py --run-elo --cache predictions_cache_val.csv --k-start 48 --k-end 4 --max-rounds 50
    python elo_cached.py --run-elo --cache predictions_cache_val.csv --item birds

Phase 3 — Sweep hyperparameters (instant, no GPU):
    python elo_cached.py --sweep --cache predictions_cache_val.csv
    python elo_cached.py --sweep --cache predictions_cache_val.csv --item birds

Since the model is deterministic (eval mode + argmax), each (A, B) pair always
produces the same winner.  We run inference once, cache every prediction, then
replay the ELO loop as many times as we like — varying K-factor schedules,
convergence rules, number of rounds, etc. — without touching the GPU again.
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def build_items_df(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct unique items from a pairs CSV."""
    items1 = pairs_df[["item", "task", "prompt", "response1", "norm_response1", "response1_id"]].rename(
        columns={"response1": "response", "norm_response1": "norm_response", "response1_id": "response_id"}
    )
    items2 = pairs_df[["item", "task", "prompt", "response2", "norm_response2", "response2_id"]].rename(
        columns={"response2": "response", "norm_response2": "norm_response", "response2_id": "response_id"}
    )
    items_df = pd.concat([items1, items2], ignore_index=True)
    items_df = items_df.drop_duplicates(subset=["response_id"]).reset_index(drop=True)
    return items_df


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Build prediction cache
# ──────────────────────────────────────────────────────────────────────

def _load_model_once():
    """Load model + tokenizer once and return (model, tokenizer, device)."""
    from inference import load_model_and_tokenizer
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    model, tokenizer, device = load_model_and_tokenizer()
    if model is None:
        raise RuntimeError("Failed to load model.")
    return model, tokenizer, device


def build_cache(data_path: str, cache_path: str, batch_size: int = 64,
                model=None, tokenizer=None, device=None):
    """
    Run batch inference on every pair, save predictions to CSV.
    If model/tokenizer/device are None, loads them (and frees GPU after).
    """
    from inference import (
        ask_llama_batch,
        build_comparison_prompt,
    )

    own_model = False
    if model is None:
        model, tokenizer, device = _load_model_once()
        own_model = True

    # Load pairs
    pairs_df = pd.read_csv(data_path)
    logging.info(f"Loaded {len(pairs_df):,} pairs from {data_path}")

    items_df = build_items_df(pairs_df)
    id_to_idx = {rid: idx for idx, rid in enumerate(items_df["response_id"])}
    logging.info(f"{len(items_df):,} unique items")

    # Resume support: skip already-cached rows
    start_row = 0
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        start_row = len(existing)
        logging.info(f"Resuming from row {start_row:,} ({cache_path} already has {start_row:,} rows)")

    # Build prompts for ALL pairs
    all_rows = []
    for _, row in pairs_df.iterrows():
        idx_a = id_to_idx.get(row["response1_id"])
        idx_b = id_to_idx.get(row["response2_id"])
        if idx_a is None or idx_b is None:
            continue
        row_a = items_df.iloc[idx_a].to_dict()
        row_b = items_df.iloc[idx_b].to_dict()
        prompt = build_comparison_prompt(row_a, row_b)
        all_rows.append({
            "response1_id": row["response1_id"],
            "response2_id": row["response2_id"],
            "idx_a": idx_a,
            "idx_b": idx_b,
            "item": row.get("item", ""),
            "prompt": prompt,
        })

    total = len(all_rows)
    logging.info(f"Total pairs to cache: {total:,} (starting at {start_row:,})")

    # Write header if starting fresh
    if start_row == 0:
        with open(cache_path, "w") as f:
            f.write("response1_id,response2_id,idx_a,idx_b,item,winner,confidence\n")

    # Batch inference with row-level append
    t0 = time.time()
    for batch_start in range(start_row, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = all_rows[batch_start:batch_end]
        prompts = [b["prompt"] for b in batch]

        results = ask_llama_batch(prompts, model, tokenizer, device)

        # Append row by row
        lines = []
        for b, r in zip(batch, results):
            lines.append(
                f"{b['response1_id']},{b['response2_id']},{b['idx_a']},{b['idx_b']},"
                f"{b['item']},{r['winner']},{r['confidence']}\n"
            )
        with open(cache_path, "a") as f:
            f.writelines(lines)

        done = batch_end
        elapsed = time.time() - t0
        rate = (done - start_row) / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        if done % (batch_size * 50) < batch_size or done == total:
            logging.info(
                f"  {done:,}/{total:,} ({done/total*100:.1f}%) "
                f"| {rate:.0f} pairs/s | ETA {eta/60:.1f}m"
            )

    elapsed = time.time() - t0
    logging.info(f"Cache complete: {total:,} predictions in {elapsed:.0f}s → {cache_path}")

    # Free GPU memory only if we loaded the model ourselves
    if own_model:
        import torch
        del model
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# Phase 2: ELO simulation from cache
# ──────────────────────────────────────────────────────────────────────

def elo_update(rating_a, rating_b, score_a, score_b, k):
    """Standard ELO update with tie support."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    return (
        rating_a + k * (score_a - expected_a),
        rating_b + k * (score_b - expected_b),
    )


def simulate_elo(
    n_items,
    cached_preds,
    norm_response,
    k_start=32,
    k_end=2,
    max_rounds=None,
    convergence_patience=20,
    decay_rounds=30,
    initial_elo=1500.0,
    min_r_delta=0.01,
    min_r_delta_rounds=100,
    seed=42,
    verbose=False,
):
    """
    Simulate ELO convergence using cached predictions (no model calls).

    Convergence criteria (same as inference.py):
      1. avg_change < 0.1
      2. percent_small > 0.95 (both sides changed < 1 point)
      3. Pearson r hasn't improved for convergence_patience rounds
      4. Pearson r improved < min_r_delta over the last min_r_delta_rounds

    Hard limit: max_rounds (default None → 200 safety cap).

    Returns: (normalized_elo_array, history_dict)
    """
    rng = random.Random(seed)
    ratings = [initial_elo] * n_items

    best_corr = float("-inf")
    last_improved_round = 0
    corr_history = []  # track r at each round for min-delta check
    effective_max = max_rounds if max_rounds is not None else 200  # safety cap

    round_history = []

    round_num = 0
    while round_num < effective_max:
        # Adaptive K: linear decay from k_start → k_end over decay_rounds
        current_k = k_start - (k_start - k_end) * (round_num / decay_rounds)
        current_k = max(current_k, k_end)

        shuffled = cached_preds.copy()
        rng.shuffle(shuffled)

        changes = 0.0
        small_changes = 0
        n_pairs = len(shuffled)
        winner_counts = {"A": 0, "B": 0, "Equal": 0}

        for pred in shuffled:
            i, j = pred["idx_a"], pred["idx_b"]
            winner = pred["winner"]
            winner_counts[winner] += 1

            if winner == "A":
                sa, sb = 1.0, 0.0
            elif winner == "B":
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            old_a, old_b = ratings[i], ratings[j]
            ratings[i], ratings[j] = elo_update(old_a, old_b, sa, sb, current_k)

            delta_a = abs(ratings[i] - old_a)
            delta_b = abs(ratings[j] - old_b)
            changes += delta_a + delta_b
            if delta_a < 1.0 and delta_b < 1.0:
                small_changes += 1

        # Compute current correlation
        ratings_arr = np.array(ratings)
        min_r, max_r = ratings_arr.min(), ratings_arr.max()
        if max_r > min_r:
            norm_elo = (ratings_arr - min_r) / (max_r - min_r)
        else:
            norm_elo = np.zeros(n_items)

        current_corr = np.nan
        valid = ~np.isnan(norm_response)
        if valid.sum() >= 2:
            try:
                current_corr, _ = pearsonr(norm_response[valid], norm_elo[valid])
                if current_corr > best_corr:
                    best_corr = current_corr
                    last_improved_round = round_num
                corr_history.append(current_corr)
            except Exception:
                corr_history.append(np.nan)
                pass

        avg_change = changes / (2 * n_pairs) if n_pairs > 0 else 0
        percent_small = small_changes / n_pairs if n_pairs > 0 else 0

        converge_reason = None
        if avg_change < 0.1:
            converge_reason = "avg_change<0.1"
        elif percent_small > 0.95:
            converge_reason = "95%_small"
        elif round_num - last_improved_round >= convergence_patience:
            converge_reason = f"patience_{convergence_patience}"
        elif len(corr_history) >= min_r_delta_rounds:
            r_old = corr_history[-min_r_delta_rounds]
            r_now = corr_history[-1]
            if not np.isnan(r_old) and not np.isnan(r_now) and (r_now - r_old) < min_r_delta:
                converge_reason = f"r_delta<{min_r_delta}_in_{min_r_delta_rounds}r"

        round_history.append({
            "round": round_num + 1,
            "k": round(current_k, 2),
            "pearson_r": round(current_corr, 5) if not np.isnan(current_corr) else None,
            "avg_change": round(avg_change, 4),
            "pct_small": round(percent_small, 4),
            "A": winner_counts["A"],
            "B": winner_counts["B"],
            "Equal": winner_counts["Equal"],
            "converge_reason": converge_reason,
        })

        if verbose:
            logging.info(
                f"  R{round_num+1} k={current_k:.1f} r={current_corr:.4f} "
                f"avg_Δ={avg_change:.3f} small={percent_small:.2%} "
                f"A/B/Eq={winner_counts['A']}/{winner_counts['B']}/{winner_counts['Equal']}"
                + (f" → {converge_reason}" if converge_reason else "")
            )

        if converge_reason:
            break

        round_num += 1

    # Final normalized ELO
    ratings_arr = np.array(ratings)
    min_r, max_r = ratings_arr.min(), ratings_arr.max()
    if max_r > min_r:
        norm_elo = (ratings_arr - min_r) / (max_r - min_r)
    else:
        norm_elo = np.zeros(n_items)

    return norm_elo, round_history


def compute_metrics(norm_response, norm_elo):
    """Compute all metrics between ground truth and ELO predictions."""
    valid = ~np.isnan(norm_response) & ~np.isnan(norm_elo)
    if valid.sum() < 2:
        return {"pearson_r": np.nan, "r_squared": np.nan, "mae": np.nan, "rmse": np.nan}

    y_true = norm_response[valid]
    y_pred = norm_elo[valid]
    r, p = pearsonr(y_true, y_pred)
    return {
        "pearson_r": round(r, 5),
        "p_value": p,
        "r_squared": round(r2_score(y_true, y_pred), 5),
        "mae": round(mean_absolute_error(y_true, y_pred), 5),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 5),
        "n": int(valid.sum()),
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Hyperparameter sweep
# ──────────────────────────────────────────────────────────────────────

DEFAULT_K_STARTS = [8, 16, 24, 32, 48, 64]
DEFAULT_K_ENDS = [1, 2, 4, 8]
DEFAULT_DECAY_ROUNDS = [10, 20, 30, 50]
DEFAULT_PATIENCE = [5, 10, 20, 30]


def sweep(
    items_df,
    cached_preds,
    k_starts=None,
    k_ends=None,
    decay_rounds_list=None,
    patience_list=None,
    seed=42,
):
    """
    Grid search over k_start × k_end × decay_rounds × convergence_patience.
    Returns a DataFrame of results sorted by Pearson r descending.
    """
    k_starts = k_starts or DEFAULT_K_STARTS
    k_ends = k_ends or DEFAULT_K_ENDS
    decay_rds = decay_rounds_list or DEFAULT_DECAY_ROUNDS
    patiences = patience_list or DEFAULT_PATIENCE

    n_items = len(items_df)
    norm_response = items_df["norm_response"].values

    # Filter valid combos (k_end <= k_start)
    grid = [
        (ks, ke, dr, p)
        for ks, ke, dr, p in itertools.product(k_starts, k_ends, decay_rds, patiences)
        if ke <= ks
    ]
    logging.info(f"Sweep: {len(grid)} hyperparameter combinations")

    results = []
    t0 = time.time()
    for i, (ks, ke, dr, pat) in enumerate(grid):
        norm_elo, history = simulate_elo(
            n_items, cached_preds, norm_response,
            k_start=ks, k_end=ke, decay_rounds=dr,
            convergence_patience=pat, seed=seed,
        )
        metrics = compute_metrics(norm_response, norm_elo)
        rounds_run = len(history)
        converge_reason = history[-1]["converge_reason"] if history else None

        results.append({
            "k_start": ks,
            "k_end": ke,
            "decay_rounds": dr,
            "patience": pat,
            "rounds_to_converge": rounds_run,
            "converge_reason": converge_reason,
            **metrics,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(grid):
            elapsed = time.time() - t0
            logging.info(f"  {i+1}/{len(grid)} ({elapsed:.1f}s)")

    df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────

def load_cache_and_items(cache_path, data_path, item_filter=None):
    """Load cache CSV + build items_df. Optionally filter to a single item."""
    cache_df = pd.read_csv(cache_path)
    pairs_df = pd.read_csv(data_path)

    if item_filter:
        cache_df = cache_df[cache_df["item"] == item_filter].reset_index(drop=True)
        pairs_df = pairs_df[pairs_df["item"] == item_filter].reset_index(drop=True)
        logging.info(f"Filtered to item='{item_filter}': {len(cache_df):,} pairs")

    items_df = build_items_df(pairs_df)
    id_to_idx = {rid: idx for idx, rid in enumerate(items_df["response_id"])}

    # Rebuild idx_a/idx_b relative to the filtered items_df
    cached_preds = []
    for _, row in cache_df.iterrows():
        idx_a = id_to_idx.get(row["response1_id"])
        idx_b = id_to_idx.get(row["response2_id"])
        if idx_a is not None and idx_b is not None:
            cached_preds.append({
                "idx_a": idx_a,
                "idx_b": idx_b,
                "winner": row["winner"],
            })

    logging.info(f"{len(items_df):,} items | {len(cached_preds):,} cached pairs")
    return items_df, cached_preds


# ──────────────────────────────────────────────────────────────────────
# Full pipeline: cache + ELO for all datasets
# ──────────────────────────────────────────────────────────────────────

# Dataset registry: (label, pairs_file)
DATASETS = [
    ("val",     "pairs_val.csv"),
    ("test",    "pairs_test.csv"),
    ("train",   "pairs_train.csv"),
    ("holdout", "pairs_holdout.csv"),
]


def _run_elo_for_dataset(label, cache_path, data_path, args):
    """
    Run ELO per-item for one dataset.  Returns (per-item summary rows, combined items_df).
    Saves one CSV: 13b_classification_elo_{label}.csv
    """
    cache_df = pd.read_csv(cache_path)
    pairs_df = pd.read_csv(data_path)
    all_items = sorted(cache_df["item"].dropna().unique())
    logging.info(f"[{label}] {len(cache_df):,} cached pairs | {len(all_items)} items: {all_items}")

    all_item_results = []
    summary_rows = []

    for item_name in all_items:
        items_df, cached_preds = load_cache_and_items(cache_path, data_path, item_filter=item_name)
        if len(cached_preds) == 0:
            logging.warning(f"[{label}] No cached pairs for item='{item_name}', skipping.")
            continue

        norm_elo, history = simulate_elo(
            n_items=len(items_df),
            cached_preds=cached_preds,
            norm_response=items_df["norm_response"].values,
            k_start=args.k_start,
            k_end=args.k_end,
            max_rounds=args.max_rounds,
            convergence_patience=args.convergence_patience,
            decay_rounds=args.decay_rounds,
            initial_elo=args.initial_elo,
            min_r_delta=args.min_r_delta,
            min_r_delta_rounds=args.min_r_delta_rounds,
            seed=args.seed,
            verbose=False,
        )

        items_df["Elo_normalized"] = norm_elo
        metrics = compute_metrics(items_df["norm_response"].values, norm_elo)
        rounds = len(history)
        reason = history[-1]["converge_reason"] if history else "—"

        logging.info(
            f"  [{label}] {item_name}: r={metrics['pearson_r']:.4f} "
            f"R²={metrics['r_squared']:.4f} rounds={rounds} ({reason})"
        )

        items_df["elo_item"] = item_name
        items_df["dataset"] = label
        all_item_results.append(items_df.copy())

        summary_rows.append({
            "dataset": label,
            "item": item_name,
            "n_items": len(items_df),
            "n_pairs": len(cached_preds),
            "rounds": rounds,
            "converge_reason": reason,
            **metrics,
        })

    # Save single CSV for this dataset
    if all_item_results:
        combined = pd.concat(all_item_results, ignore_index=True)
        out_csv = f"13b_classification_elo_{label}.csv"
        combined.to_csv(out_csv, index=False)
        logging.info(f"[{label}] Saved {out_csv} ({len(combined):,} rows)")
    else:
        combined = pd.DataFrame()

    return summary_rows, combined


def run_full_pipeline(args):
    """
    Full pipeline: load model once, cache predictions for all datasets,
    then run ELO per-item for each.  Saves per-dataset CSVs and a final summary.
    """
    import torch

    # ── Phase 1: Load model once, cache all datasets ──
    logging.info("=" * 60)
    logging.info("PHASE 1: Building prediction caches for all datasets")
    logging.info("=" * 60)

    model, tokenizer, device = _load_model_once()

    for label, data_file in DATASETS:
        if not os.path.exists(data_file):
            logging.warning(f"[{label}] {data_file} not found, skipping.")
            continue

        cache_path = f"predictions_cache_{label}.csv"

        # Check if cache already complete
        if os.path.exists(cache_path):
            cached_rows = sum(1 for _ in open(cache_path)) - 1  # minus header
            total_rows = sum(1 for _ in open(data_file)) - 1
            if cached_rows >= total_rows:
                logging.info(f"[{label}] Cache already complete ({cached_rows:,} rows). Skipping.")
                continue
            else:
                logging.info(f"[{label}] Cache has {cached_rows:,}/{total_rows:,} rows. Resuming.")

        logging.info(f"\n[{label}] Caching predictions from {data_file} → {cache_path}")
        build_cache(data_file, cache_path, batch_size=args.batch_size,
                     model=model, tokenizer=tokenizer, device=device)

    # Free GPU
    del model, tokenizer
    torch.cuda.empty_cache()
    logging.info("\nModel unloaded, GPU freed.")

    # ── Phase 2: Run ELO per-item for each dataset ──
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 2: Running ELO convergence per-item for all datasets")
    logging.info("=" * 60)

    all_summaries = []
    all_combined = []

    for label, data_file in DATASETS:
        cache_path = f"predictions_cache_{label}.csv"
        if not os.path.exists(cache_path):
            logging.warning(f"[{label}] No cache file {cache_path}, skipping ELO.")
            continue

        logging.info(f"\n{'─'*50}")
        logging.info(f"[{label}] Running ELO...")
        logging.info(f"{'─'*50}")

        summaries, combined = _run_elo_for_dataset(label, cache_path, data_file, args)
        all_summaries.extend(summaries)
        if len(combined) > 0:
            all_combined.append(combined)

    # ── Final summary CSV ──
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv("13b_classification_elo_summary.csv", index=False)

        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY — 13b Classification ELO")
        print(f"{'=' * 70}")
        print(summary_df.to_string(index=False))

        # Per-dataset aggregates
        print(f"\n{'─' * 70}")
        print("Per-dataset averages:")
        print(f"{'─' * 70}")
        agg = summary_df.groupby("dataset").agg({
            "pearson_r": "mean",
            "r_squared": "mean",
            "mae": "mean",
            "rmse": "mean",
            "rounds": "mean",
            "n_items": "sum",
            "n_pairs": "sum",
        }).round(4)
        print(agg.to_string())

        print(f"\nSaved: 13b_classification_elo_summary.csv")
        for label, _ in DATASETS:
            if os.path.exists(f"13b_classification_elo_{label}.csv"):
                print(f"  13b_classification_elo_{label}.csv")

    # Optional: combined all-datasets file
    if all_combined:
        mega = pd.concat(all_combined, ignore_index=True)
        mega.to_csv("13b_classification_elo_all_datasets.csv", index=False)
        print(f"  13b_classification_elo_all_datasets.csv ({len(mega):,} total rows)")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cached ELO ranking — build cache once, test hyperparameters instantly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-cache", action="store_true", help="Phase 1: run model, cache predictions")
    mode.add_argument("--run-elo", action="store_true", help="Phase 2: run ELO from cache")
    mode.add_argument("--run-all-items", action="store_true", help="Phase 2b: run ELO per-item for all items")
    mode.add_argument("--run-pipeline", action="store_true",
                       help="Full pipeline: cache + ELO for all datasets (val/test/train/holdout)")
    mode.add_argument("--sweep", action="store_true", help="Phase 3: grid search over hyperparameters")

    # Data / cache paths
    parser.add_argument("--data-path", type=str, default="inference_pairs_val_pairs.csv")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to cached predictions CSV (auto-named from data-path if omitted)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model inference (Phase 1)")
    parser.add_argument("--item", type=str, default=None, help="Filter to a single item (e.g. 'birds')")

    # ELO hyperparameters (Phase 2)
    parser.add_argument("--k-start", type=float, default=32)
    parser.add_argument("--k-end", type=float, default=2)
    parser.add_argument("--decay-rounds", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--convergence-patience", type=int, default=20)
    parser.add_argument("--min-r-delta", type=float, default=0.01,
                        help="Min Pearson r improvement required over --min-r-delta-rounds (default: 0.01)")
    parser.add_argument("--min-r-delta-rounds", type=int, default=100,
                        help="Window of rounds to check for --min-r-delta improvement (default: 100)")
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=42)

    # Sweep-specific (Phase 3)
    parser.add_argument("--k-starts", type=str, default=None,
                        help="Comma-separated k_start values (default: 8,16,24,32,48,64)")
    parser.add_argument("--k-ends", type=str, default=None,
                        help="Comma-separated k_end values (default: 1,2,4,8)")
    parser.add_argument("--decay-rounds-list", type=str, default=None,
                        help="Comma-separated decay_rounds values (default: 10,20,30,50)")
    parser.add_argument("--patience-list", type=str, default=None,
                        help="Comma-separated patience values (default: 5,10,20,30)")

    # Output
    parser.add_argument("--output-csv", type=str, default=None, help="Save results/sweep to CSV")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Auto-name cache file from data path
    if args.cache is None:
        base = os.path.splitext(os.path.basename(args.data_path))[0]
        args.cache = f"predictions_cache_{base}.csv"

    # ── Phase 1: Build cache ──
    if args.build_cache:
        build_cache(args.data_path, args.cache, batch_size=args.batch_size)
        return

    # ── Full pipeline: cache + ELO for all datasets ──
    if args.run_pipeline:
        run_full_pipeline(args)
        return

    # ── Phase 2: Single ELO run ──
    if args.run_elo:
        items_df, cached_preds = load_cache_and_items(args.cache, args.data_path, args.item)

        norm_elo, history = simulate_elo(
            n_items=len(items_df),
            cached_preds=cached_preds,
            norm_response=items_df["norm_response"].values,
            k_start=args.k_start,
            k_end=args.k_end,
            max_rounds=args.max_rounds,
            convergence_patience=args.convergence_patience,
            decay_rounds=args.decay_rounds,
            initial_elo=args.initial_elo,
            min_r_delta=args.min_r_delta,
            min_r_delta_rounds=args.min_r_delta_rounds,
            seed=args.seed,
            verbose=True,
        )

        # Compute final metrics
        metrics = compute_metrics(items_df["norm_response"].values, norm_elo)
        rounds = len(history)
        reason = history[-1]["converge_reason"] if history else "—"

        print(f"\n{'='*50}")
        print(f"ELO Results (k_start={args.k_start}, k_end={args.k_end}, "
              f"decay={args.decay_rounds}, patience={args.convergence_patience})")
        print(f"{'='*50}")
        print(f"Rounds:        {rounds} (stopped: {reason})")
        print(f"Pearson r:     {metrics['pearson_r']:.5f} (p={metrics.get('p_value', 0):.2e})")
        print(f"R²:            {metrics['r_squared']:.5f}")
        print(f"MAE:           {metrics['mae']:.5f}")
        print(f"RMSE:          {metrics['rmse']:.5f}")
        print(f"Items:         {metrics['n']}")

        # Attach ELO to items and show top/bottom 10
        items_df["Elo_normalized"] = norm_elo
        top = items_df.nlargest(10, "Elo_normalized")[["response_id", "response", "Elo_normalized", "norm_response"]]
        bottom = items_df.nsmallest(10, "Elo_normalized")[["response_id", "response", "Elo_normalized", "norm_response"]]
        print(f"\nTop 10:\n{top.to_string(index=False)}")
        print(f"\nBottom 10:\n{bottom.to_string(index=False)}")

        # Print round-by-round history
        print(f"\nRound-by-round convergence:")
        hist_df = pd.DataFrame(history)
        print(hist_df.to_string(index=False))

        # Save if requested
        if args.output_csv:
            items_df.to_csv(args.output_csv, index=False)
            logging.info(f"Saved item results to {args.output_csv}")

            # Also save round history
            hist_path = args.output_csv.replace(".csv", "_rounds.csv")
            hist_df.to_csv(hist_path, index=False)
            logging.info(f"Saved round history to {hist_path}")

        return

    # ── Phase 2b: Run ELO per-item for all items ──
    if args.run_all_items:
        cache_df = pd.read_csv(args.cache)
        pairs_df = pd.read_csv(args.data_path)
        all_items = sorted(cache_df["item"].dropna().unique())
        logging.info(f"Running ELO for {len(all_items)} items: {all_items}")

        all_item_results = []
        summary_rows = []

        for item_name in all_items:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing item: {item_name}")
            logging.info(f"{'='*50}")

            items_df, cached_preds = load_cache_and_items(args.cache, args.data_path, item_filter=item_name)
            if len(cached_preds) == 0:
                logging.warning(f"No cached pairs for item='{item_name}', skipping.")
                continue

            norm_elo, history = simulate_elo(
                n_items=len(items_df),
                cached_preds=cached_preds,
                norm_response=items_df["norm_response"].values,
                k_start=args.k_start,
                k_end=args.k_end,
                max_rounds=args.max_rounds,
                convergence_patience=args.convergence_patience,
                decay_rounds=args.decay_rounds,
                initial_elo=args.initial_elo,
                min_r_delta=args.min_r_delta,
                min_r_delta_rounds=args.min_r_delta_rounds,
                seed=args.seed,
                verbose=args.verbose,
            )

            items_df["Elo_normalized"] = norm_elo
            metrics = compute_metrics(items_df["norm_response"].values, norm_elo)
            rounds = len(history)
            reason = history[-1]["converge_reason"] if history else "—"

            logging.info(
                f"  {item_name}: r={metrics['pearson_r']:.4f} R²={metrics['r_squared']:.4f} "
                f"MAE={metrics['mae']:.4f} rounds={rounds} ({reason}) "
                f"n={metrics['n']} items, {len(cached_preds)} pairs"
            )

            # Save per-item CSV
            out_dir = args.output_csv or "elo_results_per_item"
            os.makedirs(out_dir, exist_ok=True)
            item_csv = os.path.join(out_dir, f"elo_{item_name}.csv")
            items_df.to_csv(item_csv, index=False)
            logging.info(f"  Saved {item_csv}")

            # Save per-item round history
            hist_csv = os.path.join(out_dir, f"elo_{item_name}_rounds.csv")
            pd.DataFrame(history).to_csv(hist_csv, index=False)

            # Collect for combined output
            items_df["elo_item"] = item_name
            all_item_results.append(items_df.copy())
            summary_rows.append({
                "item": item_name,
                "n_items": len(items_df),
                "n_pairs": len(cached_preds),
                "rounds": rounds,
                "converge_reason": reason,
                **metrics,
            })

        # Combined CSV with all items
        if all_item_results:
            combined = pd.concat(all_item_results, ignore_index=True)
            combined_path = os.path.join(out_dir, "elo_all_items_combined.csv")
            combined.to_csv(combined_path, index=False)
            logging.info(f"\nSaved combined results: {combined_path} ({len(combined):,} rows)")

            # Summary table
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(out_dir, "elo_summary.csv")
            summary_df.to_csv(summary_path, index=False)

            print(f"\n{'='*70}")
            print(f"Summary across all {len(all_items)} items")
            print(f"{'='*70}")
            print(summary_df.to_string(index=False))
            print(f"\nMean Pearson r: {summary_df['pearson_r'].mean():.4f}")
            print(f"Saved to: {out_dir}/")

        return

    # ── Phase 3: Sweep ──
    if args.sweep:
        items_df, cached_preds = load_cache_and_items(args.cache, args.data_path, args.item)

        k_starts = [float(x) for x in args.k_starts.split(",")] if args.k_starts else None
        k_ends = [float(x) for x in args.k_ends.split(",")] if args.k_ends else None
        decay_rds = [int(x) for x in args.decay_rounds_list.split(",")] if args.decay_rounds_list else None
        patiences = [int(x) for x in args.patience_list.split(",")] if args.patience_list else None

        sweep_df = sweep(
            items_df, cached_preds,
            k_starts=k_starts, k_ends=k_ends,
            decay_rounds_list=decay_rds, patience_list=patiences,
            seed=args.seed,
        )

        print(f"\n{'='*60}")
        print(f"Sweep Results — Top 20 by Pearson r")
        print(f"{'='*60}")
        print(sweep_df.head(20).to_string(index=False))

        print(f"\n{'='*60}")
        print(f"Bottom 10 by Pearson r")
        print(f"{'='*60}")
        print(sweep_df.tail(10).to_string(index=False))

        out = args.output_csv or "elo_sweep_results.csv"
        sweep_df.to_csv(out, index=False)
        logging.info(f"Saved sweep results ({len(sweep_df)} combos) to {out}")


if __name__ == "__main__":
    main()
