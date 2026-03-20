#!/usr/bin/env python
"""
One-shot script: rebuild holdout cache + run ELO + generate summary.
Reuses existing val/test/train caches and ELO results.
"""
import argparse
import logging
import os
import sys
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Import what we need from elo_cached
from elo_cached import (
    build_cache,
    _load_model_once,
    _run_elo_for_dataset,
    DATASETS,
)

def main():
    # Use the same defaults as elo_cached.py
    class Args:
        batch_size = 64
        k_start = 32
        k_end = 4
        max_rounds = 500
        convergence_patience = 10
        decay_rounds = 200
        initial_elo = 1500
        min_r_delta = 0.01
        min_r_delta_rounds = 100
        seed = 42

    args = Args()

    # ── Phase 1: Build holdout cache ──
    data_file = "pairs_holdout.csv"
    cache_path = "predictions_cache_holdout.csv"

    if os.path.exists(cache_path):
        cached_rows = len(pd.read_csv(cache_path))
        total_rows = len(pd.read_csv(data_file))
        if cached_rows >= total_rows:
            logging.info(f"Holdout cache already complete ({cached_rows:,} rows). Skipping.")
        else:
            logging.info(f"Holdout cache has {cached_rows:,}/{total_rows:,} rows. Resuming.")
            model, tokenizer, device = _load_model_once()
            build_cache(data_file, cache_path, batch_size=args.batch_size,
                        model=model, tokenizer=tokenizer, device=device)
            del model, tokenizer
            torch.cuda.empty_cache()
    else:
        logging.info(f"Building holdout cache from {data_file} → {cache_path}")
        model, tokenizer, device = _load_model_once()
        build_cache(data_file, cache_path, batch_size=args.batch_size,
                    model=model, tokenizer=tokenizer, device=device)
        del model, tokenizer
        torch.cuda.empty_cache()

    logging.info("Model unloaded, GPU freed.")

    # ── Phase 2: Run ELO for holdout ──
    logging.info("Running ELO for holdout...")
    summaries_holdout, _ = _run_elo_for_dataset("holdout", cache_path, data_file, args)

    # ── Phase 3: Run ELO for val/test/train too (to collect summary) ──
    all_summaries = []
    for label, data_file in DATASETS:
        cp = f"predictions_cache_{label}.csv"
        dp = data_file
        if not os.path.exists(cp):
            logging.warning(f"[{label}] No cache {cp}, skipping.")
            continue
        if label == "holdout":
            all_summaries.extend(summaries_holdout)
            continue
        # Check if ELO result already exists
        elo_csv = f"13b_classification_elo_{label}.csv"
        if os.path.exists(elo_csv):
            logging.info(f"[{label}] ELO result already exists ({elo_csv}), re-running for summary...")
        summaries, _ = _run_elo_for_dataset(label, cp, dp, args)
        all_summaries.extend(summaries)

    # ── Final summary ──
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv("13b_classification_elo_summary.csv", index=False)

        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY — 13b Classification ELO")
        print(f"{'=' * 70}")
        print(summary_df.to_string(index=False))

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

    # Combined all-datasets file
    all_combined = []
    for label, _ in DATASETS:
        elo_csv = f"13b_classification_elo_{label}.csv"
        if os.path.exists(elo_csv):
            df = pd.read_csv(elo_csv)
            all_combined.append(df)
    if all_combined:
        mega = pd.concat(all_combined, ignore_index=True)
        mega.to_csv("13b_classification_elo_all_datasets.csv", index=False)
        print(f"  13b_classification_elo_all_datasets.csv ({len(mega):,} total rows)")


if __name__ == "__main__":
    main()
