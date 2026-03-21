#!/usr/bin/env python
# coding: utf-8
"""
calculate_metrics.py  —  Resumable pairwise classification evaluation

Runs pairwise classification (A / B / Equal) on every row of
pairs_train/val/test/holdout.csv using the fine-tuned LLaMA classifier.

Key features
────────────
• **Real-time row-level saving**: Every row's prediction is appended
  immediately to a per-split CSV so no work is ever lost.
• **Automatic resume**: On restart the script detects how many rows
  have already been written for each split and picks up exactly where
  it left off — zero duplicated work.
• **Live checkpoint**: Every --checkpoint-interval rows (default 10 000)
  prints running accuracy, class counts, confusion matrix, rate, and ETA.
• **Split order**: train → test → val → holdout (override with --splits).

Output files
────────────
• classification_detail_{split}.csv  — one row per pair, ALL original
  columns + predicted_winner, confidence, correct.
• classification_metrics.csv  — one summary row per completed split.
"""

import argparse
import csv
import json
import os

from dotenv import load_dotenv
load_dotenv()
import sys
import time
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LABELS = ["A", "B", "Equal"]

SPLIT_FILES = {
    "train":   "pairs_train.csv",
    "test":    "pairs_test.csv",
    "val":     "pairs_val.csv",
    "holdout": "pairs_holdout.csv",
}

# ── prompt builder ────────────────────────────────────────────────────────────

def build_pair_prompt(row):
    task   = row.get("task",      "Unknown Task")
    prompt = row.get("prompt",    "")
    resp_a = row.get("response1", "")
    resp_b = row.get("response2", "")
    return f"{task}: {prompt}\nA: {resp_a}\nB: {resp_b}"

# ── metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=LABELS, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
    f1_per      = f1_score(y_true, y_pred, average=None,       labels=LABELS, zero_division=0)
    report      = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    conf_mat    = confusion_matrix(y_true, y_pred, labels=LABELS)
    return dict(accuracy=acc, f1_macro=f1_macro, f1_weighted=f1_weighted,
                f1_A=f1_per[0], f1_B=f1_per[1], f1_Equal=f1_per[2],
                report=report, conf_mat=conf_mat)


def print_checkpoint(split, processed, total, y_true_so_far, y_pred_so_far, elapsed):
    acc      = accuracy_score(y_true_so_far, y_pred_so_far)
    conf_mat = confusion_matrix(y_true_so_far, y_pred_so_far, labels=LABELS)
    pred_counts = {l: y_pred_so_far.count(l) for l in LABELS}
    true_counts = {l: y_true_so_far.count(l) for l in LABELS}
    pct  = 100.0 * processed / total
    rate = processed / elapsed if elapsed > 0 else 0
    eta  = (total - processed) / rate if rate > 0 else float("inf")

    print(f"\n{'─'*60}")
    print(f"  [{split.upper()}] CHECKPOINT  {processed:,}/{total:,}  ({pct:.1f}%)  "
          f"elapsed={elapsed:.0f}s  rate={rate:.0f}/s  ETA={eta:.0f}s")
    print(f"  Running Accuracy : {acc:.4f}")
    print(f"  Predicted counts : A={pred_counts['A']:,}  B={pred_counts['B']:,}  Equal={pred_counts['Equal']:,}")
    print(f"  True counts      : A={true_counts['A']:,}  B={true_counts['B']:,}  Equal={true_counts['Equal']:,}")
    print(f"  Confusion Matrix (rows=true, cols=pred)  [A / B / Equal]:")
    print(f"    {conf_mat}")
    print(f"{'─'*60}", flush=True)


def print_final_metrics(split, metrics, n):
    print(f"\n{'='*60}")
    print(f"  FINAL METRICS — {split.upper()}   (n={n:,})")
    print(f"{'='*60}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 macro      : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted   : {metrics['f1_weighted']:.4f}")
    print(f"  F1  A={metrics['f1_A']:.4f}  B={metrics['f1_B']:.4f}  Equal={metrics['f1_Equal']:.4f}")
    print(f"\n{metrics['report']}")
    print(f"  Confusion Matrix (rows=true, cols=pred)  [A / B / Equal]:")
    print(f"  {metrics['conf_mat']}", flush=True)

# ── resume helpers ────────────────────────────────────────────────────────────

def count_existing_rows(detail_path):
    """Return number of data rows already written (excluding header)."""
    if not os.path.exists(detail_path):
        return 0
    # Use a fast line count — subtract 1 for header
    with open(detail_path, "rb") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def load_existing_predictions(detail_path, n_rows):
    """Load the predicted_winner and winner columns from an existing partial file
    so running metrics include all previous rows."""
    if n_rows == 0:
        return [], []
    df = pd.read_csv(detail_path, usecols=["winner", "predicted_winner"], nrows=n_rows)
    return df["winner"].astype(str).tolist(), df["predicted_winner"].astype(str).tolist()

# ── main inference loop ───────────────────────────────────────────────────────

BATCH_SIZE = 64   # forward-pass batch size — tune up if GPU memory allows

def process_split(split, csv_path, model, tokenizer, device,
                  checkpoint_interval, detail_dir, model_tag=""):
    from inference import ask_llama_batch

    suffix = f"_{model_tag}" if model_tag else ""
    detail_path = os.path.join(detail_dir, f"classification_detail_{split}{suffix}.csv")

    # ── load source data ────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    total = len(df)

    if "winner" not in df.columns:
        logging.warning(f"[{split}] No 'winner' column — skipping.")
        return None

    df = df.reset_index(drop=True)
    out_columns = list(df.columns) + ["predicted_winner", "confidence", "correct"]

    # ── resume detection ────────────────────────────────────────────────────
    already_done = count_existing_rows(detail_path)
    if already_done >= total:
        logging.info(f"[{split}] Already complete ({already_done:,}/{total:,}) — skipping inference.")
        # Load existing predictions for summary metrics
        y_true, y_pred = load_existing_predictions(detail_path, total)
        metrics = compute_metrics(y_true, y_pred)
        print_final_metrics(split, metrics, total)
        return {
            "split": split, "n_samples": total,
            "accuracy": round(metrics["accuracy"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "f1_weighted": round(metrics["f1_weighted"], 4),
            "f1_A": round(metrics["f1_A"], 4),
            "f1_B": round(metrics["f1_B"], 4),
            "f1_Equal": round(metrics["f1_Equal"], 4),
            "n_correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
            "elapsed_s": 0,
        }

    if already_done > 0:
        logging.info(f"[{split}] RESUMING from row {already_done:,}/{total:,} "
                     f"({100.*already_done/total:.1f}% already done)")
        y_true, y_pred = load_existing_predictions(detail_path, already_done)
    else:
        y_true, y_pred = [], []

    print(f"\n{'#'*60}")
    print(f"  SPLIT: {split.upper()}   total={total:,}   "
          f"already_done={already_done:,}   remaining={total - already_done:,}")
    print(f"  File: {csv_path}")
    print(f"  Detail: {detail_path}")
    print(f"{'#'*60}", flush=True)

    # ── open detail CSV for append (create header if new) ───────────────────
    write_header = (already_done == 0)
    detail_fh = open(detail_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(detail_fh, fieldnames=out_columns,
                            extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
    if write_header:
        writer.writeheader()
        detail_fh.flush()

    # ── inference loop (batched) ─────────────────────────────────────────
    start_time = time.time()
    n_new = 0
    remaining = total - already_done

    for batch_start in range(already_done, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_df  = df.iloc[batch_start:batch_end]

        # Build prompts for the whole batch
        prompts = [build_pair_prompt(row) for _, row in batch_df.iterrows()]

        # Single forward pass for the batch
        results = ask_llama_batch(prompts, model, tokenizer, device)

        # Process each result in the batch
        for j, (idx, row) in enumerate(batch_df.iterrows()):
            winner     = results[j]["winner"]
            confidence = results[j]["confidence"]
            true_winner = str(row["winner"])
            correct     = int(winner == true_winner)

            y_true.append(true_winner)
            y_pred.append(winner)

            row_dict = row.to_dict()
            row_dict["predicted_winner"] = winner
            row_dict["confidence"]       = confidence
            row_dict["correct"]          = correct
            writer.writerow(row_dict)
            n_new += 1

        # Flush after every batch
        detail_fh.flush()

        # ── live checkpoint ─────────────────────────────────────────────────
        processed = batch_end  # total rows done (including resumed)
        if (processed // checkpoint_interval) != ((batch_start) // checkpoint_interval) \
                or processed == total:
            elapsed = time.time() - start_time
            print_checkpoint(split, processed, total, y_true, y_pred, elapsed)

    detail_fh.flush()
    detail_fh.close()

    # ── final metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)
    elapsed = time.time() - start_time
    print_final_metrics(split, metrics, total)
    print(f"  New rows this run: {n_new:,}   "
          f"Total time this run: {elapsed:.0f}s  ({elapsed/60:.1f} min)", flush=True)

    return {
        "split":       split,
        "n_samples":   total,
        "accuracy":    round(metrics["accuracy"],    4),
        "f1_macro":    round(metrics["f1_macro"],    4),
        "f1_weighted": round(metrics["f1_weighted"], 4),
        "f1_A":        round(metrics["f1_A"],        4),
        "f1_B":        round(metrics["f1_B"],        4),
        "f1_Equal":    round(metrics["f1_Equal"],    4),
        "n_correct":   sum(1 for t, p in zip(y_true, y_pred) if t == p),
        "elapsed_s":   round(elapsed, 1),
    }

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    # Derive model tag from .env MODEL_NAME (e.g. "Llama-2-7b-chat-hf")
    model_name_env = os.getenv("MODEL_NAME", "")
    model_tag = model_name_env.split("/")[-1] if model_name_env else ""

    parser = argparse.ArgumentParser(
        description="Resumable pairwise classification metrics over all splits."
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "test", "val", "holdout"],
        choices=list(SPLIT_FILES.keys()),
        help="Splits to evaluate in order (default: train test val holdout)."
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10_000,
        help="Print live metrics every N rows (default: 10000)."
    )
    default_output = f"classification_metrics_{model_tag}.csv" if model_tag else "classification_metrics.csv"
    parser.add_argument(
        "--output-csv", default=default_output,
        help=f"Summary CSV path (default: {default_output})."
    )
    parser.add_argument(
        "--detail-dir", default=".",
        help="Directory for per-split detail CSVs (default: current dir)."
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="Path to LoRA adapter (default: resolved from .env MODEL_NAME)."
    )
    args = parser.parse_args()

    os.makedirs(args.detail_dir, exist_ok=True)

    # ── load model once ──────────────────────────────────────────────────────
    from inference import load_model_and_tokenizer
    logging.info("Loading model — this may take a minute...")
    model, tokenizer, device = load_model_and_tokenizer(adapter_path=args.adapter_path)
    if model is None:
        sys.exit("Model failed to load. Aborting.")

    overall_start = time.time()
    summary_rows  = []

    for split in args.splits:
        csv_path = SPLIT_FILES[split]
        if not os.path.exists(csv_path):
            logging.warning(f"[{split}] {csv_path} not found — skipping.")
            continue

        summary = process_split(
            split, csv_path, model, tokenizer, device,
            args.checkpoint_interval, args.detail_dir, model_tag=model_tag
        )
        if summary:
            summary_rows.append(summary)
            # Save/update summary after each split so it's never lost
            pd.DataFrame(summary_rows).to_csv(args.output_csv, index=False)
            logging.info(f"[{split}] Summary updated → {args.output_csv}")

    # ── final summary ────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        total_elapsed = time.time() - overall_start
        print(f"\n{'='*60}")
        print(f"  SUMMARY — ALL SPLITS   (total time: {total_elapsed/60:.1f} min)")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved → {args.output_csv}")
        suffix = f"_{model_tag}" if model_tag else ""
        for split in args.splits:
            p = os.path.join(args.detail_dir, f"classification_detail_{split}{suffix}.csv")
            if os.path.exists(p):
                print(f"Detail saved  → {p}")
    else:
        logging.warning("No splits were evaluated.")


if __name__ == "__main__":
    main()