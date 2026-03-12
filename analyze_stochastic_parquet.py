#!/usr/bin/env python

import argparse
from pathlib import Path
import json
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pass-level stochastic parquet outputs for greedy and stochastic accuracy."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to pass-level parquet file. If omitted, auto-select latest *.parquet in repo root.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional output path to save summary metrics as JSON.",
    )
    return parser.parse_args()


def pick_default_parquet() -> Path:
    candidates = sorted(Path(".").glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No parquet files found in current directory. Pass --parquet explicitly.")
    return candidates[0]


def majority_vote_from_counts(counts: dict) -> tuple[str, bool]:
    ordered = ["A", "B", "Equal"]
    vals = {label: int(counts.get(label, 0)) for label in ordered}
    max_count = max(vals.values())
    winners = [label for label in ordered if vals[label] == max_count]
    return winners[0], len(winners) > 1


def compute_metrics_streaming(parquet_path: Path, batch_size: int = 500_000) -> dict:
    parquet_file = pq.ParquetFile(parquet_path)
    required_cols = [
        "source_row_index",
        "pass_type",
        "pass_index",
        "pred_winner",
        "true_winner",
        "is_correct",
    ]

    schema_names = set(parquet_file.schema.names)
    missing = [col for col in required_cols if col not in schema_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    total_rows = 0
    greedy_rows = 0
    stochastic_rows = 0
    greedy_correct = 0
    stochastic_correct = 0

    pass_total = defaultdict(int)
    pass_correct = defaultdict(int)

    sample_true = {}
    sample_greedy_count = defaultdict(int)
    sample_stochastic_count = defaultdict(int)
    sample_stochastic_votes = defaultdict(lambda: {"A": 0, "B": 0, "Equal": 0})

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_cols):
        batch_df = batch.to_pandas()
        total_rows += len(batch_df)

        greedy_df = batch_df[batch_df["pass_type"] == "greedy"]
        stochastic_df = batch_df[batch_df["pass_type"] == "stochastic"]

        if not greedy_df.empty:
            greedy_rows += len(greedy_df)
            greedy_correct += int(greedy_df["is_correct"].sum())

            greedy_counts = greedy_df.groupby("source_row_index").size()
            for sample_id, count in greedy_counts.items():
                sample_greedy_count[int(sample_id)] += int(count)

            greedy_true = greedy_df.groupby("source_row_index")["true_winner"].first()
            for sample_id, true_winner in greedy_true.items():
                sample_id = int(sample_id)
                if sample_id in sample_true and sample_true[sample_id] != true_winner:
                    raise ValueError(f"Inconsistent true_winner for sample {sample_id}")
                sample_true[sample_id] = true_winner

        if not stochastic_df.empty:
            stochastic_rows += len(stochastic_df)
            stochastic_correct += int(stochastic_df["is_correct"].sum())

            stochastic_counts = stochastic_df.groupby("source_row_index").size()
            for sample_id, count in stochastic_counts.items():
                sample_stochastic_count[int(sample_id)] += int(count)

            stochastic_true = stochastic_df.groupby("source_row_index")["true_winner"].first()
            for sample_id, true_winner in stochastic_true.items():
                sample_id = int(sample_id)
                if sample_id in sample_true and sample_true[sample_id] != true_winner:
                    raise ValueError(f"Inconsistent true_winner for sample {sample_id}")
                sample_true[sample_id] = true_winner

            per_pass = stochastic_df.groupby("pass_index")["is_correct"].agg(["sum", "count"])
            for pass_idx, row in per_pass.iterrows():
                pass_total[int(pass_idx)] += int(row["count"])
                pass_correct[int(pass_idx)] += int(row["sum"])

            vote_counts = stochastic_df.groupby(["source_row_index", "pred_winner"]).size()
            for (sample_id, pred_winner), count in vote_counts.items():
                sample_stochastic_votes[int(sample_id)][pred_winner] += int(count)

    total_samples = len(sample_true)
    greedy_accuracy = greedy_correct / greedy_rows if greedy_rows > 0 else float("nan")
    stochastic_row_accuracy = (
        stochastic_correct / stochastic_rows if stochastic_rows > 0 else float("nan")
    )

    majority_correct = 0
    majority_tie = 0
    for sample_id, true_winner in sample_true.items():
        vote_label, is_tie = majority_vote_from_counts(sample_stochastic_votes[sample_id])
        if vote_label == true_winner:
            majority_correct += 1
        if is_tie:
            majority_tie += 1

    stochastic_majority_accuracy = (
        majority_correct / total_samples if total_samples > 0 else float("nan")
    )
    majority_tie_rate = majority_tie / total_samples if total_samples > 0 else float("nan")

    stochastic_pass_accuracy = {
        int(pass_idx): (pass_correct[pass_idx] / pass_total[pass_idx])
        for pass_idx in sorted(pass_total.keys())
        if pass_total[pass_idx] > 0
    }

    greedy_counts_values = list(sample_greedy_count.values())
    stochastic_counts_values = list(sample_stochastic_count.values())
    valid_greedy_counts = all(val == 1 for val in greedy_counts_values) if greedy_counts_values else False
    inferred_stochastic_passes = (
        stochastic_counts_values[0]
        if stochastic_counts_values and all(v == stochastic_counts_values[0] for v in stochastic_counts_values)
        else None
    )
    valid_stochastic_counts = (
        inferred_stochastic_passes is not None and inferred_stochastic_passes > 0
    )

    expected_total_rows = (
        total_samples * (1 + inferred_stochastic_passes)
        if inferred_stochastic_passes is not None
        else None
    )

    return {
        "total_rows": int(total_rows),
        "total_samples": int(total_samples),
        "greedy_rows": int(greedy_rows),
        "stochastic_rows": int(stochastic_rows),
        "expected_total_rows": int(expected_total_rows) if expected_total_rows is not None else None,
        "greedy_accuracy": float(greedy_accuracy),
        "stochastic_row_accuracy": float(stochastic_row_accuracy),
        "stochastic_majority_accuracy": float(stochastic_majority_accuracy),
        "majority_tie_rate": float(majority_tie_rate),
        "stochastic_pass_accuracy": stochastic_pass_accuracy,
        "verify_greedy_count_per_sample": valid_greedy_counts,
        "verify_stochastic_count_per_sample": valid_stochastic_counts,
        "inferred_stochastic_passes": inferred_stochastic_passes,
    }


def print_summary(metrics: dict, parquet_path: Path) -> None:
    print("=" * 60)
    print("STOCHASTIC PARQUET ANALYSIS")
    print("=" * 60)
    print(f"File: {parquet_path}")
    print(f"Rows: {metrics['total_rows']:,}")
    print(f"Samples: {metrics['total_samples']:,}")
    print(f"Greedy rows: {metrics['greedy_rows']:,}")
    print(f"Stochastic rows: {metrics['stochastic_rows']:,}")
    if metrics["expected_total_rows"] is not None:
        print(f"Expected rows (from inferred pass count): {metrics['expected_total_rows']:,}")
    print("-" * 60)
    print(f"Greedy accuracy:              {metrics['greedy_accuracy']:.4f}")
    print(f"Stochastic row accuracy:      {metrics['stochastic_row_accuracy']:.4f}")
    print(f"Stochastic majority accuracy: {metrics['stochastic_majority_accuracy']:.4f}")
    print(f"Majority tie rate:            {metrics['majority_tie_rate']:.4f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet) if args.parquet else pick_default_parquet()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    metrics = compute_metrics_streaming(parquet_path)
    print_summary(metrics, parquet_path)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved JSON summary: {output_path}")


if __name__ == "__main__":
    main()
