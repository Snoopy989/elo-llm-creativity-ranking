#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


LABEL_MAP = {"A": 0, "B": 1, "Equal": 2}
ID2LABEL = {0: "A", 1: "B", 2: "Equal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one greedy + N stochastic classification passes per sample and save pass-level parquet."
        )
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="pairs_test.csv",
        help="Path to pairwise split CSV (default: pairs_test.csv)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="meta-llama/Llama-2-13b-hf",
        help="Base model path or HF repo id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="sctt_results_curriculum_10_epochs_Llama-2-13b-hf/phase_3/checkpoint-540000",
        help="Path to LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="meta-llama/Llama-2-13b-hf",
        help="Tokenizer source path/repo id.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Number of dataset rows to process; default uses all rows in split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model forward passes.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=180,
        help="Tokenizer max_length.",
    )
    parser.add_argument(
        "--num-stochastic-passes",
        type=int,
        default=60,
        help="Number of stochastic forward passes per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature applied to logits for stochastic sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and subset selection.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default=None,
        help="Output parquet path. If omitted, generated automatically.",
    )
    parser.add_argument(
        "--include-text-columns",
        action="store_true",
        help="Include task/prompt/response text columns in output parquet (large files).",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: str,
    tokenizer_path: str,
):
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    logging.info("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id

    logging.info("Loading adapter and merging...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logging.info("Model ready on %s", device)
    return model, tokenizer, device


def build_input_text(row: pd.Series) -> str:
    return (
        f"{row['task']}: {row['prompt']}\n"
        f"A: {row['response1']}\n"
        f"B: {row['response2']}"
    )


def prepare_dataset(split_path: str, max_samples: int, seed: int) -> pd.DataFrame:
    split = pd.read_csv(split_path)
    if max_samples is not None:
        sample_n = min(max_samples, len(split))
        split = split.sample(n=sample_n, random_state=seed)

    split = split.reset_index().rename(columns={"index": "source_row_index"})

    required_cols = ["task", "prompt", "response1", "response2", "winner"]
    missing = [col for col in required_cols if col not in split.columns]
    if missing:
        raise ValueError(f"Missing required columns in split: {missing}")

    return split


def add_pass_records(
    records: list,
    batch_df: pd.DataFrame,
    logits: torch.Tensor,
    probs: torch.Tensor,
    pred_ids: torch.Tensor,
    confidence: torch.Tensor,
    run_id: str,
    split_name: str,
    pass_type: str,
    pass_index: int,
    temperature: float,
    include_text_columns: bool,
):
    logits_np = logits.detach().cpu().float().numpy()
    probs_np = probs.detach().cpu().float().numpy()
    pred_ids_np = pred_ids.detach().cpu().numpy()
    conf_np = confidence.detach().cpu().float().numpy()

    for row_pos, (_, row) in enumerate(batch_df.iterrows()):
        pred_id = int(pred_ids_np[row_pos])
        pred_winner = ID2LABEL[pred_id]
        true_winner = row["winner"]

        record = {
            "run_id": run_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "split_name": split_name,
            "source_row_index": int(row["source_row_index"]),
            "response1_id": row.get("response1_id", None),
            "response2_id": row.get("response2_id", None),
            "true_winner": true_winner,
            "true_label_id": LABEL_MAP.get(true_winner, -1),
            "pass_type": pass_type,
            "pass_index": pass_index,
            "temperature": temperature,
            "stochastic_method": "temperature_logits",
            "logit_A": float(logits_np[row_pos, 0]),
            "logit_B": float(logits_np[row_pos, 1]),
            "logit_Equal": float(logits_np[row_pos, 2]),
            "prob_A": float(probs_np[row_pos, 0]),
            "prob_B": float(probs_np[row_pos, 1]),
            "prob_Equal": float(probs_np[row_pos, 2]),
            "prob_sum": float(probs_np[row_pos].sum()),
            "pred_label_id": pred_id,
            "pred_winner": pred_winner,
            "confidence": float(conf_np[row_pos]),
            "is_correct": bool(pred_winner == true_winner),
        }
        if include_text_columns:
            record["task"] = row.get("task", "")
            record["prompt"] = row.get("prompt", "")
            record["response1"] = row.get("response1", "")
            record["response2"] = row.get("response2", "")
        records.append(record)


def run_passes_to_parquet(
    data: pd.DataFrame,
    model,
    tokenizer,
    device,
    batch_size: int,
    max_length: int,
    num_stochastic_passes: int,
    temperature: float,
    seed: int,
    output_parquet: str,
    include_text_columns: bool,
) -> Dict[str, int]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    torch.manual_seed(seed)
    np.random.seed(seed)

    run_id = datetime.utcnow().strftime("stochastic_%Y%m%d_%H%M%S")
    split_name = Path(output_parquet).stem
    output_file = Path(output_parquet)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows_written = 0

    total_batches = (len(data) + batch_size - 1) // batch_size
    logging.info("Running passes for %s samples (%s batches)", len(data), total_batches)

    for batch_idx, start in enumerate(range(0, len(data), batch_size), start=1):
        batch_df = data.iloc[start : start + batch_size]
        texts = [build_input_text(row) for _, row in batch_df.iterrows()]

        inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            greedy_logits = model(**inputs).logits
        greedy_probs = torch.softmax(greedy_logits.float(), dim=-1)
        greedy_pred_ids = torch.argmax(greedy_probs, dim=-1)
        greedy_conf = greedy_probs.gather(1, greedy_pred_ids.unsqueeze(1)).squeeze(1)

        if not torch.isfinite(greedy_probs).all():
            raise ValueError("Non-finite probabilities detected in greedy pass")

        greedy_sum_diff = (greedy_probs.sum(dim=-1) - 1.0).abs().max().item()
        if greedy_sum_diff > 1e-3:
            raise ValueError(f"Greedy probability normalization drift too high: {greedy_sum_diff:.6f}")

        batch_records = []

        add_pass_records(
            records=batch_records,
            batch_df=batch_df,
            logits=greedy_logits,
            probs=greedy_probs,
            pred_ids=greedy_pred_ids,
            confidence=greedy_conf,
            run_id=run_id,
            split_name=split_name,
            pass_type="greedy",
            pass_index=0,
            temperature=1.0,
            include_text_columns=include_text_columns,
        )

        scaled_logits = greedy_logits.float() / temperature
        stochastic_probs = torch.softmax(scaled_logits, dim=-1)
        if not torch.isfinite(stochastic_probs).all():
            raise ValueError("Non-finite probabilities detected in stochastic pass probabilities")

        stochastic_sum_diff = (stochastic_probs.sum(dim=-1) - 1.0).abs().max().item()
        if stochastic_sum_diff > 1e-3:
            raise ValueError(f"Stochastic probability normalization drift too high: {stochastic_sum_diff:.6f}")

        for pass_index in range(1, num_stochastic_passes + 1):
            stochastic_pred_ids = torch.multinomial(stochastic_probs, num_samples=1).squeeze(1)
            stochastic_conf = stochastic_probs.gather(1, stochastic_pred_ids.unsqueeze(1)).squeeze(1)

            add_pass_records(
                records=batch_records,
                batch_df=batch_df,
                logits=greedy_logits,
                probs=stochastic_probs,
                pred_ids=stochastic_pred_ids,
                confidence=stochastic_conf,
                run_id=run_id,
                split_name=split_name,
                pass_type="stochastic",
                pass_index=pass_index,
                temperature=temperature,
                include_text_columns=include_text_columns,
            )

        batch_out = pd.DataFrame(batch_records)
        table = pa.Table.from_pandas(batch_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows_written += len(batch_out)

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            logging.info(
                "Processed batch %s/%s (rows written: %s)",
                batch_idx,
                total_batches,
                total_rows_written,
            )

    if writer is not None:
        writer.close()

    return {
        "rows_written": total_rows_written,
        "samples_processed": len(data),
    }


def verify_file_nonempty(output_path: str) -> int:
    table = pq.read_table(output_path, columns=["source_row_index"])
    num_rows = table.num_rows
    if num_rows == 0:
        raise ValueError("Parquet read-back check failed: file has zero rows")
    return num_rows


def main() -> None:
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    data = prepare_dataset(args.split_path, args.max_samples, args.seed)
    expected_samples = len(data)

    if args.output_parquet is None:
        split_stem = Path(args.split_path).stem
        args.output_parquet = (
            f"stochastic_{split_stem}_n{expected_samples}_passes{args.num_stochastic_passes}_"
            f"temp{args.temperature}.parquet"
        )

    model, tokenizer, device = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        tokenizer_path=args.tokenizer_path,
    )

    summary = run_passes_to_parquet(
        data=data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_stochastic_passes=args.num_stochastic_passes,
        temperature=args.temperature,
        seed=args.seed,
        output_parquet=args.output_parquet,
        include_text_columns=args.include_text_columns,
    )

    expected_rows = expected_samples * (args.num_stochastic_passes + 1)
    if summary["rows_written"] != expected_rows:
        raise ValueError(
            f"Row count mismatch during write: expected {expected_rows}, wrote {summary['rows_written']}"
        )

    on_disk_rows = verify_file_nonempty(args.output_parquet)
    if on_disk_rows != expected_rows:
        raise ValueError(
            f"Parquet row count mismatch on disk: expected {expected_rows}, got {on_disk_rows}"
        )

    logging.info("Verification checks passed")
    logging.info("Saved pass-level parquet: %s", args.output_parquet)
    logging.info("Rows: %s", on_disk_rows)


if __name__ == "__main__":
    main()
