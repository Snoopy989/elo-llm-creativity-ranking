#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import sys
from pathlib import Path
import numpy as np
from itertools import combinations
import warnings
import hashlib

def find_input_file():
    """Find the input CSV file."""
    input_file_options = [
        "all_sctt_jrt.csv",
        "../all_sctt_jrt.csv",
        "../../all_sctt_jrt.csv",
        os.path.join(os.getcwd(), "all_sctt_jrt.csv")
    ]

    for file_path in input_file_options:
        if os.path.exists(file_path):
            return file_path
    return None

def load_data(input_file):
    """Load and preprocess the data."""
    data = pd.read_csv(input_file)

    # Rename columns
    column_mapping = {'jrt': 'judge_response', 'se': 'standard_error', 'dist': 'distribution'}
    existing_mapping = {old: new for old, new in column_mapping.items() if old in data.columns}
    if existing_mapping:
        data.rename(columns=existing_mapping, inplace=True)

    print(f"Loaded {len(data)} rows from {input_file}")
    print(f"Columns: {data.columns.tolist()}")

    return data

def remove_duplicate_responses(df, strategy='keep_first'):
    """Remove duplicate responses to prevent data leakage."""
    original_count = len(df)
    response_counts = df['response'].value_counts()
    duplicates = response_counts[response_counts > 1]

    print(f"Before: {original_count:,} rows, {len(duplicates):,} duplicate response texts")

    if strategy == 'keep_first':
        df = df.drop_duplicates(subset=['response'], keep='first')
    elif strategy == 'keep_last':
        df = df.drop_duplicates(subset=['response'], keep='last')
    elif strategy == 'keep_highest_score':
        df = df.sort_values('judge_response', ascending=False).drop_duplicates(subset=['response'], keep='first').sort_index()

    print(f"After: {len(df):,} rows (removed {original_count - len(df):,})")
    return df

def create_individual_files(df, output_dir="grouped_data"):
    """Create individual CSV files for each group."""
    os.makedirs(output_dir, exist_ok=True)

    required_columns = ['item', 'task', 'prompt', 'judge_response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    grouped = df.groupby(['item', 'task', 'prompt'])
    files_created = 0

    for (item, task, prompt), group in grouped:
        group = group.copy()

        # Normalize judge_response
        if 'judge_response' in group.columns and len(group) > 1:
            min_val = group['judge_response'].min()
            max_val = group['judge_response'].max()
            if max_val != min_val:
                group['norm_response'] = (group['judge_response'] - min_val) / (max_val - min_val)
            else:
                group['norm_response'] = 0.5
            group['norm_response'] = group['norm_response'].round(2)
        else:
            group['norm_response'] = 0.5

        # Save to file
        safe_item = "".join(c for c in str(item) if c.isalnum() or c in "_-")
        safe_task = "".join(c for c in str(task) if c.isalnum() or c in "_-")
        safe_prompt = "".join(c for c in str(prompt) if c.isalnum() or c in "_-")
        filename = f"{safe_item}_{safe_task}_{safe_prompt}.csv"
        filepath = os.path.join(output_dir, filename)
        group.to_csv(filepath, index=False)
        files_created += 1

    print(f"Created {files_created} CSV files in {output_dir}/")
    return files_created

def create_pairs_dataset(df, train_ratio=0.8, test_ratio=0.2, random_seed=42, use_deterministic_split=True, save_split_assignments=True):
    """Create pairs dataset for training."""
    warnings.filterwarnings('ignore')
    np.random.seed(random_seed)

    pairs_output_file = "pairs_dataset.csv"
    split_assignments_file = "split_assignments.csv"

    all_pairs = []
    split_records = []

    required_columns = ['item', 'task', 'prompt', 'judge_response', 'response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    grouped = df.groupby(['item', 'task', 'prompt'])
    groups_processed = 0
    groups_skipped = 0

    def create_deterministic_hash(item, task, prompt, response_id):
        content = f"{item}_{task}_{prompt}_{response_id}"
        return int(hashlib.md5(content.encode()).hexdigest(), 16) % 100

    for (item, task, prompt), group in grouped:
        groups_processed += 1
        group = group.copy()

        # Normalize judge_response
        if 'judge_response' in group.columns:
            min_val = group['judge_response'].min()
            max_val = group['judge_response'].max()
            if max_val != min_val:
                group['norm_response'] = (group['judge_response'] - min_val) / (max_val - min_val)
            else:
                group['norm_response'] = 0.5
            group['norm_response'] = group['norm_response'].round(2)

        if len(group) < 2:
            groups_skipped += 1
            continue

        # Split responses using deterministic hash
        if use_deterministic_split:
            group['hash_value'] = group.apply(
                lambda row: create_deterministic_hash(item, task, prompt, row.get('ID', row.name)),
                axis=1
            )
            group = group.sort_values('hash_value').reset_index(drop=True)
            hash_threshold = int(train_ratio * 100)
            train_mask = group['hash_value'] < hash_threshold
            group.loc[train_mask, 'split'] = 'train'
            group.loc[~train_mask, 'split'] = 'test'

        # Record split assignments
        if save_split_assignments:
            for idx, row in group.iterrows():
                split_records.append({
                    'item': item,
                    'task': task,
                    'prompt': prompt,
                    'response_id': row.get('ID', idx),
                    'split': row.get('split', 'unknown'),
                    'norm_response': row.get('norm_response', 0),
                    'hash_value': row.get('hash_value', 'N/A') if use_deterministic_split else 'N/A'
                })

        # Generate pairs within each split
        for split in ['train', 'test']:
            split_group = group[group['split'] == split]
            if len(split_group) < 2:
                continue
            for (idx1, row1), (idx2, row2) in combinations(split_group.iterrows(), 2):
                winner = "A" if row1['norm_response'] > row2['norm_response'] else ("B" if row2['norm_response'] > row1['norm_response'] else "Equal")
                all_pairs.append({
                    'item': item,
                    'task': task,
                    'prompt': prompt,
                    'response1': row1.get('response', ''),
                    'response2': row2.get('response', ''),
                    'norm_response1': row1['norm_response'],
                    'norm_response2': row2['norm_response'],
                    'norm_diff': abs(row1['norm_response'] - row2['norm_response']),
                    'winner': winner,
                    'split': split
                })

    # Save results
    if save_split_assignments and split_records:
        pd.DataFrame(split_records).to_csv(split_assignments_file, index=False)

    if all_pairs:
        pairs_df = pd.DataFrame(all_pairs)
        pairs_df.to_csv(pairs_output_file, index=False)
        n_train = len(pairs_df[pairs_df['split'] == 'train'])
        n_test = len(pairs_df[pairs_df['split'] == 'test'])
        print(f"Saved {len(pairs_df):,} pairs to {pairs_output_file}")
        print(f"Train: {n_train:,} ({n_train/(n_train+n_test):.1%}), Test: {n_test:,} ({n_test/(n_train+n_test):.1%})")
        print(f"Winner distribution: {pairs_df['winner'].value_counts().to_dict()}")

        return pairs_df
    return None

def check_data_leakage(pairs_df):
    """Check for data leakage between train and test sets."""
    train_df = pairs_df[pairs_df['split'] == 'train']
    test_df = pairs_df[pairs_df['split'] == 'test']

    print(f"Dataset: {len(pairs_df):,} pairs ({len(train_df):,} train, {len(test_df):,} test)")

    # Check response-level leakage
    train_responses = set(train_df['response1'].unique()).union(set(train_df['response2'].unique()))
    test_responses = set(test_df['response1'].unique()).union(set(test_df['response2'].unique()))
    overlapping = train_responses.intersection(test_responses)

    print("Response leakage:")
    print(f"  Train responses: {len(train_responses):,}")
    print(f"  Test responses: {len(test_responses):,}")
    print(f"  Overlapping: {len(overlapping):,} ({len(overlapping)/len(train_responses.union(test_responses))*100:.1f}%)")

    if len(overlapping) > 0:
        test_affected = len(test_df[(test_df['response1'].isin(overlapping)) | (test_df['response2'].isin(overlapping))])
        print(f"  Test pairs affected: {test_affected:,} ({test_affected/len(test_df)*100:.1f}%)")
        return {'has_leakage': True, 'overlapping_count': len(overlapping)}
    else:
        print("  No leakage detected")
        return {'has_leakage': False, 'overlapping_count': 0}

def main():
    """Main function to run the data parsing pipeline."""
    input_file = find_input_file()
    if input_file is None:
        print("ERROR: Could not find 'all_sctt_jrt.csv'")
        sys.exit(1)

    df = load_data(input_file)
    df = remove_duplicate_responses(df)
    create_individual_files(df)
    pairs_df = create_pairs_dataset(df)

    if pairs_df is not None:
        leakage_report = check_data_leakage(pairs_df)
        print(f"Leakage report: {leakage_report}")

if __name__ == "__main__":
    main()

