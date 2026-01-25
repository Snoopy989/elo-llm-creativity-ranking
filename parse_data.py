#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import hashlib
import warnings
from pathlib import Path
from itertools import combinations
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Process and prepare LLM creativity ranking data."""
    
    def __init__(self, input_file: str = "all_sctt_jrt.csv", 
                 holdout_file: str = "sctt_item-generalization_jrt.csv"):
        self.input_file = input_file
        self.holdout_file = holdout_file
        self.scaler = MinMaxScaler()
        
    def load_and_clean(self, prefix: str = '', 
                       connector1: str = ': ', 
                       connector2: str = '\n') -> pd.DataFrame:
        """Load and clean the main dataset."""
        # Load main data
        data = pd.read_csv(self.input_file)
        df = data.copy()
        
        # Process duplicates
        df = self._handle_duplicates(df)
        
        # Create text field
        df['text'] = prefix + df['task'] + connector1 + df['prompt'] + connector2 + df['response'].str.lower()
        df['text'] = df['text'].astype(str)
        
        # Normalize JRT to label
        df['jrt'] = df['jrt'] + abs(df['jrt'].min())
        df['label'] = self.scaler.fit_transform(df['jrt'].to_numpy().reshape(-1, 1))
        
        # Select columns
        df = df[['ID', 'item', 'task', 'prompt', 'response', 'jrt', 'text', 'label']]
        df = df.dropna()
        
        # Save cleaned data
        df.to_csv('data_cleaned.csv', index=False)
        print(f"Saved {len(df):,} cleaned rows to data_cleaned.csv")
        
        # Process holdout set if exists
        self._process_holdout(prefix, connector1, connector2)
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate responses by averaging their scores."""
        dupes = df[df.duplicated(subset=['item', 'task', 'prompt', 'response'], keep=False)].copy()
        
        if len(dupes) > 0:
            dupes.to_csv('d_dupes.csv', index=False)
            dupes[['study_id', 'sub_id', 'response_id', 'ID']] = 'dupe_average'
            
            dupes_avg = dupes.groupby(['item', 'task', 'prompt', 'response'], as_index=False).agg({
                'study_id': 'first',
                'sub_id': 'first',
                'response_id': 'first',
                'ID': 'first',
                'jrt': 'mean'
            }).reset_index(drop=True)
            
            dupes_avg.to_csv('d_dupes_avg.csv', index=False)
            # Then it drops the original duplicates and adds the averaged ones
            df = df.drop(list(dupes.index), axis=0)
            df = pd.concat([df, dupes_avg]).reset_index(drop=True)
        
        return df
    
    def _process_holdout(self, prefix: str, connector1: str, connector2: str) -> None:
        """Process holdout generalization set if it exists."""
        if not os.path.exists(self.holdout_file):
            return
        
        gen = pd.read_csv(self.holdout_file)
        print(f"Loaded {len(gen):,} rows from {self.holdout_file}")
        
        gen['text'] = prefix + gen['task'] + connector1 + gen['prompt'] + connector2 + gen['response'].str.lower()
        gen['text'] = gen['text'].astype(str)
        
        scaler_gen = MinMaxScaler()
        gen['jrt'] = gen['jrt'] + abs(gen['jrt'].min())
        gen['label'] = scaler_gen.fit_transform(gen['jrt'].to_numpy().reshape(-1, 1))
        
        gen['ID'] = gen['study_id'].astype(str) + '-' + gen['response_id'].astype(str)
        gen = gen[['ID', 'item', 'task', 'prompt', 'response', 'text', 'label']]
        gen = gen.dropna()
        
        gen.to_csv('holdout_generalization_cleaned.csv', index=False)
        print(f"Saved {len(gen):,} holdout rows to holdout_generalization_cleaned.csv")


class StratifiedPairDatasetBuilder:
    """Build pairwise comparison datasets with stratified splits and curriculum learning."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        warnings.filterwarnings('ignore')
    
    def build(self, df: pd.DataFrame, 
              val_ratio: float = 0.10,
              test_ratio: float = 0.20,
              max_samples: Optional[int] = None,
              curriculum_strategy: str = 'progressive',
              save_split_assignments: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build stratified pairwise dataset with curriculum learning.
        
        Strategy:
        1. Split RESPONSES first (stratified by item/task/score)
        2. Generate pairs WITHIN each split
        3. Sample pairs if max_samples specified (stratified)
        4. Add difficulty scores to pairs
        5. Add curriculum phases to training pairs only
        
        Args:
            df: Cleaned response data
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            max_samples: Maximum total number of pairs (None = use all)
            curriculum_strategy: 'progressive', 'gradual', or 'mixed'
            save_split_assignments: Save response-level split assignments
        
        Returns:
            train_df, val_df, test_df with curriculum phases in train_df
        """
        
        self._validate_columns(df)
        
        print("\n" + "="*60)
        print("STRATIFIED PAIR DATASET BUILDER")
        print("="*60)
        
        # Step 1: Split responses (not pairs!)
        print("\nStep 1: Splitting responses into train/val/test...")
        train_responses, val_responses, test_responses = self._split_responses(
            df, 
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Step 2: Generate pairs within each split
        print("\nStep 2: Generating pairwise comparisons...")
        train_pairs = self._generate_all_pairs(train_responses)
        val_pairs = self._generate_all_pairs(val_responses)
        test_pairs = self._generate_all_pairs(test_responses)
        
        print(f"  Train pairs: {len(train_pairs):,}")
        print(f"  Val pairs:   {len(val_pairs):,}")
        print(f"  Test pairs:  {len(test_pairs):,}")
        total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
        print(f"  Total pairs: {total_pairs:,}")
        
        # Step 2.5: Sample if max_samples specified
        if max_samples is not None and total_pairs > max_samples:
            print(f"\nStep 2.5: Sampling {max_samples:,} pairs from {total_pairs:,}...")
            train_pairs, val_pairs, test_pairs = self._stratified_sample_pairs(
                train_pairs, val_pairs, test_pairs,
                max_samples=max_samples,
                train_ratio=1 - val_ratio - test_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
            print(f"  Sampled train pairs: {len(train_pairs):,}")
            print(f"  Sampled val pairs:   {len(val_pairs):,}")
            print(f"  Sampled test pairs:  {len(test_pairs):,}")
        
        # Step 3: Add difficulty scores to all pairs
        print("\nStep 3: Calculating difficulty scores...")
        train_pairs = self._add_difficulty_scores(train_pairs)
        val_pairs = self._add_difficulty_scores(val_pairs)
        test_pairs = self._add_difficulty_scores(test_pairs)
        
        train_pairs['split'] = 'train'
        val_pairs['split'] = 'val'
        test_pairs['split'] = 'test'
        
        self._print_difficulty_stats(train_pairs, 'Train')
        self._print_difficulty_stats(val_pairs, 'Val')
        self._print_difficulty_stats(test_pairs, 'Test')
        
        # Step 4: Add curriculum phases to training set only
        print("\nStep 4: Adding curriculum phases to training set...")
        train_pairs = self._add_curriculum_phases(train_pairs, strategy=curriculum_strategy)
        
        # Step 5: Verify no leakage and clean if needed
        print("\nStep 5: Verifying and removing data leakage...")
        train_pairs, val_pairs, test_pairs = self._verify_and_remove_leakage(
            train_responses, val_responses, test_responses,
            train_pairs, val_pairs, test_pairs
        )
        
        # Step 6: Save results
        print("\nStep 6: Saving datasets...")
        self._save_results(train_pairs, val_pairs, test_pairs, save_split_assignments)
        
        return train_pairs, val_pairs, test_pairs
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required = ['item', 'task', 'prompt', 'label', 'response']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    def _calculate_optimal_response_ratios(self, df: pd.DataFrame,
                                           target_val_ratio: float,
                                           target_test_ratio: float,
                                           tol: float = 1e-3) -> Tuple[float, float]:
        """
        Calculate optimal response-level split ratios to achieve target pair-level ratios.
        
        Since pairs = n*(n-1)/2, we need to solve for response ratios that yield
        the desired pair ratios across splits.
        """
        target_train_ratio = 1 - target_val_ratio - target_test_ratio
        
        def compute_pair_ratios(train_r: float, val_r: float):
            """Compute actual pair ratios given response ratios."""
            total_train_pairs = 0
            total_val_pairs = 0
            total_test_pairs = 0
            
            grouped = df.groupby(['item', 'task', 'prompt'])
            for (item, task, prompt), group in grouped:
                n = len(group)
                n_train = int(train_r * n)
                n_val = int(val_r * n)
                n_test = n - n_train - n_val
                
                total_train_pairs += n_train * (n_train - 1) // 2
                total_val_pairs += n_val * (n_val - 1) // 2
                total_test_pairs += n_test * (n_test - 1) // 2
            
            total = total_train_pairs + total_val_pairs + total_test_pairs
            if total == 0:
                return 0, 0, 0
            
            return (total_train_pairs / total, 
                    total_val_pairs / total, 
                    total_test_pairs / total)
        
        # Bisection search for optimal train ratio
        train_low, train_high = 0.4, 0.85
        for _ in range(30):
            train_mid = (train_low + train_high) / 2
            
            # For given train ratio, search for optimal val ratio
            val_low, val_high = 0.05, 0.25
            for _ in range(30):
                val_mid = (val_low + val_high) / 2
                
                actual_train, actual_val, actual_test = compute_pair_ratios(train_mid, val_mid)
                
                # Check if we're close enough to target
                if (abs(actual_train - target_train_ratio) < tol and 
                    abs(actual_val - target_val_ratio) < tol and
                    abs(actual_test - target_test_ratio) < tol):
                    print(f"    Optimal response ratios: Train {train_mid:.1%}, Val {val_mid:.1%}, Test {1-train_mid-val_mid:.1%}")
                    print(f"    Expected pair ratios: Train {actual_train:.1%}, Val {actual_val:.1%}, Test {actual_test:.1%}")
                    return train_mid, val_mid
                
                if actual_val < target_val_ratio:
                    val_low = val_mid
                else:
                    val_high = val_mid
            
            # Adjust train ratio based on how close we got
            actual_train, actual_val, actual_test = compute_pair_ratios(train_mid, val_mid)
            if actual_train < target_train_ratio:
                train_low = train_mid
            else:
                train_high = train_mid
        
        # Return best found values
        print(f"    Optimal response ratios: Train {train_mid:.1%}, Val {val_mid:.1%}, Test {1-train_mid-val_mid:.1%}")
        print(f"    Expected pair ratios: Train {actual_train:.1%}, Val {actual_val:.1%}, Test {actual_test:.1%}")
        return train_mid, val_mid
    
    def _split_responses(self, df: pd.DataFrame,
                         val_ratio: float,
                         test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split RESPONSES (not pairs) into train/val/test.
        Stratified by item × task × score_quartile.
        
        Calculates optimal response-level ratios to achieve desired pair-level ratios.
        """
        
        # Add score quartile for stratification
        df = df.copy()
        df['score_quartile'] = pd.qcut(df['label'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        
        # Create stratification key
        df['stratify_key'] = df['item'] + '_' + df['task'] + '_' + df['score_quartile'].astype(str)
        
        # Calculate optimal response-level ratios for desired pair-level ratios
        print(f"  Calculating optimal response splits for target pair ratios:")
        print(f"    Target: Train {(1-val_ratio-test_ratio)*100:.0f}%, Val {val_ratio*100:.0f}%, Test {test_ratio*100:.0f}%")
        
        optimal_train_ratio, optimal_val_ratio = self._calculate_optimal_response_ratios(
            df, 
            target_val_ratio=val_ratio,
            target_test_ratio=test_ratio
        )
        
        optimal_train_ratio, optimal_val_ratio = self._calculate_optimal_response_ratios(
            df, 
            target_val_ratio=val_ratio,
            target_test_ratio=test_ratio
        )
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=1 - optimal_train_ratio - optimal_val_ratio,
            stratify=df['stratify_key'],
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        val_size_adjusted = optimal_val_ratio / (optimal_train_ratio + optimal_val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['stratify_key'],
            random_state=self.random_seed
        )
        
        print(f"  Train responses: {len(train_df):,} ({len(train_df)/len(df):.1%})")
        print(f"  Val responses:   {len(val_df):,} ({len(val_df)/len(df):.1%})")
        print(f"  Test responses:  {len(test_df):,} ({len(test_df)/len(df):.1%})")
        
        # Verify stratification
        self._verify_response_stratification(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _verify_response_stratification(self, train_df: pd.DataFrame,
                                        val_df: pd.DataFrame,
                                        test_df: pd.DataFrame) -> None:
        """Verify response-level stratification is balanced."""
        print("\n  Response stratification verification:")
        
        # Check score quartile distribution
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            train_pct = (train_df['score_quartile'] == quartile).sum() / len(train_df) * 100
            val_pct = (val_df['score_quartile'] == quartile).sum() / len(val_df) * 100
            test_pct = (test_df['score_quartile'] == quartile).sum() / len(test_df) * 100
            print(f"    {quartile}: Train {train_pct:5.1f}% | Val {val_pct:5.1f}% | Test {test_pct:5.1f}%")
    
    def _stratified_sample_pairs(self, train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  max_samples: int,
                                  train_ratio: float,
                                  val_ratio: float,
                                  test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified sample from each split to reach max_samples total.
        Samples proportionally by item×task to maintain representation.
        """
        # Calculate target samples for each split
        n_train_target = int(max_samples * train_ratio)
        n_val_target = int(max_samples * val_ratio)
        n_test_target = max_samples - n_train_target - n_val_target  # Remainder
        
        def sample_split(df: pd.DataFrame, n_target: int) -> pd.DataFrame:
            """Stratified sample from a split by item×task."""
            if len(df) <= n_target:
                return df
            
            # Create stratification key
            df['sample_strat_key'] = df['item'] + '_' + df['task']
            
            # Sample proportionally from each group
            sampled_groups = []
            for key, group in df.groupby('sample_strat_key'):
                group_frac = len(group) / len(df)
                n_group = max(1, int(n_target * group_frac))
                n_group = min(n_group, len(group))  # Don't oversample
                
                sampled = group.sample(n=n_group, random_state=self.random_seed)
                sampled_groups.append(sampled)
            
            result = pd.concat(sampled_groups).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            
            # If we're still over, take exact amount
            if len(result) > n_target:
                result = result.sample(n=n_target, random_state=self.random_seed).reset_index(drop=True)
            
            result = result.drop(columns=['sample_strat_key'], errors='ignore')
            return result
        
        train_sampled = sample_split(train_df, n_train_target)
        val_sampled = sample_split(val_df, n_val_target)
        test_sampled = sample_split(test_df, n_test_target)
        
        return train_sampled, val_sampled, test_sampled
    
    def _verify_and_remove_leakage(self, train_responses: pd.DataFrame,
                                    val_responses: pd.DataFrame,
                                    test_responses: pd.DataFrame,
                                    train_pairs: pd.DataFrame,
                                    val_pairs: pd.DataFrame,
                                    test_pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Verify no response overlap between splits and remove if detected."""
        
        train_ids = set(train_responses['ID'].unique())
        val_ids = set(val_responses['ID'].unique())
        test_ids = set(test_responses['ID'].unique())
        
        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        
        print(f"  Train response IDs: {len(train_ids):,}")
        print(f"  Val response IDs:   {len(val_ids):,}")
        print(f"  Test response IDs:  {len(test_ids):,}")
        print(f"  Train-Val overlap:  {len(train_val_overlap):,}")
        print(f"  Train-Test overlap: {len(train_test_overlap):,}")
        print(f"  Val-Test overlap:   {len(val_test_overlap):,}")
        
        if any([train_val_overlap, train_test_overlap, val_test_overlap]):
            print("  ⚠ Overlap detected - removing pairs with overlapping responses...")
            
            original_val_len = len(val_pairs)
            original_test_len = len(test_pairs)
            
            # Remove val pairs containing responses that overlap with train
            if len(train_val_overlap) > 0:
                val_pairs = val_pairs[~((val_pairs['response1_id'].isin(train_val_overlap)) | 
                                       (val_pairs['response2_id'].isin(train_val_overlap)))]
                print(f"    Removed {original_val_len - len(val_pairs):,} val pairs (train overlap)")
            
            # Remove test pairs containing responses that overlap with train
            if len(train_test_overlap) > 0:
                test_pairs = test_pairs[~((test_pairs['response1_id'].isin(train_test_overlap)) | 
                                         (test_pairs['response2_id'].isin(train_test_overlap)))]
                print(f"    Removed {original_test_len - len(test_pairs):,} test pairs (train overlap)")
            
            # Remove test pairs containing responses that overlap with val
            if len(val_test_overlap) > 0:
                current_test_len = len(test_pairs)
                test_pairs = test_pairs[~((test_pairs['response1_id'].isin(val_test_overlap)) | 
                                         (test_pairs['response2_id'].isin(val_test_overlap)))]
                print(f"    Removed {current_test_len - len(test_pairs):,} test pairs (val overlap)")
            
            # Recheck for leakage
            print("\n  Rechecking after cleanup...")
            train_pair_ids = set(train_pairs['response1_id'].unique()).union(set(train_pairs['response2_id'].unique()))
            val_pair_ids = set(val_pairs['response1_id'].unique()).union(set(val_pairs['response2_id'].unique()))
            test_pair_ids = set(test_pairs['response1_id'].unique()).union(set(test_pairs['response2_id'].unique()))
            
            train_val_overlap_final = train_pair_ids.intersection(val_pair_ids)
            train_test_overlap_final = train_pair_ids.intersection(test_pair_ids)
            val_test_overlap_final = val_pair_ids.intersection(test_pair_ids)
            
            print(f"    Final train-val overlap:  {len(train_val_overlap_final):,}")
            print(f"    Final train-test overlap: {len(train_test_overlap_final):,}")
            print(f"    Final val-test overlap:   {len(val_test_overlap_final):,}")
            
            if not any([train_val_overlap_final, train_test_overlap_final, val_test_overlap_final]):
                print("    ✓ No leakage detected after cleanup!")
            else:
                print("    ⚠ WARNING: Some overlap remains after cleanup!")
        else:
            print("  ✓ No response leakage detected - all clear!")
        
        return train_pairs, val_pairs, test_pairs
    
    def _generate_all_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all pairwise comparisons."""
        all_pairs = []
        
        for (item, task, prompt), group in df.groupby(['item', 'task', 'prompt']):
            if len(group) < 2:
                continue
            
            for (_, row1), (_, row2) in combinations(group.iterrows(), 2):
                winner = "A" if row1['label'] > row2['label'] else ("B" if row2['label'] > row1['label'] else "Equal")
                
                all_pairs.append({
                    'item': item,
                    'task': task,
                    'prompt': prompt,
                    'response1': row1.get('response', ''),
                    'response2': row2.get('response', ''),
                    'response1_id': row1.get('ID', ''),
                    'response2_id': row2.get('ID', ''),
                    'norm_response1': row1['label'],
                    'norm_response2': row2['label'],
                    'winner': winner,
                })
        
        return pd.DataFrame(all_pairs)
    
    def _add_difficulty_scores(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Add difficulty metrics based on score difference."""
        df = pairs_df.copy()
        
        # Calculate absolute score difference
        df['norm_diff'] = abs(df['norm_response1'] - df['norm_response2'])
        
        # Difficulty score: 0=easy (large diff), 1=hard (small diff)
        df['difficulty_score'] = 1 - df['norm_diff']
        
        # Categorize by quartiles
        q25 = df['norm_diff'].quantile(0.25)
        q50 = df['norm_diff'].quantile(0.50)
        q75 = df['norm_diff'].quantile(0.75)
        
        def categorize(diff):
            if diff >= q75:
                return 'Easy'
            elif diff >= q50:
                return 'Medium-Easy'
            elif diff >= q25:
                return 'Medium-Hard'
            else:
                return 'Hard'
        
        df['difficulty_category'] = df['norm_diff'].apply(categorize)
        
        # Numerical difficulty level (1=easiest, 4=hardest)
        category_to_level = {
            'Easy': 1,
            'Medium-Easy': 2,
            'Medium-Hard': 3,
            'Hard': 4
        }
        df['difficulty_level'] = df['difficulty_category'].map(category_to_level)
        
        # Create stratification key: difficulty_quartile + group
        df['stratify_key'] = df['difficulty_category'] + '_' + df['item'] + '_' + df['task']
        
        return df
    
    def _add_curriculum_phases(self, train_df: pd.DataFrame, 
                                strategy: str = 'progressive') -> pd.DataFrame:
        """Add curriculum learning phases to training set."""
        df = train_df.copy()
        
        if strategy == 'progressive':
            # Phase 1: Easy only
            # Phase 2: Easy + Medium-Easy + Medium-Hard  
            # Phase 3: All data
            def assign_phase(row):
                if row['difficulty_level'] == 1:  # Easy
                    return 1
                elif row['difficulty_level'] in [2, 3]:  # Medium
                    return 2
                else:  # Hard
                    return 3
            
            df['curriculum_phase'] = df.apply(assign_phase, axis=1)
            
        elif strategy == 'gradual':
            # Smooth transition based on difficulty score
            df['curriculum_phase'] = pd.cut(
                df['difficulty_score'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=[1, 2, 3]
            ).astype(int)
        
        elif strategy == 'mixed':
            # All phases use all data with different sampling weights
            df['curriculum_phase'] = 1
            df['phase1_weight'] = np.where(df['difficulty_level'] == 1, 3, 1)
            df['phase2_weight'] = np.where(df['difficulty_level'] <= 2, 2, 1)
            df['phase3_weight'] = 1
        
        # Print phase distribution
        for phase in sorted(df['curriculum_phase'].unique()):
            count = (df['curriculum_phase'] == phase).sum()
            pct = count / len(df) * 100
            print(f"    Phase {phase}: {count:,} pairs ({pct:.1f}%)")
        
        return df
    
    def _save_results(self, train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      save_split_assignments: bool) -> None:
        """Save datasets to CSV files."""
        
        # Save complete dataset
        all_pairs = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        all_pairs.to_csv('pairs_dataset.csv', index=False)
        
        # Save split-specific files
        train_df.to_csv('pairs_train.csv', index=False)
        val_df.to_csv('pairs_val.csv', index=False)
        test_df.to_csv('pairs_test.csv', index=False)
        
        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)
        total = n_train + n_val + n_test
        
        print(f"\n  ✓ Saved {total:,} pairs to pairs_dataset.csv")
        print(f"  ✓ Train: {n_train:,} ({n_train/total:.1%}) → pairs_train.csv")
        print(f"  ✓ Val:   {n_val:,} ({n_val/total:.1%}) → pairs_val.csv")
        print(f"  ✓ Test:  {n_test:,} ({n_test/total:.1%}) → pairs_test.csv")
        print(f"\n  Winner distribution: {all_pairs['winner'].value_counts().to_dict()}")
    
    def _print_difficulty_stats(self, pairs_df: pd.DataFrame, split_name: str = '') -> None:
        """Print difficulty distribution statistics for a split."""
        prefix = f"  {split_name} difficulty distribution:" if split_name else "  Difficulty categories:"
        print(prefix)
        for cat in ['Easy', 'Medium-Easy', 'Medium-Hard', 'Hard']:
            count = (pairs_df['difficulty_category'] == cat).sum()
            pct = count / len(pairs_df) * 100 if len(pairs_df) > 0 else 0
            equal_pct = (pairs_df[pairs_df['difficulty_category'] == cat]['winner'] == 'Equal').sum() / count * 100 if count > 0 else 0
            print(f"    {cat:12s}: {count:,} pairs ({pct:5.1f}%) - {equal_pct:4.1f}% Equal")


def main():
    """Main execution function."""
    print("=" * 60)
    print("DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and clean data
    processor = DataProcessor()
    cleaned_data = processor.load_and_clean()
    
    # Step 2: Build stratified pairs dataset with curriculum
    builder = StratifiedPairDatasetBuilder(random_seed=42)
    train_df, val_df, test_df = builder.build(
        cleaned_data,
        val_ratio=0.10,
        test_ratio=0.20,
        max_samples=50000,  # Set to e.g., 40000 to limit total pairs
        curriculum_strategy='progressive'
    )
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()