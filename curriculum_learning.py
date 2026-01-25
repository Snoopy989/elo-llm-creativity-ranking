#!/usr/bin/env python
# coding: utf-8

"""
Curriculum Learning for Pairwise Classification

Trains on EASY examples first, then gradually introduces HARDER examples.


Key Insight:
- Easy pairs: Large score difference (0.9 vs 0.1) → Clear winner
- Hard pairs: Small score difference (0.52 vs 0.48) → Ambiguous, could be Equal
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_difficulty_scores(pairs_df):
    """
    Add difficulty metrics to pairs dataset.
    
    Difficulty is based on normalized score difference:
    - Large difference (0.8+) = EASY (clear winner)
    - Medium difference (0.3-0.8) = MEDIUM  
    - Small difference (0.0-0.3) = HARD (ambiguous, often Equal)
    
    Args:
        pairs_df: DataFrame with norm_response1, norm_response2, winner columns
    
    Returns:
        DataFrame with added difficulty columns
    """
    df = pairs_df.copy()
    
    # Calculate absolute score difference
    df['norm_diff'] = abs(df['norm_response1'] - df['norm_response2'])
    
    # Difficulty score: 0=easy, 1=hard
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
    
    return df


def create_curriculum_splits(pairs_df, strategy='progressive'):
    """
    Create curriculum learning splits.
    
    Strategies:
    - 'progressive': Phase 1 (easy) → Phase 2 (easy+medium) → Phase 3 (all)
    - 'gradual': Smoothly transition from easy to hard over epochs
    - 'mixed': Each batch has mostly easy + some hard (anti-forgetting)
    
    Args:
        pairs_df: DataFrame with difficulty scores
        strategy: Curriculum strategy
    
    Returns:
        DataFrame with curriculum_phase column
    """
    df = pairs_df.copy()
    
    if strategy == 'progressive':
        # Phase 1: Easy only
        # Phase 2: Easy + Medium-Easy + Medium-Hard  
        # Phase 3: All data
        def assign_phase(row):
            if row['difficulty_level'] == 1:  # Easy
                return 1  # Available from phase 1
            elif row['difficulty_level'] in [2, 3]:  # Medium
                return 2  # Available from phase 2
            else:  # Hard
                return 3  # Available from phase 3
        
        df['curriculum_phase'] = df.apply(assign_phase, axis=1)
        
    elif strategy == 'gradual':
        # Smooth transition based on difficulty score
        # Easier examples available earlier
        df['curriculum_phase'] = pd.cut(
            df['difficulty_score'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=[1, 2, 3]
        ).astype(int)
    
    elif strategy == 'mixed':
        # All phases use all data, but with different sampling weights
        df['curriculum_phase'] = 1  # All data available from start
        df['phase1_weight'] = np.where(df['difficulty_level'] == 1, 3, 1)  # Oversample easy
        df['phase2_weight'] = np.where(df['difficulty_level'] <= 2, 2, 1)  # Balanced
        df['phase3_weight'] = 1  # Uniform
    
    return df


def get_curriculum_subset(df, current_epoch, total_epochs=10, strategy='progressive'):
    """
    Get training subset for current epoch based on curriculum strategy.
    
    Args:
        df: Full dataset with curriculum_phase column
        current_epoch: Current epoch (1-indexed)
        total_epochs: Total training epochs
        strategy: Curriculum strategy
    
    Returns:
        Subset of data to train on for this epoch
    """
    if strategy == 'progressive':
        # Map epochs to curriculum phases
        if current_epoch <= total_epochs * 0.4:  # First 40%
            max_phase = 1  # Easy only
        elif current_epoch <= total_epochs * 0.7:  # Next 30%
            max_phase = 2  # Easy + Medium
        else:  # Final 30%
            max_phase = 3  # All data
        
        return df[df['curriculum_phase'] <= max_phase]
    
    elif strategy == 'gradual':
        # Gradually increase max difficulty
        max_difficulty = (current_epoch / total_epochs) * df['difficulty_score'].max()
        return df[df['difficulty_score'] <= max_difficulty]
    
    elif strategy == 'mixed':
        # Use sampling weights (implement in DataLoader)
        return df  # Return all, but with sampling weights
    
    else:
        return df  # No curriculum


def print_curriculum_stats(df):
    """Print statistics about curriculum distribution."""
    print("="*60)
    print("CURRICULUM LEARNING STATISTICS")
    print("="*60)
    
    print(f"\nTotal pairs: {len(df):,}")
    
    print("\n--- Difficulty Distribution ---")
    print(df['difficulty_category'].value_counts().sort_index())
    
    print("\n--- Score Difference Stats ---")
    print(f"Mean:   {df['norm_diff'].mean():.3f}")
    print(f"Median: {df['norm_diff'].median():.3f}")
    print(f"Q25:    {df['norm_diff'].quantile(0.25):.3f}")
    print(f"Q75:    {df['norm_diff'].quantile(0.75):.3f}")
    
    if 'curriculum_phase' in df.columns:
        print("\n--- Curriculum Phase Distribution ---")
        for phase in sorted(df['curriculum_phase'].unique()):
            count = (df['curriculum_phase'] == phase).sum()
            pct = count / len(df) * 100
            print(f"Phase {phase}: {count:,} pairs ({pct:.1f}%)")
    
    print("\n--- Winner Distribution by Difficulty ---")
    for cat in ['Easy', 'Medium-Easy', 'Medium-Hard', 'Hard']:
        if cat in df['difficulty_category'].values:
            subset = df[df['difficulty_category'] == cat]
            equal_pct = (subset['winner'] == 'Equal').sum() / len(subset) * 100
            print(f"{cat:12s}: {equal_pct:5.1f}% Equal")


# ============================================
# EXAMPLE USAGE
# ============================================

def main():
    """Demonstrate curriculum learning setup."""
    
    print("Loading training pairs dataset...")
    
    # Load training pairs only (curriculum learning only applies to training)
    pairs_file = Path('pairs_train.csv')
    if pairs_file.exists():
        df = pd.read_csv(pairs_file)
        print(f"✓ Loaded {len(df):,} training pairs from {pairs_file}")
    else:
        print("⚠ pairs_train.csv not found")
        print("Run parse_data.py first to create pairs dataset")
        return
    
    # Add difficulty scores
    print("\nCalculating difficulty scores...")
    df = add_difficulty_scores(df)
    
    # Create curriculum splits
    print("\nCreating curriculum splits (progressive strategy)...")
    df = create_curriculum_splits(df, strategy='progressive')
    
    # Print statistics
    print_curriculum_stats(df)
    
    # Simulate training phases
    print("\n" + "="*60)
    print("SIMULATED TRAINING PHASES")
    print("="*60)
    
    total_epochs = 10
    for epoch in range(1, total_epochs + 1):
        subset = get_curriculum_subset(df, epoch, total_epochs, strategy='progressive')
        print(f"\nEpoch {epoch}:")
        print(f"  Training on {len(subset):,} pairs ({len(subset)/len(df)*100:.1f}%)")
        
        # Show difficulty distribution
        diff_counts = subset['difficulty_category'].value_counts()
        diff_levels = subset['difficulty_level'].unique()
        if len(diff_levels) == 1:
            print(f"  Difficulty: {subset['difficulty_category'].iloc[0]} only")
        else:
            min_level = subset['difficulty_level'].min()
            max_level = subset['difficulty_level'].max()
            level_map = {1: 'Easy', 2: 'Medium-Easy', 3: 'Medium-Hard', 4: 'Hard'}
            print(f"  Difficulty range: {level_map[min_level]} to {level_map[max_level]}")
        
        equal_pct = (subset['winner'] == 'Equal').sum() / len(subset) * 100
        print(f"  Equal pairs: {equal_pct:.1f}%")
    
    # Save enhanced dataset
    output_file = 'pairs_train_curriculum.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved curriculum training dataset to {output_file}")
    

if __name__ == "__main__":
    main()
