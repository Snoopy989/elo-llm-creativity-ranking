import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('epoch_wise_curriculum_results.csv')

# Group by checkpoint and split to get unique metrics
metrics_summary = df.groupby(['peft_model_id', 'split']).agg({
    'accuracy': 'first',  # Same for all rows in group
    'pearson': 'first',
    'steps': 'first'
}).reset_index()

# Pivot to have validation and test side-by-side
val_metrics = metrics_summary[metrics_summary['split'] == 'validation'].copy()
val_metrics.columns = ['checkpoint', 'split', 'val_accuracy', 'val_pearson', 'steps']

test_metrics = metrics_summary[metrics_summary['split'] == 'test'].copy()
test_metrics.columns = ['checkpoint', 'split', 'test_accuracy', 'test_pearson', 'steps']

# Merge
combined = pd.merge(
    val_metrics[['checkpoint', 'steps', 'val_accuracy', 'val_pearson']], 
    test_metrics[['checkpoint', 'test_accuracy', 'test_pearson']], 
    on='checkpoint'
)

# Sort by validation accuracy
combined = combined.sort_values('val_accuracy', ascending=False)

print("Checkpoint Performance Summary")
print("=" * 100)
print(combined.to_string(index=False))
print("\n" + "=" * 100)

# Best checkpoint
best_val = combined.iloc[0]
print(f"\nBest checkpoint by validation accuracy:")
print(f"  Checkpoint: {best_val['checkpoint']}")
print(f"  Val Accuracy: {best_val['val_accuracy']:.4f}")
print(f"  Val Pearson: {best_val['val_pearson']:.4f}")
print(f"  Test Accuracy: {best_val['test_accuracy']:.4f}")
print(f"  Test Pearson: {best_val['test_pearson']:.4f}")

# Save summary
combined.to_csv('checkpoint_metrics_summary.csv', index=False)
print(f"\nSummary saved to checkpoint_metrics_summary.csv")
