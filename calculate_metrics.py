import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

def calculate_metrics(preds, true):
    mae = mean_absolute_error(true, preds)
    mse = mean_squared_error(true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, preds)
    corr, _ = pearsonr(preds, true)
    return mae, mse, rmse, r2, corr

def main():
    splits = ['test', 'validation']

    for split in splits:
        file_path = f'{split}_output_sctt_results_LORA_10_epochs_Llama-2-7b-chat-hf.csv'
        try:
            df = pd.read_csv(file_path)
            preds = df['preds'].values
            true = df['ratings'].values

            mae, mse, rmse, r2, corr = calculate_metrics(preds, true)

            print(f"\n{split.upper()} SET METRICS:")
            print(f"  MAE:     {mae:.4f}")
            print(f"  MSE:     {mse:.4f}")
            print(f"  RMSE:    {rmse:.4f}")
            print(f"  R²:      {r2:.4f}")
            print(f"  Pearson: {corr:.4f}")
            print(f"  Samples: {len(df)}")

        except FileNotFoundError:
            print(f"\n{split.upper()} SET: File not found - {file_path}")
        except KeyError as e:
            print(f"\n{split.upper()} SET: Missing column - {e}")

if __name__ == "__main__":
    main()