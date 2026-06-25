import pandas as pd
import json

df = pd.read_csv('data/processed/dataset.csv', usecols=['TIME', 'flare_now', 'flare_future_10min', 'flare_class'])
df_sorted = df.sort_values('TIME').reset_index(drop=True)

n = len(df_sorted)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

splits = {
    'train': df_sorted.iloc[:train_end],
    'val': df_sorted.iloc[train_end:val_end],
    'test': df_sorted.iloc[val_end:]
}

results = {}
for split_name, split_df in splits.items():
    results[split_name] = {
        'total_rows': len(split_df),
        'flare_now': split_df['flare_now'].value_counts().to_dict(),
        'flare_future_10min': split_df['flare_future_10min'].value_counts().to_dict(),
        'flare_class': split_df['flare_class'].value_counts().to_dict()
    }

print(json.dumps(results, indent=4))
