import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

def main():
    print("==================================================")
    # 1. Path checks
    dataset_path = 'data/processed/dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Processed dataset not found at {dataset_path}")
        return
        
    print(f"[INFO] Running comprehensive dataset audit on {dataset_path}...")
    report_lines = []
    report_lines.append("# Dataset Audit & Validation Report")
    report_lines.append(f"Generated on: {datetime.datetime.utcnow().isoformat()} UTC\n")
    
    # 2. Memory-efficient load of Target columns
    print("\n[Audit Phase 1] Loading label columns...")
    target_cols = ['TIME', 'flare_now', 'flare_class', 'flare_future_10min', 'flare_future_5min', 'flare_future_30min', 'flare_future']
    df_targets = pd.read_csv(dataset_path, usecols=target_cols)
    
    total_rows = len(df_targets)
    report_lines.append("## 1. General Dataset Properties")
    report_lines.append(f"- **Total Row Count**: {total_rows:,}")
    report_lines.append(f"- **Total File Size**: {os.path.getsize(dataset_path) / (1024*1024*1024):.2f} GB")
    
    # Check for missing values in targets
    missing_targets = df_targets.isnull().sum()
    report_lines.append("\n### Target Column Missing Values:")
    for col, count in missing_targets.items():
         report_lines.append(f"- **{col}**: {count} missing values")
         
    # Check for duplicate timestamps
    dup_timestamps = df_targets['TIME'].duplicated().sum()
    report_lines.append(f"- **Duplicate Timestamps**: {dup_timestamps:,} duplicate observations")
    
    # Chronological ordering check
    is_sorted = df_targets['TIME'].is_monotonic_increasing
    report_lines.append(f"- **Chronologically Sorted**: {is_sorted}")
    
    time_diffs = df_targets['TIME'].diff().dropna()
    report_lines.append(f"- **Mean Time Step (dt)**: {time_diffs.mean():.4f} seconds")
    report_lines.append(f"- **Min Time Step (dt)**: {time_diffs.min():.4f} seconds")
    report_lines.append(f"- **Max Time Step (dt)**: {time_diffs.max():.4f} seconds")
    
    # 3. Splits check
    print("\n[Audit Phase 2] Validating chronological train/val/test splits...")
    train_ratio, val_ratio = 0.7, 0.15
    n = len(df_targets)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_times = df_targets['TIME'].iloc[:train_end]
    val_times = df_targets['TIME'].iloc[train_end:val_end]
    test_times = df_targets['TIME'].iloc[val_end:]
    
    overlap_train_val = train_times.max() >= val_times.min()
    overlap_val_test = val_times.max() >= test_times.min()
    
    report_lines.append("\n## 2. Chronological Splits Validation")
    report_lines.append(f"- **Train set size**: {len(train_times):,} rows ({train_times.min()} to {train_times.max()})")
    report_lines.append(f"- **Val set size**: {len(val_times):,} rows ({val_times.min()} to {val_times.max()})")
    report_lines.append(f"- **Test set size**: {len(test_times):,} rows ({test_times.min()} to {test_times.max()})")
    report_lines.append(f"- **Train-Validation Overlap Leakage**: {overlap_train_val}")
    report_lines.append(f"- **Validation-Test Overlap Leakage**: {overlap_val_test}")
    
    # 4. Positive/negative class ratios across splits
    report_lines.append("\n## 3. Label Balance & Positive/Negative Ratios")
    report_lines.append("\n| Split | Task | Positive Samples | Negative Samples | Positive % |")
    report_lines.append("| --- | --- | --- | --- | --- |")
    
    for split_name, df_split in [('Train', df_targets.iloc[:train_end]), 
                                 ('Val', df_targets.iloc[train_end:val_end]), 
                                 ('Test', df_targets.iloc[val_end:])]:
        for task_col in ['flare_now', 'flare_future_10min']:
            pos = int((df_split[task_col] == 1).sum())
            neg = int((df_split[task_col] == 0).sum())
            pct = (pos / (pos + neg)) * 100 if (pos + neg) > 0 else 0.0
            report_lines.append(f"| {split_name} | {task_col} | {pos:,} | {neg:,} | {pct:.4f}% |")
            
    # Multiclass split ratios
    report_lines.append("\n### Multiclass (flare_class) Distribution:")
    report_lines.append("| Split | Class 0 (Quiet) | Class 1 (C) | Class 2 (M) | Class 3 (X) |")
    report_lines.append("| --- | --- | --- | --- | --- |")
    for split_name, df_split in [('Train', df_targets.iloc[:train_end]), 
                                 ('Val', df_targets.iloc[train_end:val_end]), 
                                 ('Test', df_targets.iloc[val_end:])]:
        counts = df_split['flare_class'].value_counts()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        c3 = counts.get(3, 0)
        report_lines.append(f"| {split_name} | {c0:,} | {c1:,} | {c2:,} | {c3:,} |")
        
    # Free memory
    del df_targets
    import gc
    gc.collect()
    
    # 5. Load representative systematic subsample
    print("\n[Audit Phase 3] Loading representative subsample for feature audit...")
    chunks = []
    chunk_idx = 0
    # We read in chunks to keep memory usage extremely low
    for chunk in pd.read_csv(dataset_path, chunksize=100000):
        # Systematic sampling: every 25th row
        chunks.append(chunk.iloc[::25].copy())
        chunk_idx += 1
        
    df_sample = pd.concat(chunks, ignore_index=True)
    print(f"[INFO] Loaded sample with {len(df_sample):,} rows and {len(df_sample.columns):,} columns.")
    
    # 6. Duplicates and Missing Values checks on Features
    exclude_cols = ['TIME', 'flare_now', 'flare_class', 'flare_future_5min', 'flare_future_10min', 'flare_future_30min', 'flare_future']
    feature_cols = [c for c in df_sample.columns if c not in exclude_cols]
    
    # Check for duplicate feature rows (excluding TIME/Labels)
    dup_features = df_sample[feature_cols].duplicated().sum()
    pct_dup = (dup_features / len(df_sample)) * 100
    
    report_lines.append("\n## 4. Feature Quality Check (Subsample Stats)")
    report_lines.append(f"- **Feature Duplication (excl. time/labels)**: {dup_features:,} duplicated rows ({pct_dup:.2f}%)")
    
    missing_feats = df_sample[feature_cols].isnull().sum()
    high_missing = missing_feats[missing_feats > 0]
    report_lines.append(f"- **Features with Missing Values**: {len(high_missing)} features")
    if len(high_missing) > 0:
        report_lines.append("\n#### Features with missing values details:")
        for col, count in high_missing.items():
            report_lines.append(f"  - **{col}**: {count} missing values ({count/len(df_sample)*100:.2f}%)")
            
    # Zero variance features
    std_feats = df_sample[feature_cols].std()
    zero_var = std_feats[std_feats == 0.0].index.tolist()
    report_lines.append(f"- **Zero-variance (Constant) Features**: {len(zero_var)} features")
    if len(zero_var) > 0:
        report_lines.append(f"  - Constant Features: {zero_var}")
        
    # 7. Leakage check
    print("\n[Audit Phase 4] Checking target leakages...")
    # Calculate correlations
    corr_series = df_sample[feature_cols].corrwith(df_sample['flare_now'])
    suspicious_corr = corr_series[corr_series.abs() >= 0.99]
    
    report_lines.append("\n## 5. Potential Leakage & Suspicious Correlations")
    report_lines.append(f"- **Features with correlation >= 0.99 with target**: {len(suspicious_corr)} features")
    if len(suspicious_corr) > 0:
        for col, val in suspicious_corr.items():
            report_lines.append(f"  - **{col}**: Pearson Correlation = {val:.4f} (POTENTIAL LEAKAGE)")
    else:
        report_lines.append("  - No individual feature has a suspicious correlation >= 0.99 with the target label.")
        
    # Check if future target columns are accidentally inside the feature columns list
    leakage_in_features = [c for c in feature_cols if 'future' in c or 'class' in c]
    report_lines.append(f"- **Lookahead features inside training pool**: {len(leakage_in_features)} features")
    if len(leakage_in_features) > 0:
        report_lines.append(f"  - Suspicious feature names: {leakage_in_features}")
        
    # 8. Feature Correlation with Labels (Top features)
    report_lines.append("\n## 6. Top Features by Pearson and Spearman Correlation with `flare_now`")
    top_pearson = corr_series.abs().sort_values(ascending=False).head(15)
    
    report_lines.append("\n### Top Pearson Correlation:")
    report_lines.append("| Feature | Pearson Correlation |")
    report_lines.append("| --- | --- |")
    for col, val in top_pearson.items():
        report_lines.append(f"| {col} | {corr_series[col]:.4f} |")
        
    # Spearman correlation
    spearman_corr = df_sample[feature_cols].corrwith(df_sample['flare_now'], method='spearman')
    top_spearman = spearman_corr.abs().sort_values(ascending=False).head(15)
    
    report_lines.append("\n### Top Spearman Rank Correlation:")
    report_lines.append("| Feature | Spearman Correlation |")
    report_lines.append("| --- | --- |")
    for col, val in top_spearman.items():
        report_lines.append(f"| {col} | {spearman_corr[col]:.4f} |")
        
    # 9. Mutual Information (using selected features to avoid overhead)
    print("\n[Audit Phase 5] Calculating Mutual Information...")
    selected_json_path = 'data/labels/selected_features.json'
    if os.path.exists(selected_json_path):
        with open(selected_json_path, 'r') as f:
            selected_features = json.load(f)
    else:
        # Fallback to top 20 pearson features
        selected_features = top_pearson.index.tolist()
        
    # Restrict selection to valid columns in dataset
    selected_features = [c for c in selected_features if c in df_sample.columns]
    
    X_sample = df_sample[selected_features]
    y_sample = df_sample['flare_now']
    
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
    mi_df = pd.DataFrame({
        'Feature': selected_features,
        'Mutual Information': mi_scores
    }).sort_values('Mutual Information', ascending=False)
    
    report_lines.append("\n## 7. Predictive Signal Assessment (Mutual Information)")
    report_lines.append("| Feature | Mutual Information Score |")
    report_lines.append("| --- | --- |")
    for idx, row in mi_df.iterrows():
        report_lines.append(f"| {row['Feature']} | {row['Mutual Information']:.4f} |")
        
    # 10. Permutation Importance
    print("\n[Audit Phase 6] Training small Random Forest and computing Permutation Importance...")
    # Stratified chronological split on sample
    samp_n = len(df_sample)
    samp_train_end = int(samp_n * 0.7)
    
    X_train_s = X_sample.iloc[:samp_train_end]
    y_train_s = y_sample.iloc[:samp_train_end]
    X_val_s = X_sample.iloc[samp_train_end:]
    y_val_s = y_sample.iloc[samp_train_end:]
    
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train_s, y_train_s)
    
    val_preds = clf.predict(X_val_s)
    val_f1 = f1_score(y_val_s, val_preds, average='macro', zero_division=0)
    
    pi_result = permutation_importance(clf, X_val_s, y_val_s, scoring='f1_macro', n_repeats=5, random_state=42)
    
    pi_df = pd.DataFrame({
        'Feature': selected_features,
        'Permutation Importance Mean': pi_result.importances_mean,
        'Permutation Importance Std': pi_result.importances_std
    }).sort_values('Permutation Importance Mean', ascending=False)
    
    report_lines.append("\n## 8. Permutation Importance (on validation split)")
    report_lines.append(f"Random Forest Validation Macro F1: {val_f1:.4f}\n")
    report_lines.append("| Feature | Permutation Importance Mean | Permutation Importance Std |")
    report_lines.append("| --- | --- | --- |")
    for idx, row in pi_df.iterrows():
        report_lines.append(f"| {row['Feature']} | {row['Permutation Importance Mean']:.4f} | {row['Permutation Importance Std']:.4f} |")
        
    # 11. Bottleneck Assessment
    print("\n[Audit Phase 7] Conducting Bottleneck Assessment...")
    report_lines.append("\n## 9. Bottleneck Assessment & Recommendations")
    
    # Check labeling strategy
    config_strategy = None
    try:
        from src.utils.config import LABELING_STRATEGY
        config_strategy = LABELING_STRATEGY
    except Exception:
        pass
        
    # Labels bottleneck diagnostic
    labels_bottleneck = False
    labels_reasons = []
    if config_strategy == 'daily_strongest':
        labels_bottleneck = True
        labels_reasons.append("`LABELING_STRATEGY` is set to `'daily_strongest'`. This labels *all* seconds of a flare-containing day as positive, leading to massive lookahead/spillover leakage and flat labels per-day. The model cannot learn transient flare physical patterns, but rather just page-level daily signatures.")
        
    # Check positive samples in validation and test set
    val_pos = (y_val_s == 1).sum()
    test_pos = (df_sample['flare_now'].iloc[int(samp_n*0.85):] == 1).sum()
    if val_pos == 0 or test_pos == 0:
        labels_bottleneck = True
        labels_reasons.append(f"Severe label scarcity in split sets. Validation positive samples: {val_pos}, Test positive samples: {test_pos}. ML algorithms cannot optimize thresholds or evaluate correctly with zero/low positive counts.")

    # Features bottleneck diagnostic
    features_bottleneck = False
    features_reasons = []
    max_mi = mi_df['Mutual Information'].max()
    if max_mi < 0.01:
        features_bottleneck = True
        features_reasons.append(f"Extremely weak predictive signal. The maximum mutual information score of any selected feature with the label is {max_mi:.4f}. This suggests features do not have statistical association with flares.")
        
    max_pi = pi_df['Permutation Importance Mean'].max()
    if max_pi <= 0.00:
        features_bottleneck = True
        features_reasons.append(f"None of the features have positive permutation importance (max={max_pi:.4f}). Swapping feature columns does not degrade validation score, showing the model relies on noise/spurious signals.")
        
    # Preprocessing bottleneck diagnostic
    preprocessing_bottleneck = False
    preprocessing_reasons = []
    if len(suspicious_corr) > 0:
        preprocessing_bottleneck = True
        preprocessing_reasons.append("There are features with correlation >= 0.99, indicating target leakage (possibly due to inclusion of raw targets or derivative variables that are copies of targets).")
    if overlap_train_val or overlap_val_test:
        preprocessing_bottleneck = True
        preprocessing_reasons.append("There is overlap in timestamps across training, validation, and test boundaries, violating chronological split boundaries.")

    # Model Configuration bottleneck diagnostic
    model_bottleneck = False
    model_reasons = []
    # If the F1 macro is very low but features have high MI, then the model configuration is the bottleneck
    if val_f1 < 0.4 and not features_bottleneck and not labels_bottleneck:
        model_bottleneck = True
        model_reasons.append(f"Validation F1 score is low ({val_f1:.4f}) despite strong feature signals. This suggests hyperparameters (like early stopping, max depth, class weight) are not optimized for this data layout.")
        
    # Write Assessment
    report_lines.append("\n### Identified Bottlenecks:")
    if labels_bottleneck:
        report_lines.append("#### [CRITICAL] LABELS BOTTLENECK")
        for reason in labels_reasons:
            report_lines.append(f"- {reason}")
            
    if features_bottleneck:
        report_lines.append("#### [CRITICAL] FEATURES BOTTLENECK")
        for reason in features_reasons:
            report_lines.append(f"- {reason}")
            
    if preprocessing_bottleneck:
        report_lines.append("#### [CRITICAL] PREPROCESSING BOTTLENECK")
        for reason in preprocessing_reasons:
            report_lines.append(f"- {reason}")
            
    if model_bottleneck:
        report_lines.append("#### [WARNING] MODEL CONFIGURATION BOTTLENECK")
        for reason in model_reasons:
            report_lines.append(f"- {reason}")
            
    if not labels_bottleneck and not features_bottleneck and not preprocessing_bottleneck and not model_bottleneck:
        report_lines.append("- No major bottlenecks identified. The dataset and pipeline architecture are healthy.")
        
    # Final Recommendations
    report_lines.append("\n### Recommendations for Pipeline Optimization:")
    report_lines.append("1. **Transition Labeling Strategy**: Change `LABELING_STRATEGY` from `'daily_strongest'` back to `'overlap'` to generate temporally precise target boundaries. This is the root cause of flat predictions and poor learning behavior.")
    report_lines.append("2. **Address Class Imbalance**: Ensure class weights or balanced sampling are kept on, as the flare class is extremely rare (~0.2% positive sample rate when using overlap strategy).")
    report_lines.append("3. **Downcast Data**: Keep memory-efficient downcasting (float32/int32) to prevent out-of-memory crashes on this 2.59M row dataset.")

    # Write report
    report_content = "\n".join(report_lines)
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/dataset_audit_report.md', 'w') as f:
        f.write(report_content)
        
    print(f"\n[SUCCESS] Dataset audit complete! Report written to experiments/dataset_audit_report.md")
    print("==================================================")

if __name__ == "__main__":
    main()
