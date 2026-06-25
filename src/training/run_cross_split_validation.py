import os
import sys
import time
import numpy as np
import pandas as pd

# Set the path to the workspace root so imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import PROCESSED_DATA_DIR, EXPERIMENTS_DIR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc
)

def compute_pr_auc(y_true, y_prob):
    if y_prob is None:
        return np.nan
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return auc(recall, precision)
    except Exception:
        return np.nan

def optimize_threshold(model, X_val, y_val):
    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_score = -1.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred_val_temp = (y_prob_val >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred_val_temp, zero_division=0)
        mcc = matthews_corrcoef(y_val, y_pred_val_temp)
        bal_acc = balanced_accuracy_score(y_val, y_pred_val_temp)
        
        composite = (f1 + mcc + bal_acc) / 3.0
        if composite > best_score:
            best_score = composite
            best_thresh = thresh
    return best_thresh

def run_experiment(df, features, train_ratio, val_ratio, split_name):
    print(f"\n--- Running Split {split_name} (Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%) ---")
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Chronological partition
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train, y_train = train_df[features], train_df['flare_now'].values
    X_val, y_val = val_df[features], val_df['flare_now'].values
    X_test, y_test = test_df[features], test_df['flare_now'].values
    
    # Train Model
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    
    # Threshold Optimization on Val set
    best_thresh = optimize_threshold(model, X_val, y_val)
    
    # Test Inference and Latency
    t1 = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)
    t_inf = time.time() - t1
    inf_time_per_sample = t_inf / len(X_test)
    
    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_val = f1_score(y_test, y_pred, zero_division=0)
    macro_f1_val = f1_score(y_test, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = compute_pr_auc(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1_val,
        'Macro F1': macro_f1_val,
        'Balanced Accuracy': bal_acc,
        'MCC': mcc,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Optimal Threshold': best_thresh,
        'Training Time (s)': t_train,
        'Inference Time/Sample (μs)': inf_time_per_sample * 1e6,
        'Confusion Matrix': cm.tolist()
    }
    
    print(f"  Accuracy: {acc:.5f} | F1: {f1_val:.4f} | MCC: {mcc:.4f} | ROC-AUC: {roc_auc:.5f} | PR-AUC: {pr_auc:.5f}")
    print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    return metrics

def main():
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'dataset.csv')
    meta_path = os.path.join(EXPERIMENTS_DIR, 'experiment_053', 'metadata.json')
    
    # Load features list
    import json
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    features = meta['features']
    
    # Load and label dataset
    df = pd.read_csv(dataset_path)
    from src.labeling.labeler import label_dataset
    df = label_dataset(df)
    
    # Apply 10x downsampling to match original experiment scale
    df_sub = df.iloc[::10].copy().reset_index(drop=True)
    
    # Define three chronological splits
    splits_configs = [
        {'train': 0.70, 'val': 0.15, 'name': 'A'},
        {'train': 0.60, 'val': 0.20, 'name': 'B'},
        {'train': 0.75, 'val': 0.10, 'name': 'C'}
    ]
    
    results = {}
    for cfg in splits_configs:
        results[cfg['name']] = run_experiment(df_sub, features, cfg['train'], cfg['val'], cfg['name'])
        
    # Compile Comparison Table
    df_metrics = pd.DataFrame(results).T
    cols_to_show = ['Accuracy', 'Precision', 'Recall', 'F1', 'Macro F1', 'Balanced Accuracy', 'MCC', 'ROC-AUC', 'PR-AUC', 'Optimal Threshold', 'Training Time (s)', 'Inference Time/Sample (μs)']
    df_table = df_metrics[cols_to_show]
    
    # Compute Statistics
    stat_metrics = ['F1', 'MCC', 'ROC-AUC', 'PR-AUC']
    df_stats = df_metrics[stat_metrics].astype(float)
    
    summary_stats = pd.DataFrame({
        'Mean': df_stats.mean(),
        'Std Dev': df_stats.std(),
        'Min': df_stats.min(),
        'Max': df_stats.max()
    })
    
    # Print tables to stdout
    print("\n==================================================")
    print("CHRONOLOGICAL CROSS-SPLIT METRICS")
    print("==================================================")
    print(df_table.to_markdown())
    
    print("\n==================================================")
    print("ROBUSTNESS STATISTICS")
    print("==================================================")
    print(summary_stats.to_markdown())
    
    # Write report
    report_lines = [
        "# Chronological Cross-Split Robustness Report",
        f"Generated on: {pd.Timestamp.now().isoformat()}\n",
        "## Metrics Table",
        df_table.to_markdown(),
        "\n## Robustness Statistics",
        summary_stats.to_markdown(),
        "\n## Split Specific Details"
    ]
    
    for split_name, metr in results.items():
        cm = metr['Confusion Matrix']
        report_lines.append(f"### Split {split_name}")
        report_lines.append(f"- **Optimal Threshold**: {metr['Optimal Threshold']:.2f}")
        report_lines.append(f"- **Confusion Matrix**: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        report_lines.append(f"- **Training Time**: {metr['Training Time (s)']:.2f}s")
        report_lines.append(f"- **Inference Latency**: {metr['Inference Time/Sample (μs)']:.2f} μs/sample\n")
        
    # Final general conclusion
    mean_mcc = df_stats['MCC'].mean()
    mean_pr = df_stats['PR-AUC'].mean()
    is_stable = mean_mcc >= 0.90 and mean_pr >= 0.90
    
    conclusion = (
        "## Final Conclusion\n"
        f"Based on all three chronological partitions, the Random Forest classifier achieves a mean MCC of **{mean_mcc:.4f}** "
        f"and a mean PR-AUC of **{mean_pr:.4f}**.\n\n"
    )
    if is_stable:
        conclusion += (
            "**STATUS: STABLE**\n"
            "The model demonstrates robust temporal generalization on the available dataset. Performance metrics remain "
            "consistently high and uniform across all three independent chronological partitions, indicating that the learned "
            "signatures are stationary and generalizable across time boundaries without dependency on a single window split."
        )
    else:
        conclusion += (
            "**STATUS: VARIATION DETECTED**\n"
            "Performance varies across the splits, suggesting potential temporal non-stationarity or class distribution shifts "
            "between the training and test windows. Further investigation is required."
        )
    report_lines.append(conclusion)
    
    report_path = os.path.join(EXPERIMENTS_DIR, 'robustness_cross_split_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\n[SUCCESS] Cross-split validation report saved to {report_path}")

if __name__ == "__main__":
    main()
