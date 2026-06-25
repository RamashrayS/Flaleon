import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

def main():
    print("==================================================")
    print("GENERATING SCIENTIFIC VALIDATION PLOTS (MATPLOTLIB)")
    print("==================================================")
    
    exp_dir = 'experiments/experiment_053'
    model_path = os.path.join(exp_dir, 'model.joblib')
    dataset_path = 'data/processed/dataset.csv'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Processed dataset not found at {dataset_path}")
        return
        
    # Load model and metadata
    model = joblib.load(model_path)
    with open(os.path.join(exp_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
        
    features = meta['features']
    optimal_threshold = meta.get('optimal_threshold', 0.8)
    print(f"[INFO] Loaded model with {len(features)} features. Threshold: {optimal_threshold:.4f}")
    
    # Load dataset & apply overlap labeling
    df = pd.read_csv(dataset_path)
    from src.labeling.labeler import label_dataset
    df = label_dataset(df)
    
    # Slice the dataset matching the 10x downsampling used in experiment 053
    df_sub = df.iloc[::10].copy().reset_index(drop=True)
    n_sub = len(df_sub)
    
    train_ratio, val_ratio = 0.7, 0.15
    val_end = int(n_sub * (train_ratio + val_ratio))
    
    # Extract Test Split
    test_df = df_sub.iloc[val_end:]
    X_test = test_df[features]
    y_test = test_df['flare_now'].values
    print(f"[INFO] Extracted Test split with {len(y_test)} rows and {int((y_test==1).sum())} positive samples.")
    
    # Run Inference
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Set plot aesthetics using standard styles
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'text.color': '#212529',
        'axes.labelcolor': '#212529',
        'xtick.color': '#212529',
        'ytick.color': '#212529',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11
    })
    
    # 1. Confusion Matrix
    print("[1/7] Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Quiet', 'Flare'], yticklabels=['Quiet', 'Flare'],
           title='Confusion Matrix on Test Set',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'confusion_matrix_test.png'), dpi=300)
    plt.close()
    
    # 2. ROC Curve
    print("[2/7] Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#0d6efd', lw=2, label=f'ROC Curve (AUC = {roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], color='#6c757d', linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title('Receiver Operating Characteristic (ROC) Curve', pad=15)
    plt.xlabel('False Positive Rate', labelpad=10)
    plt.ylabel('True Positive Rate', labelpad=10)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'roc_curve_test.png'), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    print("[3/7] Plotting PR Curve...")
    prec, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, prec)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, prec, color='#dc3545', lw=2, label=f'PR Curve (AUC = {pr_auc:.5f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title('Precision-Recall Curve', pad=15)
    plt.xlabel('Recall', labelpad=10)
    plt.ylabel('Precision', labelpad=10)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'precision_recall_curve_test.png'), dpi=300)
    plt.close()
    
    # 4. Probability Distribution Histogram
    print("[4/7] Plotting Probability Distribution...")
    plt.figure(figsize=(7, 5))
    plt.hist(y_prob[y_test == 0], bins=40, color='#6c757d', label='Quiet (Negative Class)', alpha=0.5, density=True)
    plt.hist(y_prob[y_test == 1], bins=40, color='#ffc107', label='Flare (Positive Class)', alpha=0.6, density=True)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Decision Threshold ({optimal_threshold:.2f})')
    plt.title('Probability Distribution of Predictions', pad=15)
    plt.xlabel('Predicted Flare Probability', labelpad=10)
    plt.ylabel('Density', labelpad=10)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'probability_distribution_test.png'), dpi=300)
    plt.close()
    
    # 5. Calibration Curve
    print("[5/7] Plotting Calibration Curve...")
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#198754', label='Random Forest')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#6c757d', label='Perfect Calibration')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Probability Calibration Curve', pad=15)
    plt.xlabel('Mean Predicted Probability', labelpad=10)
    plt.ylabel('Fraction of Positives', labelpad=10)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'calibration_curve_test.png'), dpi=300)
    plt.close()
    
    # 6. Feature Importance Ranking
    print("[6/7] Exporting Feature Importance Plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20
    top_indices = indices[:20]
    top_importances = importances[top_indices]
    top_features = [features[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(top_features)), top_importances[::-1], align='center', color=plt.cm.viridis(np.linspace(0.8, 0.2, len(top_features))))
    plt.yticks(np.arange(len(top_features)), top_features[::-1])
    plt.title('Top 20 Feature Importances (Random Forest)', pad=15)
    plt.xlabel('Mean Decrease in Impurity', labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'feature_importance_test.png'), dpi=300)
    plt.close()
    
    # 7. SHAP summary plot (TreeSHAP)
    print("[7/7] Generating SHAP Summary...")
    try:
        import shap
        # Limit to 300 test samples to keep computation extremely fast (<5 seconds)
        shap_sub = X_test.iloc[:300]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(shap_sub)
        
        plt.figure(figsize=(10, 6))
        # Plot SHAP for Class 1 (positive flare class)
        shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals, shap_sub, show=False)
        plt.title('SHAP Feature Importance Summary (Test Set)', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'shap_summary_test.png'), dpi=300)
        plt.close()
        print("  SHAP summary generated successfully!")
    except Exception as e:
        print(f"  [WARNING] SHAP generation skipped: {e}")
        
    # Leakage check: verify no single feature dominates
    max_imp = np.max(importances)
    dominant_feature = features[np.argmax(importances)]
    leakage_detected = "FAIL" if max_imp > 0.40 else "PASS"
    print(f"\nLeakage Check: Dominant Feature: {dominant_feature} with {max_imp*100:.2f}% importance. Status: {leakage_detected}")
    
    # Save text report of metrics
    report = {
        'confusion_matrix': cm.tolist(),
        'top_20_features': [{'feature': f, 'importance': float(imp)} for f, imp in zip(top_features, top_importances)],
        'leakage_status': leakage_detected,
        'dominant_feature': dominant_feature,
        'dominant_importance': float(max_imp)
    }
    with open(os.path.join(exp_dir, 'scientific_validation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[SUCCESS] Scientific verification plots and report exported to {exp_dir}!")
    print("==================================================")

if __name__ == "__main__":
    main()
