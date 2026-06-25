import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef, 
    cohen_kappa_score, brier_score_loss, precision_recall_curve, auc, 
    average_precision_score
)

def compute_pr_auc(y_true, y_prob):
    """
    Computes Precision-Recall Area Under the Curve (PR-AUC).
    """
    if y_prob is None:
        return np.nan
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return auc(recall, precision)
    except Exception:
        return np.nan

def evaluate_detection(y_true, y_pred, y_prob=None):
    """
    Computes comprehensive classification metrics for flare detection (Task A).
    """
    # Existing metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        
    # Training improvements metrics (Part 4)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Brier Score & PR-AUC
    if y_prob is not None:
        try:
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            metrics['pr_auc'] = compute_pr_auc(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics['brier_score'] = np.nan
            metrics['pr_auc'] = np.nan
            metrics['average_precision'] = np.nan
    else:
        metrics['brier_score'] = np.nan
        metrics['pr_auc'] = np.nan
        metrics['average_precision'] = np.nan
        
    # Per-class Recall and Precision
    p_classes = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_classes = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for idx, (p, r) in enumerate(zip(p_classes, r_classes)):
        metrics[f'precision_class_{idx}'] = p
        metrics[f'recall_class_{idx}'] = r
        
    return metrics

def evaluate_forecasting(y_true, y_pred, times, catalog_df, y_prob=None, horizon=600):
    """
    Computes metrics for flare forecasting (Task B), including Precision, Recall,
    F1 Score, and Lead Time for True Positive warnings.
    """
    # Standard metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Compute Lead Time for True Positives
    lead_times = []
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    
    if len(tp_indices) > 0 and catalog_df is not None and not catalog_df.empty:
        flare_starts = catalog_df['start_time'].values
        
        for idx in tp_indices:
            t = times[idx]
            future_flares = flare_starts[(flare_starts > t) & (flare_starts <= t + horizon + 5)] # 5s buffer
            if len(future_flares) > 0:
                lead_time = future_flares.min() - t
                lead_times.append(lead_time)
                
    avg_lead_time = np.mean(lead_times) if lead_times else 0.0
    median_lead_time = np.median(lead_times) if lead_times else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_lead_time_sec': avg_lead_time,
        'median_lead_time_sec': median_lead_time,
        'num_warnings': int(y_pred.sum()),
        'num_true_positives': len(tp_indices)
    }
    
    # Additional improvements metrics (Part 4)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            metrics['pr_auc'] = compute_pr_auc(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = np.nan
            metrics['brier_score'] = np.nan
            metrics['pr_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        metrics['brier_score'] = np.nan
        metrics['pr_auc'] = np.nan
        
    # Per-class Recall and Precision
    p_classes = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_classes = recall_score(y_true, y_pred, average=None, zero_division=0)
    for idx, (p, r) in enumerate(zip(p_classes, r_classes)):
        metrics[f'precision_class_{idx}'] = p
        metrics[f'recall_class_{idx}'] = r
        
    return metrics

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Computes multiclass evaluation metrics for flare magnitude prediction (Task C).
    """
    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'confusion_matrix': cm.tolist(),
        'macro_f1': macro_f1
    }
    
    # Additional improvements metrics (Part 4)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Multiclass ROC-AUC (ovr / ovo)
    if y_prob is not None:
        try:
            # Handle unique classes check
            classes = np.unique(y_true)
            if len(classes) > 1:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            else:
                metrics['roc_auc_ovr'] = np.nan
        except Exception:
            metrics['roc_auc_ovr'] = np.nan
    else:
        metrics['roc_auc_ovr'] = np.nan
        
    # Per-class Recall and Precision
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    p_classes = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_classes = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_precision = {}
    for idx, c in enumerate(unique_classes):
        per_class_precision[f'precision_class_{c}'] = p_classes[idx]
        metrics[f'recall_class_{c}'] = r_classes[idx]
        
    metrics['per_class_precision'] = per_class_precision
    return metrics
