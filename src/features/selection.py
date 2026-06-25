import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

# Import classifiers conditionally
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

def perform_feature_selection(df, target_col, output_dir, methods=None, corr_thresh=0.95, var_thresh=1e-4, max_features=30, subsample_size=5000, random_seed=42):
    """
    Performs modular feature selection on a training DataFrame:
    1. Removes duplicate features.
    2. Removes constant and near-constant features (variance < var_thresh).
    3. Removes highly correlated features (> corr_thresh), keeping the one with higher correlation with target.
    4. Computes rank importances using selected ML models, Permutation Importance, and Mutual Information.
    5. Saves selected features to selected_features.json.
    6. Generates a feature selection report.
    """
    from src.utils.config import FORCE_RERUN
    selected_json_path = os.path.join(output_dir, 'selected_features.json')
    if not FORCE_RERUN and os.path.exists(selected_json_path):
        print(f"[CHECKPOINT] Reusing feature selection checkpoint at {selected_json_path}. Skipping selection.")
        with open(selected_json_path, 'r') as f:
            return json.load(f)

    print("\n--- Starting Feature Selection ---")
    os.makedirs(output_dir, exist_ok=True)
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("FEATURE SELECTION REPORT")
    report_lines.append(f"Timestamp: {pd.Timestamp.now()}")
    report_lines.append("==================================================")
    
    # 0. Separate features and target
    exclude = [
        'TIME', 'flare_now', 'flare_class', 'flare_future',
        'flare_future_5min', 'flare_future_10min', 'flare_future_30min'
    ]
    all_features = [c for c in df.columns if c not in exclude]
    
    X = df[all_features].copy()
    y = df[target_col].copy()
    
    initial_feature_count = X.shape[1]
    report_lines.append(f"Initial feature count: {initial_feature_count}")
    
    # 1. Remove duplicate columns
    # Find columns with duplicate values
    non_dup_cols = []
    dup_cols = []
    seen_hashes = {}
    for col in X.columns:
        # Hash column values for fast duplicate identification
        col_values = X[col].values
        col_hash = hash(col_values.tobytes())
        if col_hash in seen_hashes:
            dup_cols.append(col)
        else:
            seen_hashes[col_hash] = col
            non_dup_cols.append(col)
            
    X = X[non_dup_cols]
    report_lines.append(f"Dropped duplicate features ({len(dup_cols)}): {dup_cols}")
    
    # 2. Remove constant and near-constant features (variance threshold)
    variances = X.var()
    low_var_cols = variances[variances < var_thresh].index.tolist()
    X = X.drop(columns=low_var_cols)
    report_lines.append(f"Dropped constant/near-constant features ({len(low_var_cols)}): {low_var_cols}")
    
    # 3. Remove highly correlated features (> corr_thresh)
    # Keeping the one with higher absolute correlation to target
    corr_matrix = X.corr().abs()
    
    # Pre-calculate target correlations
    target_corrs = X.apply(lambda col: np.abs(np.corrcoef(col.values, y.values)[0, 1]) if len(np.unique(col)) > 1 else 0.0)
    target_corrs = target_corrs.fillna(0.0)
    
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    corr_pairs_dropped = []
    
    for col in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[col] > corr_thresh].tolist()
        for ref_feat in correlated_features:
            if ref_feat not in to_drop and col not in to_drop:
                # Compare their correlation with target
                if target_corrs[ref_feat] >= target_corrs[col]:
                    to_drop.add(col)
                    corr_pairs_dropped.append(f"{col} dropped (correlated with {ref_feat}, correlation={upper_tri.loc[ref_feat, col]:.3f})")
                else:
                    to_drop.add(ref_feat)
                    corr_pairs_dropped.append(f"{ref_feat} dropped (correlated with {col}, correlation={upper_tri.loc[ref_feat, col]:.3f})")
                    
    X = X.drop(columns=list(to_drop))
    report_lines.append(f"Dropped highly correlated features (> {corr_thresh}) ({len(to_drop)}):")
    for pair in corr_pairs_dropped:
        report_lines.append(f"  - {pair}")
        
    remaining_features = X.columns.tolist()
    report_lines.append(f"Features remaining after filtering: {len(remaining_features)}")
    
    if not remaining_features:
        raise ValueError("No features remaining after pre-filtering. Try lowering thresholds.")
        
    # 4. Rank remaining features
    # If methods is not specified, run all available
    if methods is None:
        methods = ['rf', 'xgb', 'lgb', 'catboost', 'permutation', 'mutual_info']
        
    # Subsample for expensive calculations
    rng = np.random.default_rng(random_seed)
    if X.shape[0] > subsample_size:
        indices = rng.choice(X.shape[0], size=subsample_size, replace=False)
        X_sub = X.iloc[indices]
        y_sub = y.iloc[indices]
    else:
        X_sub = X
        y_sub = y

    importance_scores = {col: 0.0 for col in remaining_features}
    methods_run = 0
    
    # A. Random Forest Importance
    if 'rf' in methods:
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_seed, n_jobs=-1)
            rf.fit(X_sub, y_sub)
            rf_importances = rf.feature_importances_
            # Normalize and add
            rf_importances = rf_importances / (np.sum(rf_importances) + 1e-9)
            for i, col in enumerate(remaining_features):
                importance_scores[col] += rf_importances[i]
            methods_run += 1
            report_lines.append("  - Successfully included Random Forest feature importance.")
        except Exception as e:
            report_lines.append(f"  - Failed Random Forest importance: {e}")
            
    # B. XGBoost Importance
    if 'xgb' in methods and xgb is not None:
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=random_seed, n_jobs=-1, verbosity=0)
            xgb_model.fit(X_sub, y_sub)
            xgb_importances = xgb_model.feature_importances_
            xgb_importances = xgb_importances / (np.sum(xgb_importances) + 1e-9)
            for i, col in enumerate(remaining_features):
                importance_scores[col] += xgb_importances[i]
            methods_run += 1
            report_lines.append("  - Successfully included XGBoost feature importance.")
        except Exception as e:
            report_lines.append(f"  - Failed XGBoost importance: {e}")
            
    # C. LightGBM Importance
    if 'lgb' in methods and lgb is not None:
        try:
            lgb_model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=random_seed, n_jobs=-1, verbosity=-1)
            lgb_model.fit(X_sub, y_sub)
            lgb_importances = lgb_model.feature_importances_
            lgb_importances = lgb_importances / (np.sum(lgb_importances) + 1e-9)
            for i, col in enumerate(remaining_features):
                importance_scores[col] += lgb_importances[i]
            methods_run += 1
            report_lines.append("  - Successfully included LightGBM feature importance.")
        except Exception as e:
            report_lines.append(f"  - Failed LightGBM importance: {e}")
            
    # D. CatBoost Importance
    if 'catboost' in methods and cb is not None:
        try:
            cb_model = cb.CatBoostClassifier(iterations=50, depth=5, random_seed=random_seed, verbose=0)
            cb_model.fit(X_sub, y_sub)
            cb_importances = np.array(cb_model.get_feature_importance())
            cb_importances = cb_importances / (np.sum(cb_importances) + 1e-9)
            for i, col in enumerate(remaining_features):
                importance_scores[col] += cb_importances[i]
            methods_run += 1
            report_lines.append("  - Successfully included CatBoost feature importance.")
        except Exception as e:
            report_lines.append(f"  - Failed CatBoost importance: {e}")
            
    # E. Permutation Importance
    if 'permutation' in methods:
        try:
            # Fit a quick Random Forest model to compute permutation importance
            pi_model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=random_seed, n_jobs=-1)
            pi_model.fit(X_sub, y_sub)
            res = permutation_importance(pi_model, X_sub, y_sub, n_repeats=3, random_state=random_seed, n_jobs=-1)
            p_importances = np.maximum(0, res.importances_mean) # replace negatives with 0
            p_importances = p_importances / (np.sum(p_importances) + 1e-9)
            for i, col in enumerate(remaining_features):
                importance_scores[col] += p_importances[i]
            methods_run += 1
            report_lines.append("  - Successfully included Permutation importance.")
        except Exception as e:
            report_lines.append(f"  - Failed Permutation importance: {e}")
            
    # F. Mutual Information Importance
    if 'mutual_info' in methods:
        try:
            mi = mutual_info_classif(X_sub, y_sub, random_state=random_seed)
            mi = mi / (np.sum(mi) + 1e-9)
            for i, col in enumerate(remaining_features):
                importance_scores[col] += mi[i]
            methods_run += 1
            report_lines.append("  - Successfully included Mutual Information importance.")
        except Exception as e:
            report_lines.append(f"  - Failed Mutual Information: {e}")

    # Compute final ranking based on average of method importances
    if methods_run > 0:
        for col in remaining_features:
            importance_scores[col] /= methods_run
            
    sorted_features = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 5. Select Top Features
    selected_features = [col for col, score in sorted_features[:max_features]]
    
    report_lines.append("\n==================================================")
    report_lines.append(f"TOP {max_features} SELECTED FEATURES (Ranked):")
    report_lines.append("==================================================")
    for rank, (col, score) in enumerate(sorted_features[:max_features], 1):
        report_lines.append(f"{rank:02d}. {col:<35} | Average Normalized Score: {score:.5f}")
        
    report_lines.append("\n==================================================")
    report_lines.append("REJECTED FEATURES (Lowest Importance):")
    report_lines.append("==================================================")
    for rank, (col, score) in enumerate(sorted_features[max_features:], max_features + 1):
        report_lines.append(f"{rank:02d}. {col:<35} | Average Normalized Score: {score:.5f}")

    # Write selected features to JSON
    selected_json_path = os.path.join(output_dir, 'selected_features.json')
    with open(selected_json_path, 'w') as f:
        json.dump(selected_features, f, indent=4)
    print(f"[SUCCESS] Selected features list saved to: {selected_json_path}")
    
    # Save Report
    report_path = os.path.join(output_dir, 'feature_selection_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"[SUCCESS] Feature selection report saved to: {report_path}")
    
    return selected_features
