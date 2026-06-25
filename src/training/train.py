import os
import sys
import time
import json
import joblib
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

try:
    import shap
except ImportError:
    shap = None

# Custom modules
from src.utils.config import (
    RANDOM_SEED, EXPERIMENTS_DIR, MODELS_DIR, USE_CATBOOST, USE_OPTUNA, 
    OPTUNA_N_TRIALS, USE_FEATURE_SELECTION, CORRELATION_THRESHOLD, 
    VARIANCE_THRESHOLD, USE_CLASS_WEIGHTS, USE_BALANCED_SAMPLING
)
from src.utils.metrics import evaluate_detection, evaluate_forecasting, evaluate_classification
from src.utils.visualization import plot_predictions, plot_feature_importance, setup_plot_style
from src.features.selection import perform_feature_selection

def optimize_df_memory(df):
    """
    Downcasts float64 to float32 and int64 to int32/int16/int8 to save memory.
    """
    for col in df.columns:
        if col == 'TIME':
            continue
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype('int8')
            elif c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    return df

def get_feature_columns(df, feature_list=None):
    """
    Returns list of feature column names, excluding targets and time.
    If a predefined feature list is provided, restricts to those features.
    """
    exclude = [
        'TIME', 'flare_now', 'flare_class', 'flare_future',
        'flare_future_5min', 'flare_future_10min', 'flare_future_30min'
    ]
    all_features = [col for col in df.columns if col not in exclude]
    if feature_list is not None:
        return [col for col in all_features if col in feature_list]
    return all_features

def split_data_chronological(df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the dataset chronologically to avoid lookahead/data leakage.
    """
    df_sorted = df.sort_values('TIME').reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    return train_df, val_df, test_df

def get_next_experiment_id():
    """
    Finds the next sequential experiment directory name (e.g. experiment_001).
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(EXPERIMENTS_DIR) if d.startswith('experiment_')]
    if not existing:
        return 'experiment_001'
    
    numbers = []
    for d in existing:
        try:
            num = int(d.split('_')[1])
            numbers.append(num)
        except ValueError:
            pass
            
    next_num = max(numbers) + 1 if numbers else 1
    return f'experiment_{next_num:03d}'

def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_name, task_type, n_trials=15):
    """
    Performs optional Optuna-based hyperparameter tuning using the chronological split.
    """
    if optuna is None:
        print("[WARNING] Optuna is not installed. Skipping hyperparameter optimization and using defaults.")
        return {}
        
    print(f"[OPTUNA] Optimizing hyperparameters for {model_name} ({task_type}) over {n_trials} trials...")
    
    def objective(trial):
        params = {}
        if model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 10, 22),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            
        elif model_name == 'xgboost':
            if xgb is None:
                return 0.0
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'tree_method': 'hist',
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'eval_metric': 'logloss' if task_type == 'binary' else 'mlogloss',
                'verbosity': 0
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_name == 'lightgbm':
            if lgb is None:
                return 0.0
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbosity': -1
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_name == 'catboost':
            if cb is None:
                return 0.0
            params = {
                'iterations': trial.suggest_int('iterations', 50, 150),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'random_seed': RANDOM_SEED,
                'verbose': 0
            }
            model = cb.CatBoostClassifier(**params)
        else:
            return 0.0
            
        # Add sample weights to objective fitting if configured
        if USE_CLASS_WEIGHTS or USE_BALANCED_SAMPLING:
            sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
            
        preds = model.predict(X_val)
        
        # Metric to optimize: Macro F1 to balance multiple classes/unbalanced labels
        from sklearn.metrics import f1_score
        score = f1_score(y_val, preds, average='macro', zero_division=0)
        
        # Free memory from intermediate models during search trials
        if hasattr(model, 'booster_') and model.booster_ is not None:
            try:
                model.booster_.free_dataset()
            except Exception:
                pass
        del model
        del preds
        import gc
        gc.collect()
        
        return score
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[OPTUNA] Optimization complete. Best Macro F1 on validation set: {study.best_value:.4f}")
    return study.best_params

def train_model(X_train, y_train, X_val=None, y_val=None, model_name='random_forest', task_type='binary', params=None, use_class_weights=True, use_balanced_sampling=True):
    """
    Trains the specified machine learning model. Supports class weight adjustment.
    """
    if params is None:
        params = {}
        
    # Calculate sample weights for unbalanced datasets if configured
    sample_weight = None
    if (use_class_weights or use_balanced_sampling) and len(np.unique(y_train)) > 1:
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        
    if model_name == 'random_forest':
        clf_params = {
            'n_estimators': 100,
            'max_depth': 18,
            'min_samples_leaf': 4,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
        if use_class_weights:
            clf_params['class_weight'] = 'balanced'
        clf_params.update(params)
        model = RandomForestClassifier(**clf_params)
        model.fit(X_train, y_train)
        
    elif model_name == 'xgboost':
        if xgb is None:
            raise ImportError("XGBoost is not installed or available.")
            
        clf_params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
        if task_type == 'multiclass':
            clf_params['objective'] = 'multi:softprob'
            clf_params['eval_metric'] = 'mlogloss'
        else:
            clf_params['objective'] = 'binary:logistic'
            clf_params['eval_metric'] = 'logloss'
            if use_class_weights:
                pos_count = np.sum(y_train == 1)
                neg_count = np.sum(y_train == 0)
                if pos_count > 0:
                    clf_params['scale_pos_weight'] = neg_count / pos_count
                    
        clf_params.update(params)
        
        if X_val is not None and y_val is not None:
            model = xgb.XGBClassifier(early_stopping_rounds=15, **clf_params)
            fit_params = {}
            if task_type == 'multiclass' and sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_params)
        else:
            model = xgb.XGBClassifier(**clf_params)
            fit_params = {}
            if task_type == 'multiclass' and sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            model.fit(X_train, y_train, **fit_params)
            
    elif model_name == 'lightgbm':
        if lgb is None:
            raise ImportError("LightGBM is not installed or available.")
            
        clf_params = {
            'n_estimators': 150,
            'max_depth': 5,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_SEED,
            'verbosity': -1,
            'n_jobs': -1,
            'free_raw_data': True
        }
        if task_type == 'multiclass':
            clf_params['objective'] = 'multiclass'
        else:
            clf_params['objective'] = 'binary'
            
        if use_class_weights:
            clf_params['class_weight'] = 'balanced'
            
        clf_params.update(params)
        model = lgb.LGBMClassifier(**clf_params)
        
        if X_val is not None and y_val is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=15, verbose=False)]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
        else:
            if sample_weight is not None and not use_class_weights:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
                
        # Release internal datasets immediately after training
        if hasattr(model, 'booster_') and model.booster_ is not None:
            try:
                model.booster_.free_dataset()
            except Exception:
                pass
            
    elif model_name == 'catboost':
        if cb is None:
            raise ImportError("CatBoost is not installed or available.")
            
        clf_params = {
            'iterations': 150,
            'depth': 5,
            'learning_rate': 0.05,
            'random_seed': RANDOM_SEED,
            'verbose': 0
        }
        if task_type == 'multiclass':
            clf_params['loss_function'] = 'MultiClass'
        else:
            clf_params['loss_function'] = 'Logloss'
            
        if use_class_weights:
            clf_params['auto_class_weights'] = 'Balanced'
            
        clf_params.update(params)
        model = cb.CatBoostClassifier(**clf_params)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=15, verbose=0)
        else:
            if sample_weight is not None and not use_class_weights:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model

def run_experiment(df, task='detection', model_name='random_forest', model_params=None, use_feature_selection=None, use_optuna=None):
    """
    Runs a full experiment:
    1. Removes lookahead variables and applies chronological split.
    2. Performs optional automatic feature selection.
    3. Runs optional Optuna-based hyperparameter tuning.
    4. Trains the estimator, recording training and inference latency, and memory usage.
    5. Computes comprehensive evaluation metrics (Task A, B, or C).
    6. Generates SHAP plots, feature importance rankings, and sequential logs.
    """
    # Check if this experiment has already run and saved its latest model globally
    global_model_path = os.path.join(MODELS_DIR, f'{task}_{model_name}_latest.joblib')
    from src.utils.config import FORCE_RERUN, RETRAIN_MODELS
    
    use_checkpoint = False
    if not FORCE_RERUN and not RETRAIN_MODELS and os.path.exists(global_model_path):
        use_checkpoint = True
        
    if use_checkpoint:
        print(f"[CHECKPOINT] Reusing model checkpoint for {task}_{model_name}. Skipping training.")
        try:
            best_metrics = None
            if os.path.exists(EXPERIMENTS_DIR):
                latest_time = datetime.datetime.min
                for d in os.listdir(EXPERIMENTS_DIR):
                    meta_p = os.path.join(EXPERIMENTS_DIR, d, 'metadata.json')
                    metrics_p = os.path.join(EXPERIMENTS_DIR, d, 'metrics.json')
                    if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d)) and os.path.exists(meta_p) and os.path.exists(metrics_p):
                        with open(meta_p, 'r') as f:
                            meta = json.load(f)
                        if meta.get('task') == task and meta.get('model_name') == model_name:
                            t_str = meta.get('timestamp')
                            try:
                                t_val = datetime.datetime.fromisoformat(t_str)
                            except Exception:
                                t_val = datetime.datetime.min
                            if t_val > latest_time:
                                latest_time = t_val
                                with open(metrics_p, 'r') as f:
                                    best_metrics = json.load(f)
            if best_metrics is not None:
                return best_metrics
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint metrics: {e}")
        return {}
    else:
        print(f"[TRAINING] Retraining model {task}_{model_name} from scratch.")

    # Load defaults from config if not supplied
    if use_feature_selection is None:
        use_feature_selection = USE_FEATURE_SELECTION
    if use_optuna is None:
        use_optuna = USE_OPTUNA
        
    exp_id = get_next_experiment_id()
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n==================================================")
    print(f"Starting Experiment: {exp_id}")
    print(f"Task: {task.upper()} | Model: {model_name.upper()}")
    print(f"==================================================")
    
    # 1. Map target labels based on task
    if task == 'detection':
        target_col = 'flare_now'
        task_type = 'binary'
    elif task == 'forecast':
        target_col = 'flare_future_10min'
        task_type = 'binary'
    elif task == 'classification':
        target_col = 'flare_class'
        task_type = 'multiclass'
    else:
        raise ValueError(f"Unknown task type: {task}")
        
    # 2. Chronological Split
    train_df, val_df, test_df = split_data_chronological(df)
    
    # 3. Modular Feature Selection
    selected_features = None
    if use_feature_selection:
        # Generate selected features in processed dir
        processed_sel_dir = os.path.dirname(os.path.join(df.attrs.get('labels_dir', 'data/labels'), 'selected_features.json'))
        # Perform feature selection based on train split to avoid test leakage
        selected_features = perform_feature_selection(
            train_df, 
            target_col, 
            output_dir=processed_sel_dir,
            methods=['rf', 'xgb', 'lgb', 'correlation'],
            corr_thresh=CORRELATION_THRESHOLD,
            var_thresh=VARIANCE_THRESHOLD,
            max_features=30
        )
        # Duplicate report/JSON inside the specific experiment directory
        with open(os.path.join(exp_dir, 'selected_features.json'), 'w') as f:
            json.dump(selected_features, f, indent=4)
            
        import shutil
        src_report = os.path.join(processed_sel_dir, 'feature_selection_report.txt')
        dst_report = os.path.join(exp_dir, 'feature_selection_report.txt')
        if os.path.exists(src_report):
            shutil.copy(src_report, dst_report)
        
    features = get_feature_columns(df, selected_features)
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    
    X_val = val_df[features]
    y_val = val_df[target_col]
    
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    # 4. Optional Optuna Tuning
    best_params = {}
    if use_optuna:
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, model_name, task_type, n_trials=OPTUNA_N_TRIALS)
        if model_params is None:
            model_params = {}
        model_params.update(best_params)
        
    # 5. Model Training with Metrics & Resource Measurements
    # Memory usage before training
    ram_before = 0.0
    try:
        import psutil
        import resource
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
        
    t_start = time.time()
    model = train_model(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        model_name=model_name, 
        task_type=task_type, 
        params=model_params,
        use_class_weights=USE_CLASS_WEIGHTS,
        use_balanced_sampling=USE_BALANCED_SAMPLING
    )
    t_train = time.time() - t_start
    
    # Memory usage after training
    ram_after = 0.0
    peak_ram = 0.0
    try:
        ram_after = process.memory_info().rss / (1024 * 1024)
        peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        pass
        
    print(f"RAM before training: {ram_before:.2f} MB")
    print(f"RAM after training: {ram_after:.2f} MB")
    print(f"Peak RAM: {peak_ram:.2f} MB")
    memory_used_mb = max(0.0, ram_after - ram_before)
    
    # 6. Predict and Measure Inference Latency
    best_thresh = 0.5
    if task_type == 'binary' and hasattr(model, 'predict_proba'):
        from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
        try:
            y_prob_val = model.predict_proba(X_val)[:, 1]
            best_score = -1.0
            for thresh in np.arange(0.05, 0.95, 0.01):
                y_pred_val_temp = (y_prob_val >= thresh).astype(int)
                f1 = f1_score(y_val, y_pred_val_temp, zero_division=0)
                mcc = matthews_corrcoef(y_val, y_pred_val_temp)
                bal_acc = balanced_accuracy_score(y_val, y_pred_val_temp)
                
                # Composite score to optimize
                composite = (f1 + mcc + bal_acc) / 3.0
                if composite > best_score:
                    best_score = composite
                    best_thresh = thresh
            print(f"[THRESHOLD OPTIMIZATION] Best validation threshold: {best_thresh:.3f} (Composite Score: {best_score:.4f})")
        except Exception as e:
            print(f"[WARNING] Threshold optimization failed: {e}. Defaulting to 0.5.")
            best_thresh = 0.5
            
    # Save the optimal threshold as an attribute on the model object so it persists
    model.optimal_threshold_ = best_thresh
    
    t_inf_start = time.time()
    if task_type == 'binary' and hasattr(model, 'predict_proba'):
        try:
            y_prob_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = (y_prob_test >= best_thresh).astype(int)
        except Exception:
            y_pred_test = model.predict(X_test)
    else:
        y_pred_test = model.predict(X_test)
    t_inf = time.time() - t_inf_start
    inference_time_per_sample_sec = t_inf / len(X_test) if len(X_test) > 0 else 0.0
    
    # Extract Probabilities
    y_prob_test = None
    y_prob_all = None
    if task_type == 'binary' and hasattr(model, 'predict_proba'):
        try:
            y_prob_test = model.predict_proba(X_test)[:, 1]
            y_prob_all = model.predict_proba(df[features])[:, 1]
        except Exception:
            pass
    elif task_type == 'multiclass' and hasattr(model, 'predict_proba'):
        try:
            y_prob_test = model.predict_proba(X_test)
            y_prob_all = model.predict_proba(df[features])
        except Exception:
            pass
            
    # 7. Evaluate
    if task == 'detection':
        metrics = evaluate_detection(y_test, y_pred_test, y_prob_test)
    elif task == 'forecast':
        catalog_df = None
        for cat_name in ['goes_flares.csv', 'mock_goes_flares.csv']:
            cat_path = os.path.join(df.attrs.get('labels_dir', 'data/labels'), cat_name)
            if os.path.exists(cat_path):
                catalog_df = pd.read_csv(cat_path)
                break
        metrics = evaluate_forecasting(y_test, y_pred_test, test_df['TIME'].values, catalog_df, y_prob=y_prob_test, horizon=600)
    elif task == 'classification':
        metrics = evaluate_classification(y_test, y_pred_test, y_prob=y_prob_test)
        
    metrics['training_time_sec'] = t_train
    metrics['inference_time_per_sample_sec'] = inference_time_per_sample_sec
    metrics['memory_used_mb'] = memory_used_mb
    metrics['feature_count'] = len(features)
    metrics['optimal_threshold'] = float(best_thresh) if task_type == 'binary' else 0.5
    
    # 8. Save Model
    model_filepath = os.path.join(exp_dir, 'model.joblib')
    joblib.dump(model, model_filepath)
    
    # Global directory latest model pointer
    global_model_path = os.path.join(MODELS_DIR, f'{task}_{model_name}_latest.joblib')
    joblib.dump(model, global_model_path)
    
    # Record model file size
    model_size_bytes = os.path.getsize(model_filepath)
    metrics['model_size_bytes'] = model_size_bytes
    
    # Clean numpy variables for json logging
    serialized_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            serialized_metrics[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            serialized_metrics[k] = int(v)
        elif isinstance(v, dict):
            serialized_metrics[k] = {ik: (float(iv) if isinstance(iv, (np.float32, np.float64)) else iv) for ik, iv in v.items()}
        else:
            serialized_metrics[k] = v
            
    print(f"\nTest Set Metrics:")
    for k, v in serialized_metrics.items():
        if k != 'confusion_matrix' and k != 'per_class_precision':
            print(f"  {k}: {v}")
            
    # Save Metadata and Metrics
    metadata = {
        'experiment_id': exp_id,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'task': task,
        'model_name': model_name,
        'features': features,
        'train_rows': len(X_train),
        'val_rows': len(X_val),
        'test_rows': len(X_test),
        'optimal_threshold': float(best_thresh) if task_type == 'binary' else 0.5,
        'parameters': model_params or {}
    }
    
    with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(serialized_metrics, f, indent=4)
        
    # 9. Generate Plots & Reports
    # Prediction overlay plot
    if task_type == 'binary' and hasattr(model, 'predict_proba'):
        try:
            y_prob_all = model.predict_proba(df[features])[:, 1]
            y_pred_all = (y_prob_all >= best_thresh).astype(int)
        except Exception:
            y_pred_all = model.predict(df[features])
    else:
        y_pred_all = model.predict(df[features])
    plot_predictions(
        df,
        df[target_col].values,
        y_pred_all,
        title=f"{task.upper()} - {model_name.upper()} Model Predictions",
        save_path=os.path.join(exp_dir, 'predictions.png')
    )
    
    # Feature Importances Report (Part 5)
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = np.array(model.get_feature_importance())
        
    if importances is not None:
        # Create feature importance CSV
        feat_imp_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        feat_imp_df.to_csv(os.path.join(exp_dir, 'feature_importance.csv'), index=False)
        feat_imp_df.to_csv(os.path.join(MODELS_DIR, f'{task}_{model_name}_feature_importance.csv'), index=False)
        
        # Plot horizontal rankings
        plot_feature_importance(
            importances,
            features,
            top_n=20,
            save_path=os.path.join(exp_dir, 'feature_importance.png')
        )
        
    # SHAP Summary Beeswarm Plot (Part 5)
    if shap is not None:
        try:
            # SHAP works best on a small subsample of the test set
            shap_sub = X_test.iloc[:500] if len(X_test) > 500 else X_test
            explainer = shap.TreeExplainer(model)
            
            # For LightGBM/XGBoost/CatBoost/RandomForest, TreeExplainer is fast
            # CatBoost handles shape differently, so catch exceptions
            shap_vals = explainer.shap_values(shap_sub)
            
            plt.figure(figsize=(10, 6))
            setup_plot_style()
            
            if isinstance(shap_vals, list):
                # Multiclass or RandomForest returns a list of classes
                # Plot SHAP for the positive class (or first class if multiclass)
                shap.summary_plot(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0], shap_sub, show=False)
            else:
                shap.summary_plot(shap_vals, shap_sub, show=False)
                
            plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            shap_save_path = os.path.join(exp_dir, 'shap_summary.png')
            plt.savefig(shap_save_path, facecolor='#1e1e24')
            plt.close()
            print(f"[INFO] SHAP summary plot saved to {shap_save_path}")
        except Exception as e:
            print(f"[WARNING] SHAP beeswarm plot skipped due to TreeExplainer incompatibility: {e}")
            
    print(f"\n[SUCCESS] Experiment {exp_id} completed successfully. Artifacts saved in {exp_dir}")
    
    # Explicit memory cleanup (Requirement 1, 7, 8)
    try:
        import gc
        import matplotlib.pyplot as plt
        plt.close("all")
        
        # Free internal booster datasets if applicable
        if 'model' in locals():
            if hasattr(model, 'booster_') and model.booster_ is not None:
                try:
                    model.booster_.free_dataset()
                except Exception:
                    pass
            if hasattr(model, '_Booster') and model._Booster is not None:
                try:
                    model._Booster.free_dataset()
                except Exception:
                    pass
            if hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    if booster is not None:
                        booster.free_dataset()
                except Exception:
                    pass
            # Delete booster reference
            if hasattr(model, 'booster_'):
                del model.booster_
            if hasattr(model, '_Booster'):
                del model._Booster
            del model
            
        # Delete local variables
        if 'X_train' in locals(): del X_train
        if 'y_train' in locals(): del y_train
        if 'X_val' in locals(): del X_val
        if 'y_val' in locals(): del y_val
        if 'X_test' in locals(): del X_test
        if 'y_test' in locals(): del y_test
        if 'train_df' in locals(): del train_df
        if 'val_df' in locals(): del val_df
        if 'test_df' in locals(): del test_df
        if 'features' in locals(): del features
        if 'y_pred_test' in locals(): del y_pred_test
        if 'y_prob_test' in locals(): del y_prob_test
        if 'y_prob_all' in locals(): del y_prob_all
        if 'y_pred_all' in locals(): del y_pred_all
        if 'importances' in locals(): del importances
        if 'shap_vals' in locals(): del shap_vals
        if 'shap_sub' in locals(): del shap_sub
        if 'explainer' in locals(): del explainer
        
        gc.collect()
    except Exception as cleanup_err:
        print(f"[WARNING] Exception during local variable cleanup: {cleanup_err}")
        
    return serialized_metrics

def generate_model_comparison_report(task, experiment_ids=None, save_dir=EXPERIMENTS_DIR):
    """
    Builds a summary comparison report across completed experiments.
    Reads experiment metrics and metadata, outputs comparison tables to
    model_comparison_report.md and model_comparison_report.csv.
    """
    print(f"\n--- Generating Model Comparison Report for {task.upper()} ---")
    
    if experiment_ids is None:
        # Auto-discover all experiments in save_dir
        experiment_ids = []
        if os.path.exists(save_dir):
            for d in os.listdir(save_dir):
                if d.startswith('experiment_') and os.path.isdir(os.path.join(save_dir, d)):
                    experiment_ids.append(d)
        experiment_ids = sorted(experiment_ids)
        
    records = []
    for exp_id in experiment_ids:
        meta_path = os.path.join(save_dir, exp_id, 'metadata.json')
        metrics_path = os.path.join(save_dir, exp_id, 'metrics.json')
        
        if os.path.exists(meta_path) and os.path.exists(metrics_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            # Filter by task type
            if meta.get('task') == task:
                records.append({
                    'Experiment ID': exp_id,
                    'Model Name': meta.get('model_name').upper(),
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'Precision': metrics.get('precision', np.nan),
                    'Recall': metrics.get('recall', np.nan),
                    'Macro F1': metrics.get('macro_f1', np.nan),
                    'Balanced Accuracy': metrics.get('balanced_accuracy', np.nan),
                    'ROC-AUC': metrics.get('roc_auc', metrics.get('roc_auc_ovr', np.nan)),
                    'Training Time (s)': metrics.get('training_time_sec', 0.0),
                    'Inference Time/Sample (s)': metrics.get('inference_time_per_sample_sec', 0.0),
                    'Model Size (KB)': metrics.get('model_size_bytes', 0) / 1024.0,
                    'Memory Usage (MB)': metrics.get('memory_used_mb', 0.0),
                    'Feature Count': metrics.get('feature_count', 0),
                    'Best Hyperparameters': json.dumps(meta.get('parameters', {}))
                })
                
    if not records:
        print("[WARNING] No matching experiments found to generate a comparison report.")
        return None
        
    df_comp = pd.DataFrame(records)
    
    # Save as CSV
    csv_path = os.path.join(save_dir, f'model_comparison_{task}.csv')
    df_comp.to_csv(csv_path, index=False)
    print(f"[SUCCESS] CSV comparison saved to: {csv_path}")
    
    # Save as Markdown
    md_lines = []
    md_lines.append(f"# Model Comparison Report for task: **{task.upper()}**")
    md_lines.append(f"Generated at: {datetime.datetime.utcnow().isoformat()} UTC\n")
    md_lines.append(df_comp.to_markdown(index=False))
    
    md_path = os.path.join(save_dir, f'model_comparison_{task}.md')
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"[SUCCESS] Markdown comparison saved to: {md_path}")
    
    return df_comp

def generate_feature_importance_comparison_report(task, save_dir=EXPERIMENTS_DIR):
    """
    Finds the latest experiment for each model type under the given task,
    loads their feature importance CSVs, aggregates them, saves a ranking comparison
    to feature_importance_comparison.csv, and plots a grouped horizontal bar chart 
    for the Top-20 features to feature_importance_comparison.png.
    """
    print(f"\n--- Generating Feature Importance Comparison for {task.upper()} ---")
    
    # Locate all experiments
    if not os.path.exists(save_dir):
        print(f"[WARNING] Save directory {save_dir} does not exist. Cannot generate report.")
        return None
        
    model_types = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
    latest_exps = {m: None for m in model_types}
    latest_times = {m: datetime.datetime.min for m in model_types}
    
    # Parse directories
    for d in os.listdir(save_dir):
        meta_path = os.path.join(save_dir, d, 'metadata.json')
        imp_path = os.path.join(save_dir, d, 'feature_importance.csv')
        if os.path.isdir(os.path.join(save_dir, d)) and os.path.exists(meta_path) and os.path.exists(imp_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('task') == task:
                m_name = meta.get('model_name')
                t_str = meta.get('timestamp')
                try:
                    t_val = datetime.datetime.fromisoformat(t_str)
                except Exception:
                    t_val = datetime.datetime.min
                if m_name in latest_exps:
                    if t_val > latest_times[m_name]:
                        latest_times[m_name] = t_val
                        latest_exps[m_name] = os.path.join(save_dir, d)
                        
    # Load and merge CSVs
    dfs = []
    for m, path in latest_exps.items():
        if path is not None:
            imp_file = os.path.join(path, 'feature_importance.csv')
            df_imp = pd.read_csv(imp_file)
            df_imp = df_imp.rename(columns={'Importance': f'{m.upper()}_importance'})
            dfs.append(df_imp)
            
    if not dfs:
        print("[WARNING] No feature importance files found for comparison.")
        return None
        
    # Merge all
    df_merged = dfs[0]
    for df_next in dfs[1:]:
        df_merged = pd.merge(df_merged, df_next, on='Feature', how='outer')
        
    df_merged = df_merged.fillna(0.0)
    
    # Calculate Average Importance across models
    imp_cols = [c for c in df_merged.columns if c != 'Feature']
    df_merged['Average_importance'] = df_merged[imp_cols].mean(axis=1)
    df_merged = df_merged.sort_values('Average_importance', ascending=False).reset_index(drop=True)
    
    # Save comparison CSV
    csv_path = os.path.join(save_dir, f'feature_importance_comparison_{task}.csv')
    df_merged.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Feature importance comparison saved to: {csv_path}")
    
    # Plot comparison for Top-20 features
    try:
        top_n = min(20, len(df_merged))
        df_top = df_merged.head(top_n)
        
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features_reversed = df_top['Feature'].values[::-1]
        y_pos = np.arange(len(features_reversed))
        height = 0.2
        
        for idx, col in enumerate(imp_cols):
            importances_reversed = df_top[col].values[::-1]
            ax.barh(y_pos + (idx - len(imp_cols)/2.0)*height + height/2.0, importances_reversed, height, label=col.replace('_importance', '').upper())
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features_reversed, fontsize=9)
        ax.set_xlabel("Normalized Feature Importance", fontsize=12)
        ax.set_title(f"Top-{top_n} Feature Importance Comparison ({task.upper()})", fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', framealpha=0.3)
        ax.grid(True, axis='x', ls="-", alpha=0.3)
        plt.tight_layout()
        
        png_path = os.path.join(save_dir, f'feature_importance_comparison_{task}.png')
        plt.savefig(png_path, facecolor='#1e1e24')
        plt.close()
        print(f"[SUCCESS] Feature importance comparison plot saved to: {png_path}")
    except Exception as e:
        print(f"[WARNING] Failed to plot feature importance comparison: {e}")
        
    return df_merged

