import os
import joblib
import numpy as np
import pandas as pd
from src.utils.config import OUTPUTS_DIR, RAW_DATA_DIR, MODELS_DIR
from src.data.ingest import load_solexs_day, load_helios_day
from src.preprocessing.alignment import align_payloads
from src.features.engineering import add_physics_features
from src.training.train import get_feature_columns

class WeightedEnsembleClassifier:
    def __init__(self, models_dict, weights=None):
        """
        models_dict (dict): Dictionary mapping model_name -> model object.
        weights (dict): Dictionary mapping model_name -> float weight.
        """
        self.models = models_dict
        if weights is None:
            # Assign equal weights by default
            self.weights = {name: 1.0 for name in models_dict.keys()}
        else:
            self.weights = weights
            
    def predict_proba(self, df_features):
        """
        Calculates weighted soft-voting probability scores.
        """
        prob_sum = None
        weight_sum = 0.0
        
        for name, model in self.models.items():
            # Get features for this specific model
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
            elif hasattr(model, 'feature_name'):  # LightGBM
                features = model.feature_name()
            elif hasattr(model, 'feature_names_'):  # CatBoost
                features = model.feature_names_
            else:
                import json
                selected_path = os.path.join('data', 'processed', 'selected_features.json')
                if os.path.exists(selected_path):
                    with open(selected_path, 'r') as f:
                        features = json.load(f)
                else:
                    features = [c for c in df_features.columns if c not in ['TIME', 'predicted_class', 'prediction_probability']]
            
            # Align features
            features = [f for f in features if f in df_features.columns]
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(df_features[features])
                weight = self.weights.get(name, 1.0)
                weight_sum += weight
                if prob_sum is None:
                    prob_sum = probs * weight
                else:
                    prob_sum += probs * weight
            else:
                # Fallback if model doesn't have predict_proba
                preds = model.predict(df_features[features])
                classes = getattr(model, 'classes_', np.unique(preds))
                probs = np.zeros((len(df_features), len(classes)))
                for idx, c in enumerate(classes):
                    probs[preds == c, idx] = 1.0
                weight = self.weights.get(name, 1.0)
                weight_sum += weight
                if prob_sum is None:
                    prob_sum = probs * weight
                else:
                    prob_sum += probs * weight
                    
        if prob_sum is None or weight_sum == 0.0:
            raise ValueError("No models in ensemble could produce probabilities.")
            
        return prob_sum / weight_sum

    def predict(self, df_features):
        """
        Predicts classes based on weighted probabilities.
        """
        probs = self.predict_proba(df_features)
        if probs.shape[1] == 2:  # Binary classification
            # Compute weighted threshold from base models
            thresh_sum = 0.0
            weight_sum = 0.0
            for name, model in self.models.items():
                thresh = getattr(model, 'optimal_threshold_', 0.5)
                w = self.weights.get(name, 1.0)
                thresh_sum += thresh * w
                weight_sum += w
            ensemble_thresh = thresh_sum / weight_sum if weight_sum > 0 else 0.5
            print(f"[INFERENCE ENSEMBLE] Weighted threshold: {ensemble_thresh:.3f}")
            return (probs[:, 1] >= ensemble_thresh).astype(int)
        else:  # Multiclass classification
            return np.argmax(probs, axis=1)

def load_trained_model(model_path):
    """
    Loads a saved model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def predict_on_day(date_str, model_path, task='detection', detector='all'):
    """
    Runs the inference pipeline on a given day folder: Ingests, Aligns,
    Features, and Predicts. Saves the final predictions to outputs/.
    
    Parameters:
        date_str (str): The folder date (e.g. '2026-06-21').
        model_path (str): Path to the joblib model file, or 'ensemble' to use all latest models.
        task (str): Task type. Defines naming convention of predictions.
        detector (str): HEL1OS detector to load.
    """
    if model_path == 'ensemble':
        print(f"\n[INFERENCE] Running ENSEMBLE inference for date: {date_str} (Task: {task})")
    else:
        print(f"\n[INFERENCE] Running inference for date: {date_str} using {model_path}")
    
    # 1. Discover and load data
    date_folder = os.path.join(RAW_DATA_DIR, date_str)
    df_solexs = load_solexs_day(date_folder)
    df_helios = load_helios_day(date_folder, detector=detector)
    
    # 2. Align payloads
    df_aligned = align_payloads(df_solexs, df_helios)
    
    # 3. Feature engineering
    df_features = add_physics_features(df_aligned)
    
    # 5. Load model/ensemble and predict
    if model_path == 'ensemble':
        task_prefix = f"{task}_"
        models_dict = {}
        # Try loading latest validation F1 scores to use as weights if comparison reports exist
        weights = {}
        comp_report_path = os.path.join('experiments', f'model_comparison_{task}.csv')
        if os.path.exists(comp_report_path):
            try:
                comp_df = pd.read_csv(comp_report_path)
                for _, row in comp_df.iterrows():
                    m_name = str(row['Model Name']).lower()
                    f1_score = float(row['Macro F1'])
                    if not np.isnan(f1_score):
                        weights[m_name] = max(0.01, f1_score) # use F1 score as weight
                print(f"[INFERENCE] Loaded model weights from comparison report: {weights}")
            except Exception as e:
                print(f"[WARNING] Could not load weights from comparison report: {e}. Defaulting to equal weights.")
        
        for filename in os.listdir(MODELS_DIR):
            if filename.startswith(task_prefix) and filename.endswith("_latest.joblib"):
                model_name = filename[len(task_prefix):-len("_latest.joblib")]
                full_path = os.path.join(MODELS_DIR, filename)
                models_dict[model_name] = joblib.load(full_path)
                print(f"[INFERENCE] Loaded model '{model_name}' into ensemble.")
                
        if not models_dict:
            raise FileNotFoundError(f"No latest models found for task '{task}' in {MODELS_DIR}")
            
        # Filter weights dictionary to only contain loaded models
        ensemble_weights = {name: weights.get(name, 1.0) for name in models_dict.keys()}
        model = WeightedEnsembleClassifier(models_dict, weights=ensemble_weights)
    else:
        model = load_trained_model(model_path)
    
    # 4. Extract features dynamically and predict
    if isinstance(model, WeightedEnsembleClassifier):
        predictions = model.predict(df_features)
    else:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        elif hasattr(model, 'feature_name'):  # LightGBM
            features = model.feature_name()
        elif hasattr(model, 'feature_names_'):  # CatBoost
            features = model.feature_names_
        else:
            # Fallback to selected_features.json or all feature columns
            import json
            selected_path = os.path.join('data', 'processed', 'selected_features.json')
            if os.path.exists(selected_path):
                with open(selected_path, 'r') as f:
                    features = json.load(f)
            else:
                features = get_feature_columns(df_features)
                
        features = [f for f in features if f in df_features.columns]
        
        # Check if the single model has an optimal threshold attribute
        optimal_thresh = getattr(model, 'optimal_threshold_', 0.5)
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(df_features[features])
                if probs.shape[1] == 2:
                    predictions = (probs[:, 1] >= optimal_thresh).astype(int)
                    print(f"[INFERENCE] Using model-specific optimal threshold: {optimal_thresh:.3f}")
                else:
                    predictions = model.predict(df_features[features])
            except Exception:
                predictions = model.predict(df_features[features])
        else:
            predictions = model.predict(df_features[features])
    
    # Add predictions to dataframe
    df_features['predicted_class'] = predictions
    
    if hasattr(model, 'predict_proba'):
        try:
            if isinstance(model, WeightedEnsembleClassifier):
                probabilities = model.predict_proba(df_features)
            else:
                probabilities = model.predict_proba(df_features[features])
            if probabilities.shape[1] == 2:
                df_features['prediction_probability'] = probabilities[:, 1]
        except Exception as e:
            print(f"[WARNING] Probability extraction failed: {e}")
            
    # 6. Save results
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_filename = f"predictions_{task}_{date_str.replace('-', '')}.csv"
    out_path = os.path.join(OUTPUTS_DIR, out_filename)
    
    # We save a subset of columns for clean outputs
    output_cols = ['TIME', 'solexs_counts', 'helios_counts', 'hardness_ratio', 'predicted_class']
    if 'helios_czt_counts' in df_features.columns:
        output_cols.insert(3, 'helios_czt_counts')
    if 'prediction_probability' in df_features.columns:
        output_cols.append('prediction_probability')
        
    df_output = df_features[output_cols]
    df_output.to_csv(out_path, index=False)
    
    print(f"[INFERENCE SUCCESS] Saved predictions to {out_path}")
    
    # Explicit memory cleanup
    try:
        import gc
        if 'model' in locals():
            del model
        if 'models_dict' in locals():
            del models_dict
        if 'df_features' in locals():
            del df_features
        if 'df_aligned' in locals():
            del df_aligned
        if 'df_solexs' in locals():
            del df_solexs
        if 'df_helios' in locals():
            del df_helios
        gc.collect()
    except Exception:
        pass
        
    return df_output
