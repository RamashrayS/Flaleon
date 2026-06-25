# ⚙️ Installation

Ensure you have Python 3.8+ installed. Install the package dependencies using:

```bash
pip install -r requirements.txt
```

*(If using an externally managed system-wide Python environment, append `--break-system-packages` or set up a virtual environment).*

## ⚡ Usage Examples

### Running the End-to-End Pipeline
To run the entire data preprocessing, labeling, model training, and model comparison reports on local files, run:

```bash
python3 run_pipeline.py
```

### Programmatic API Execution

#### 1. Ingesting & Aligning Data
```python
from src.data.ingest import load_solexs_day, load_helios_day
from src.preprocessing.alignment import align_payloads

# Load SoLEXS and HEL1OS observations for a given day
df_solexs = load_solexs_day('2026-06-21')

# detector='all' loads CDTe and CZT averages, or specify 'cdte1'/'cdte2'/'czt1'/'czt2'
df_helios = load_helios_day('2026-06-21', detector='all')

# Align timestamps
df_aligned = align_payloads(df_solexs, df_helios)
```

#### 2. Feature Engineering
```python
from src.features.engineering import add_physics_features

# Computes derivatives, rolling stats (30s, 60s, 300s), lags, and trends
df_features = add_physics_features(df_aligned)
```

#### 3. Catalog-Based Labeling
```python
from src.labeling.labeler import label_dataset

# Labels nowcasting (flare_now), forecasting (flare_future), and classification (flare_class)
df_labeled = label_dataset(df_features)
```

---

## 📈 ML Experiments & Evaluation

The training framework automatically prevents data leakage by executing **chronological/time-based splits** (Train: 70%, Val: 15%, Test: 15%) rather than random splits, maintaining the sequence of flare evolution.

### Logged Experiment Artifacts
Each run under `experiments/experiment_XXX/` saves a comprehensive suite of diagnostic and scientific reports:
* `metadata.json`: Feature listings, row counts, and run parameters.
* `metrics.json`: Evaluated metrics on chronological test set (including Accuracy, F1, Macro F1, MCC, ROC-AUC, PR-AUC).
* `predictions.png`: Visual timeline overlay of ground truth and predictions.
* `feature_importance.png` / `feature_importance.csv`: Horizontal feature importance rankings.
* `shap_summary.png`: SHAP beeswarm plot representing feature impacts on model predictions.
* `confusion_matrix_test.png` / `calibration_curve_test.png`: Classification threshold performance curves.
* `roc_curve_test.png` / `precision_recall_curve_test.png` / `probability_distribution_test.png`: Diagnostic curves.
* `selected_features.json` / `feature_selection_report.txt`: Outputs of modular feature selection.
* `final_scientific_validation_report.md` / `scientific_validation_report.json`: Space weather validation audit.
* `model.joblib`: Serializable trained estimator.

---

## 📖 Operational Guide: Data Expansion & Model Execution

### 1. How to Add More Data for Training
The pipeline is designed to dynamically discover and ingest any new observation days without code modifications. To add a new day:
1. Create a new directory in `data/raw/` named in `YYYY-MM-DD` format (e.g. `data/raw/2026-07-12`).
2. Inside that directory, place:
   * **SoLEXS Data:** A folder containing the SoLEXS files (the folder name can start with `AL1_SLX` or `SOLEXS`, containing the lightcurve `.lc.gz` file).
   * **HEL1OS Data:** Two folders representing the CDTe lightcurve segments (their folder names should start with `HLS_` or `HEL1OS`, containing the `lightcurve_cdte1.fits` file).

*The pipeline will automatically sort dates chronologically, reconstruct the event catalog `goes_flares.csv`, merge the timeseries data, and retrain the models with the new additions.*

### 2. How to Execute the Pipeline
To run the execution pipeline, you can run `run_pipeline.py` with several CLI arguments:
* **Run with defaults (rebuilds dataset, skips retraining if models are cached):**
  ```bash
  python3 run_pipeline.py
  ```
* **Retrain models from scratch (ignores checkpoints):**
  ```bash
  python3 run_pipeline.py --retrain-models
  ```
* **Resume execution using existing checkpoints:**
  ```bash
  python3 run_pipeline.py --resume
  ```
* **Fast verification (trains only Random Forest on a 10% representative downsample):**
  ```bash
  python3 run_pipeline.py --fast-verify
  ```
* **Train specific models only:**
  ```bash
  python3 run_pipeline.py --models random_forest,lightgbm
  ```

### 3. How to Check the Training Results
After running `run_pipeline.py`, inspect the directories:
1. **`data/labels/goes_flares.csv`**: Check this file to verify the list of automatically detected solar flares (with start, peak, end, class, and intensity values).
2. **`experiments/`**: Open the latest directory (e.g., `experiments/experiment_053/`):
   * Inspect `metrics.json` for validation scores (accuracy, precision, recall, lead time, F1).
   * View `predictions.png` to check prediction alignments vs the actual data.
   * View `feature_importance.png` to analyze which X-ray features were most predictive.
   * Read `final_scientific_validation_report.md` for space weather operational validation metrics.

### 4. How to Run Predictions (Inference) on a Specific Day
Use the inference script to predict on any target day folder in `data/raw`:

* **Single Model Inference:**
  ```python
  from src.inference.predict import predict_on_day

  # Load a specific model and predict on a specific day
  predict_on_day(
      date_str="2026-06-21", 
      model_path="models/detection_random_forest_latest.joblib",
      task="detection"
  )
  # Output predictions are exported to outputs/predictions_detection_20260621.csv
  ```

* **Weighted Soft-Voting Ensemble Inference:**
  Pass `'ensemble'` as the model path to load all latest models for the task and weigh them by their macro-F1 test scores:
  ```python
  from src.inference.predict import predict_on_day

  predict_on_day(
      date_str="2026-06-21", 
      model_path="ensemble",
      task="detection"
  )
  # Output predictions are exported to outputs/predictions_detection_20260621.csv
  ```

---

