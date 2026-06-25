# ⚙️ Installation

Ensure you have Python 3.8+ installed. Install the package dependencies using:

```bash
pip install -r requirements.txt
```

*(If using an externally managed system-wide Python environment, append `--break-system-packages` or set up a virtual environment).*

## ⚡ Usage Examples

### Running the Interactive Web Dashboard
To start the interactive Web Dashboard for running live predictions, uploading custom FITS files, viewing interactive plots, and exploring model leaderboards, run:

```bash
streamlit run app.py
```

### Running the End-to-End Pipeline
To run the entire data preprocessing, labeling, model training, and model comparison reports on local files, run:

```bash
python3 run_pipeline.py
```

### 1. Ingesting & Aligning Data
```python
from src.data_ingestion.ingest import load_solexs_day, load_helios_day
from src.preprocessing.alignment import align_payloads

# Load SoLEXS and HEL1OS observations for a given day
df_solexs = load_solexs_day('2026-06-21')
df_helios = load_helios_day('2026-06-21', detector='cdte1')

# Align timestamps
df_aligned = align_payloads(df_solexs, df_helios)
```

### 2. Feature Engineering
```python
from src.features.engineering import add_physics_features

# Computes derivatives, rolling stats (30s, 60s, 300s), lags, and trends
df_features = add_physics_features(df_aligned)
```

### 3. Catalog-Based Labeling
```python
from src.labeling.labeler import label_dataset

# Labels nowcasting (flare_now), forecasting (flare_future), and classification (flare_class)
df_labeled = label_dataset(df_features)
```

---

---

## 📈 ML Experiments & Evaluation

The training framework automatically prevents data leakage by executing **chronological/time-based splits** (Train: 70%, Val: 15%, Test: 15%) rather than random splits, maintaining the sequence of flare evolution.

### Logged Experiment Artifacts
Each run under `experiments/experiment_XXX/` saves:
* `metadata.json`: Feature listings, row counts, and run parameters.
* `metrics.json`: Evaluated metrics on chronological test set.
* `predictions.png`: Visual overlay of ground truth and predictions.
* `feature_importance.png`: Feature weight ranking.
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
To clean the cache, rebuild the dataset from scratch, detect flares, and train the baseline classifiers (Random Forest & XGBoost), run:
```bash
# To rebuild the catalog and run training:
rm -f data/labels/goes_flares.csv data/processed/dataset*
python3 run_pipeline.py
```

### 3. How to Check the Training Results
After running `run_pipeline.py`, inspect the directories:
1. **`data/labels/goes_flares.csv`**: Check this file to verify the list of automatically detected solar flares (with start, peak, end, class, and intensity values).
2. **`experiments/`**: Open the latest directory (e.g., `experiments/experiment_008/`):
   * Inspect `metrics.json` for validation scores (accuracy, precision, recall, lead time, F1).
   * View `predictions.png` to check prediction alignments vs the actual data.
   * View `feature_importance.png` to analyze which X-ray features were most predictive.

### 4. How to Run Predictions (Inference) on a Specific Day
Use the inference script to predict on any target day folder in `data/raw`:
```python
from src.inference.predict import predict_on_day

# Load a model and predict on a specific day
predict_on_day(
    date_str="2026-06-21", 
    model_path="models/detection_random_forest_latest.joblib",
    task="detection"
)
# Output predictions are exported to outputs/predictions_detection_20260621.csv
```

---

## 🚀 Future Work & Architecture Roadmap

The project is structured to easily integrate future features without core refactoring:
1. **Multi-Payload Ensembles:** Easy extension to incorporate SUIT (UV imaging), ASPEX (solar wind particles), and PAPA payloads.
2. **Spectral Fitting:** Computing coronal temperatures and emission measures dynamically from spectra files (`.pi.gz`).
3. **External Magnetograms:** Incorporating SDO/HMI active region magnetograms as spatial features.
4. **Deep Learning Integration:** Transitioning from baseline tabular estimators to temporal architectures (LSTMs, GRUs, or Transformers) once multi-month data builds are completed.
