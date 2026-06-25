<p align="center">
<img width="3780" height="1890" alt="banner_flaleon" src="https://github.com/user-attachments/assets/4f7a9196-b45c-4d79-ad74-ceb16b169a3d" />
Flaleon is a ML framework to Forecast, Nowcast and Classify solar flares into X,M or C classes

## Data used

The models are trained on:

- SoLEXS (Solar Low Energy X-ray Spectrometer) &
- HEL1OS (High Energy L1 Orbiting X-ray Spectrometer)

sent by [AdityaL1](https://www.isro.gov.in/Aditya_L1.html) (available for download from the [ISSDC](https://pradan1.issdc.gov.in/al1/) portal)

The training set was of 30 days (not continuous, the dates were particularly picked to make sure the models are fed healthy amounts of X,M,C class flares and quiet days)

Out of 30 days:
- X-class days: **6**
- M-class days: **5**
- C-class days: **7**
- Quiet days: **12**

*The dates were picked from 2024, 2025 and 2026 according to the solar flares news on the internet.*

Total Raw data size: **39.8** GB  
Total size of final csv datset produced: **6.9** GB  
Row_count: **2592000**  
Feature_count: **227**  

## Models used:

- Random Forest
- XGBoost
- LightGBM
- Catboost

### Overall Best Model:
**Random Forest**  
Metrics from the latest run:  

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Accuracy** | **99.96%** | High classification accuracy on raw observations |
| **Precision** | **93.99%** | Very low false alarm rate |
| **Recall** | **99.07%** | Misses less than 1% of active flare intervals |
| **Macro F1 Score** | **98.22%** | Balanced performance across Quiet and Flare classes |
| **MCC** | **0.9648** | Strong correlation with ground truth |
| **ROC-AUC** | **0.99998** | Nearly perfect class separation capability |
| **PR-AUC** | **0.9977** | Outstanding Precision-Recall envelope |
| **Optimal Threshold** | **0.62** | Optimizes validation composite metric |
| **Training Time** | **125.19s** | Trained in ~2 minutes using 12 CPU cores |
| **Inference Time / Sample** | **0.94 μs** | Sub-microsecond latency per second of data |
| **Model Size** | **8.35 MB** | Compact model file storage size |
| **RAM Used** | **969.1 MB** | Peak memory consumption during training |

---

*Read [ABOUT.md](https://github.com/RamashrayS/Flaleon/blob/main/ABOUT.md) for more scientific details regarding the project.*  
*Read [INSTALLATION.md](https://github.com/RamashrayS/Flaleon/blob/main/INSTALLATION.md) for the setup guide.*


