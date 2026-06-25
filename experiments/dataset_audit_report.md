# Dataset Audit & Validation Report
Generated on: 2026-06-25T10:15:19.469871 UTC

## 1. General Dataset Properties
- **Total Row Count**: 2,592,000
- **Total File Size**: 6.46 GB

### Target Column Missing Values:
- **TIME**: 0 missing values
- **flare_now**: 0 missing values
- **flare_class**: 0 missing values
- **flare_future_5min**: 0 missing values
- **flare_future_10min**: 0 missing values
- **flare_future_30min**: 0 missing values
- **flare_future**: 0 missing values
- **Duplicate Timestamps**: 0 duplicate observations
- **Chronologically Sorted**: True
- **Mean Time Step (dt)**: 28.6000 seconds
- **Min Time Step (dt)**: 1.0000 seconds
- **Max Time Step (dt)**: 17884801.0000 seconds

## 2. Chronological Splits Validation
- **Train set size**: 1,814,400 rows (1708041600.0 to 1777075199.0)
- **Val set size**: 388,800 rows (1777334400.0 to 1781783999.0)
- **Test set size**: 388,800 rows (1781784000.0 to 1782172799.0)
- **Train-Validation Overlap Leakage**: False
- **Validation-Test Overlap Leakage**: False

## 3. Label Balance & Positive/Negative Ratios

| Split | Task | Positive Samples | Negative Samples | Positive % |
| --- | --- | --- | --- | --- |
| Train | flare_now | 1,296,000 | 518,400 | 71.4286% |
| Train | flare_future_10min | 1,296,000 | 518,400 | 71.4286% |
| Val | flare_now | 172,800 | 216,000 | 44.4444% |
| Val | flare_future_10min | 172,800 | 216,000 | 44.4444% |
| Test | flare_now | 86,400 | 302,400 | 22.2222% |
| Test | flare_future_10min | 86,400 | 302,400 | 22.2222% |

### Multiclass (flare_class) Distribution:
| Split | Class 0 (Quiet) | Class 1 (C) | Class 2 (M) | Class 3 (X) |
| --- | --- | --- | --- | --- |
| Train | 518,400 | 259,200 | 345,600 | 691,200 |
| Val | 216,000 | 86,400 | 86,400 | 0 |
| Test | 302,400 | 86,400 | 0 | 0 |

## 4. Feature Quality Check (Subsample Stats)
- **Feature Duplication (excl. time/labels)**: 24 duplicated rows (0.02%)
- **Features with Missing Values**: 0 features
- **Zero-variance (Constant) Features**: 0 features

## 5. Potential Leakage & Suspicious Correlations
- **Features with correlation >= 0.99 with target**: 0 features
  - No individual feature has a suspicious correlation >= 0.99 with the target label.
- **Lookahead features inside training pool**: 0 features

## 6. Top Features by Pearson and Spearman Correlation with `flare_now`

### Top Pearson Correlation:
| Feature | Pearson Correlation |
| --- | --- |
| helios_counts_roll_skew_60s | -0.3353 |
| helios_counts_roll_cv_60s | -0.3327 |
| helios_counts_roll_cv_30s | -0.3212 |
| helios_counts_roll_cv_300s | -0.3186 |
| helios_czt_counts_roll_cv_60s | -0.3164 |
| helios_czt_counts_roll_cv_30s | -0.3115 |
| helios_counts_roll_skew_300s | -0.3058 |
| helios_czt_counts_roll_skew_60s | -0.2950 |
| helios_counts_roll_kurt_60s | -0.2943 |
| helios_czt_counts_roll_cv_300s | -0.2840 |
| helios_czt_counts_roll_skew_300s | -0.2529 |
| helios_counts_decay_ratio_300s | -0.2225 |
| helios_czt_counts_roll_kurt_60s | -0.2217 |
| helios_counts_decay_ratio_60s | -0.2122 |
| helios_counts_roll_kurt_300s | -0.2113 |

### Top Spearman Rank Correlation:
| Feature | Spearman Correlation |
| --- | --- |
| hardness_ratio_czt_solexs_roll_std_60s | -0.4576 |
| hardness_ratio_czt_solexs_roll_mean_60s | -0.4189 |
| solexs_counts_roll_cv_30s | -0.3716 |
| solexs_counts_roll_min_300s | 0.3700 |
| solexs_counts_roll_min_60s | 0.3634 |
| solexs_counts_roll_min_30s | 0.3593 |
| solexs_counts_ema_300s | 0.3550 |
| solexs_counts_roll_cv_60s | -0.3548 |
| solexs_counts_roll_rms_300s | 0.3536 |
| solexs_counts_roll_mean_300s | 0.3536 |
| solexs_counts_roll_max_300s | 0.3533 |
| solexs_counts_ema_60s | 0.3530 |
| solexs_counts_ema_30s | 0.3528 |
| solexs_counts_roll_median_300s | 0.3527 |
| solexs_counts_roll_rms_60s | 0.3525 |

## 7. Predictive Signal Assessment (Mutual Information)
| Feature | Mutual Information Score |
| --- | --- |
| solexs_counts_roll_cv_300s | 0.1708 |
| hardness_ratio_czt_solexs_roll_mean_60s | 0.1639 |
| hardness_ratio_czt_solexs_roll_std_60s | 0.1524 |
| solexs_counts_ema_300s | 0.1467 |
| solexs_counts_roll_cv_30s | 0.1454 |
| solexs_counts_roll_var_300s | 0.1290 |
| solexs_counts_roll_range_300s | 0.1248 |
| solexs_counts_roll_var_60s | 0.1231 |
| helios_counts_roll_cv_300s | 0.1222 |
| solexs_counts_roll_range_60s | 0.1211 |
| solexs_counts_lag_300s | 0.1200 |
| hardness_ratio_czt_cdte_roll_std_60s | 0.1195 |
| helios_counts_roll_max_300s | 0.1166 |
| helios_czt_counts_roll_cv_300s | 0.1140 |
| helios_counts_roll_var_300s | 0.1139 |
| hardness_ratio_czt_cdte_roll_mean_60s | 0.1037 |
| helios_czt_counts_roll_mean_300s | 0.1012 |
| helios_counts_roll_kurt_60s | 0.0980 |
| helios_czt_counts_roll_std_300s | 0.0937 |
| helios_czt_counts_roll_var_300s | 0.0936 |
| helios_counts_z_score_300s | 0.0933 |
| helios_czt_counts_z_score_300s | 0.0933 |
| hardness_ratio_roll_mean_60s | 0.0932 |
| helios_czt_counts_roll_kurt_300s | 0.0925 |
| helios_czt_counts_roll_cv_60s | 0.0910 |
| helios_counts_roll_kurt_300s | 0.0891 |
| solexs_counts_ema_diff_60_300s | 0.0765 |
| helios_czt_corr_300s | 0.0632 |
| helios_czt_counts_dist_max_300s | 0.0312 |
| elapsed_observation_fraction | 0.0000 |

## 8. Permutation Importance (on validation split)
Random Forest Validation Macro F1: 0.5032

| Feature | Permutation Importance Mean | Permutation Importance Std |
| --- | --- | --- |
| helios_counts_roll_var_300s | 0.0082 | 0.0007 |
| hardness_ratio_czt_cdte_roll_mean_60s | 0.0077 | 0.0007 |
| helios_counts_roll_max_300s | 0.0066 | 0.0008 |
| hardness_ratio_czt_cdte_roll_std_60s | 0.0043 | 0.0005 |
| helios_czt_counts_roll_kurt_300s | 0.0021 | 0.0010 |
| solexs_counts_roll_cv_300s | 0.0018 | 0.0006 |
| solexs_counts_lag_300s | 0.0017 | 0.0006 |
| solexs_counts_roll_cv_30s | 0.0009 | 0.0004 |
| helios_czt_counts_roll_cv_60s | 0.0004 | 0.0006 |
| elapsed_observation_fraction | 0.0003 | 0.0008 |
| helios_counts_roll_kurt_60s | 0.0001 | 0.0002 |
| helios_czt_counts_dist_max_300s | -0.0001 | 0.0001 |
| helios_czt_counts_roll_std_300s | -0.0001 | 0.0004 |
| solexs_counts_ema_diff_60_300s | -0.0010 | 0.0008 |
| helios_czt_counts_roll_var_300s | -0.0011 | 0.0005 |
| helios_counts_roll_kurt_300s | -0.0012 | 0.0005 |
| solexs_counts_roll_var_300s | -0.0012 | 0.0005 |
| helios_czt_counts_z_score_300s | -0.0013 | 0.0003 |
| helios_counts_z_score_300s | -0.0016 | 0.0006 |
| helios_czt_corr_300s | -0.0017 | 0.0005 |
| solexs_counts_roll_var_60s | -0.0018 | 0.0006 |
| hardness_ratio_roll_mean_60s | -0.0023 | 0.0004 |
| solexs_counts_ema_300s | -0.0026 | 0.0005 |
| helios_czt_counts_roll_cv_300s | -0.0027 | 0.0007 |
| solexs_counts_roll_range_60s | -0.0034 | 0.0004 |
| solexs_counts_roll_range_300s | -0.0034 | 0.0005 |
| hardness_ratio_czt_solexs_roll_std_60s | -0.0039 | 0.0007 |
| helios_czt_counts_roll_mean_300s | -0.0055 | 0.0009 |
| hardness_ratio_czt_solexs_roll_mean_60s | -0.0067 | 0.0007 |
| helios_counts_roll_cv_300s | -0.0073 | 0.0009 |

## 9. Bottleneck Assessment & Recommendations

### Identified Bottlenecks:
#### [CRITICAL] LABELS BOTTLENECK
- `LABELING_STRATEGY` is set to `'daily_strongest'`. This labels *all* seconds of a flare-containing day as positive, leading to massive lookahead/spillover leakage and flat labels per-day. The model cannot learn transient flare physical patterns, but rather just page-level daily signatures.

### Recommendations for Pipeline Optimization:
1. **Transition Labeling Strategy**: Change `LABELING_STRATEGY` from `'daily_strongest'` back to `'overlap'` to generate temporally precise target boundaries. This is the root cause of flat predictions and poor learning behavior.
2. **Address Class Imbalance**: Ensure class weights or balanced sampling are kept on, as the flare class is extremely rare (~0.2% positive sample rate when using overlap strategy).
3. **Downcast Data**: Keep memory-efficient downcasting (float32/int32) to prevent out-of-memory crashes on this 2.59M row dataset.