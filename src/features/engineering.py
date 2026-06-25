import pandas as pd
import numpy as np

def add_physics_features(df):
    """
    Computes physical and statistical features from aligned SoLEXS and HEL1OS count rates.
    Ensures that all features are computed only on historical and current observations,
    completely avoiding lookahead data leakage.
    
    Parameters:
        df (DataFrame): Aligned DataFrame containing 'TIME', 'solexs_counts', 'helios_counts', and optionally 'helios_czt_counts'.
    """
    df = df.sort_values('TIME').reset_index(drop=True)
    
    # Epsilon to avoid division by zero
    eps = 1e-6
    
    # ----------------------------------------------------
    # 0. Basic Setup
    # ----------------------------------------------------
    has_czt = 'helios_czt_counts' in df.columns
    payloads = ['solexs_counts', 'helios_counts']
    if has_czt:
        payloads.append('helios_czt_counts')
        
    dt = df['TIME'].diff().fillna(1.0)
    dt = dt.replace(0.0, 1.0) # Avoid division by zero in case of duplicate timestamps
    
    # ----------------------------------------------------
    # 1. Flux Derivatives (dFlux/dt and d2Flux/dt2)
    # ----------------------------------------------------
    df['solexs_dFlux_dt'] = df['solexs_counts'].diff().fillna(0.0) / dt
    df['helios_dFlux_dt'] = df['helios_counts'].diff().fillna(0.0) / dt
    if has_czt:
        df['helios_czt_dFlux_dt'] = df['helios_czt_counts'].diff().fillna(0.0) / dt
        
    # Second derivative (d2Flux/dt2)
    df['solexs_d2Flux_dt2'] = df['solexs_dFlux_dt'].diff().fillna(0.0) / dt
    df['helios_d2Flux_dt2'] = df['helios_dFlux_dt'].diff().fillna(0.0) / dt
    if has_czt:
        df['helios_czt_d2Flux_dt2'] = df['helios_czt_dFlux_dt'].diff().fillna(0.0) / dt
        
    # ----------------------------------------------------
    # 2. Hardness Features & Hardness Spectral Features
    # ----------------------------------------------------
    df['hardness_ratio'] = (df['helios_counts'] + eps) / (df['solexs_counts'] + eps)
    df['hardness_change'] = df['hardness_ratio'].diff().fillna(0.0) / dt
    
    if has_czt:
        # CZT to SoLEXS hardness
        df['hardness_ratio_czt_solexs'] = (df['helios_czt_counts'] + eps) / (df['solexs_counts'] + eps)
        df['hardness_change_czt_solexs'] = df['hardness_ratio_czt_solexs'].diff().fillna(0.0) / dt
        # CZT to CDTe hardness (HEL1OS-internal hardness)
        df['hardness_ratio_czt_cdte'] = (df['helios_czt_counts'] + eps) / (df['helios_counts'] + eps)
        df['hardness_change_czt_cdte'] = df['hardness_ratio_czt_cdte'].diff().fillna(0.0) / dt

    # Hardness Spectral rolling statistics (60s window)
    hardness_cols = ['hardness_ratio']
    if has_czt:
        hardness_cols.extend(['hardness_ratio_czt_solexs', 'hardness_ratio_czt_cdte'])
        
    for h_col in hardness_cols:
        df[f'{h_col}_roll_mean_60s'] = df[h_col].rolling(window=60, min_periods=1).mean()
        df[f'{h_col}_roll_std_60s'] = df[h_col].rolling(window=60, min_periods=1).std().fillna(0.0)
        df[f'{h_col}_roll_slope_60s'] = (df[h_col] - df[h_col].shift(60).fillna(0.0)) / 60.0

    # ----------------------------------------------------
    # 3. Rolling Statistics & Trend Features (existing + additions)
    # ----------------------------------------------------
    # Since sampling is 1s, windows are 30, 60, and 300 rows.
    windows = [30, 60, 300]
    
    for col in payloads:
        # Exponential Moving Averages (EMAs)
        for span in [10, 30, 60, 300]:
            df[f'{col}_ema_{span}s'] = df[col].ewm(span=span, adjust=False).mean()
            
        # EMA Differences
        df[f'{col}_ema_diff_10_60s'] = df[f'{col}_ema_10s'] - df[f'{col}_ema_60s']
        df[f'{col}_ema_diff_60_300s'] = df[f'{col}_ema_60s'] - df[f'{col}_ema_300s']

        for w in windows:
            # Rolling Mean (existing)
            df[f'{col}_roll_mean_{w}s'] = df[col].rolling(window=w, min_periods=1).mean()
            
            # Rolling Std (existing)
            df[f'{col}_roll_std_{w}s'] = df[col].rolling(window=w, min_periods=1).std().fillna(0.0)
            
            # Moving Average Ratio (existing)
            df[f'{col}_ma_ratio_{w}s'] = df[col] / (df[f'{col}_roll_mean_{w}s'] + eps)
            
            # Local Peak Prominence (existing)
            df[f'{col}_peak_prom_{w}s'] = df[col] - df[col].rolling(window=w, min_periods=1).min()
            
            # Rolling Median (new)
            df[f'{col}_roll_median_{w}s'] = df[col].rolling(window=w, min_periods=1).median()
            
            # Rolling Min (new)
            df[f'{col}_roll_min_{w}s'] = df[col].rolling(window=w, min_periods=1).min()
            
            # Rolling Max (new)
            df[f'{col}_roll_max_{w}s'] = df[col].rolling(window=w, min_periods=1).max()
            
            # Rolling Range (new)
            df[f'{col}_roll_range_{w}s'] = df[f'{col}_roll_max_{w}s'] - df[f'{col}_roll_min_{w}s']
            
            # Rolling RMS (new)
            df[f'{col}_roll_rms_{w}s'] = np.sqrt((df[col]**2).rolling(window=w, min_periods=1).mean() + eps)
            
            # Rolling Variance (new)
            df[f'{col}_roll_var_{w}s'] = df[col].rolling(window=w, min_periods=1).var().fillna(0.0)
            
            # Rolling Coefficient of Variation (new)
            df[f'{col}_roll_cv_{w}s'] = df[f'{col}_roll_std_{w}s'] / (df[f'{col}_roll_mean_{w}s'] + eps)

        # Rolling Slope (existing) for 30s and 60s windows
        for w in [30, 60]:
            df[f'{col}_slope_{w}s'] = (df[col] - df[col].shift(w).fillna(0.0)) / float(w)
            
        # Rolling acceleration (rolling mean of first derivative) for 30s and 60s windows
        for w in [30, 60]:
            deriv_col = f'{col.replace("_counts", "")}_dFlux_dt'
            if deriv_col in df.columns:
                df[f'{col}_acceleration_{w}s'] = df[deriv_col].rolling(window=w, min_periods=1).mean()

        # Lag Features (existing)
        lags = [10, 30, 60, 300]
        for lag in lags:
            df[f'{col}_lag_{lag}s'] = df[col].shift(lag).fillna(0.0)
            
        # ----------------------------------------------------
        # 4. Statistical Descriptors & Peak Behavior (60s and 300s)
        # ----------------------------------------------------
        for w in [60, 300]:
            # Z-Score (new)
            mean_val = df[f'{col}_roll_mean_{w}s']
            std_val = df[f'{col}_roll_std_{w}s']
            df[f'{col}_z_score_{w}s'] = (df[col] - mean_val) / (std_val + eps)
            
            # Local Anomaly Score (absolute Z-score)
            df[f'{col}_anomaly_score_{w}s'] = np.abs(df[f'{col}_z_score_{w}s'])
            
            # Skewness
            df[f'{col}_roll_skew_{w}s'] = df[col].rolling(window=w, min_periods=1).skew().fillna(0.0)
            
            # Kurtosis
            df[f'{col}_roll_kurt_{w}s'] = df[col].rolling(window=w, min_periods=1).kurt().fillna(0.0)
            
            # Rise ratio and decay ratio (new)
            rng = df[f'{col}_roll_range_{w}s']
            df[f'{col}_rise_ratio_{w}s'] = (df[col] - df[f'{col}_roll_min_{w}s']) / (rng + eps)
            df[f'{col}_decay_ratio_{w}s'] = (df[f'{col}_roll_max_{w}s'] - df[col]) / (rng + eps)

        # Distance from local maximum and minimum
        # Using index distance within rolling window to avoid slow python apply loops where possible.
        # We can implement a fast approximation using argmax/argmin on rolling windows of size 60 and 300
        for w in [60, 300]:
            # Use pandas rolling with custom functions for index distance
            # Since size is 60 and 300, we apply native argmax/argmin
            df[f'{col}_dist_max_{w}s'] = df[col].rolling(window=w, min_periods=1).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)
            df[f'{col}_dist_min_{w}s'] = df[col].rolling(window=w, min_periods=1).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)

    # ----------------------------------------------------
    # 5. Cross-Instrument Relationships
    # ----------------------------------------------------
    # Rolling correlation and covariance between SoLEXS and HEL1OS CDTe
    df['solexs_helios_corr_300s'] = df['solexs_counts'].rolling(window=300, min_periods=1).corr(df['helios_counts']).fillna(0.0)
    df['solexs_helios_cov_300s'] = df['solexs_counts'].rolling(window=300, min_periods=1).cov(df['helios_counts']).fillna(0.0)
    
    # Normalized flux ratio
    df['normalized_flux_ratio'] = (df['solexs_counts'] - df['helios_counts']) / (df['solexs_counts'] + df['helios_counts'] + eps)
    
    if has_czt:
        # Agreement between CDTe and CZT
        df['cdte_czt_agreement'] = (df['helios_counts'] - df['helios_czt_counts']) / (df['helios_counts'] + df['helios_czt_counts'] + eps)
        df['helios_czt_corr_300s'] = df['helios_counts'].rolling(window=300, min_periods=1).corr(df['helios_czt_counts']).fillna(0.0)
    else:
        df['cdte_czt_agreement'] = 0.0
        df['helios_czt_corr_300s'] = 0.0

    # ----------------------------------------------------
    # 6. Time Context
    # ----------------------------------------------------
    t_min = df['TIME'].min()
    t_max = df['TIME'].max()
    df['seconds_since_observation_start'] = df['TIME'] - t_min
    df['elapsed_observation_fraction'] = (df['TIME'] - t_min) / (t_max - t_min + eps)

    # Remove any NaN values that could remain
    df = df.fillna(0.0)
    
    return df
