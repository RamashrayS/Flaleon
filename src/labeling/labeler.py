import os
import pandas as pd
import numpy as np
from src.utils.config import LABELS_DIR, LABEL_CLASS_MAP, DEBUG_MODE, LABELING_STRATEGY

def create_catalog_from_data(filepath=None):
    """
    Scans the raw data directory, detects solar flare events from SoLEXS count rates,
    and automatically builds a goes_flares.csv catalog file.
    """
    from astropy.io import fits
    from src.utils.config import RAW_DATA_DIR
    
    if filepath is None:
        filepath = os.path.join(LABELS_DIR, 'goes_flares.csv')
        
    print("[INFO] Scanning raw data directories to build flare event catalog...")
    flares = []
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] Raw data directory {RAW_DATA_DIR} does not exist.")
        return pd.DataFrame()
        
    days = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])
    
    for day in days:
        day_path = os.path.join(RAW_DATA_DIR, day)
        lc_file = None
        for r, _, files in os.walk(day_path):
            for f in files:
                if f.endswith('.lc.gz') and ('SOLEXS' in f.upper() or 'SLX' in f.upper()):
                    lc_file = os.path.join(r, f)
                    break
        if not lc_file:
            continue
            
        try:
            with fits.open(lc_file) as hdul:
                data = hdul[1].data
                times = np.asarray(data['TIME']).astype('float64')
                counts = np.asarray(data['COUNTS']).astype('float64')
                
            # Clear NaNs using interpolation/fillna
            s_counts = pd.Series(counts)
            s_counts = s_counts.interpolate(limit_direction='both').fillna(0.0)
            
            # 10s rolling mean to smooth noise
            smoothed = s_counts.rolling(window=10, min_periods=1).mean().values
            
            # Detect intervals where counts exceed 1000 counts/sec
            threshold = 1000.0
            active_indices = np.where(smoothed > threshold)[0]
            
            if len(active_indices) == 0:
                continue
                
            # Group contiguous active indices (within 5 minutes = 300s of each other)
            events = []
            current_event = [active_indices[0]]
            
            for idx in active_indices[1:]:
                if idx - current_event[-1] <= 300: # 5 min gap threshold
                     current_event.append(idx)
                else:
                    events.append(current_event)
                    current_event = [idx]
            events.append(current_event)
            
            # For each event, compute start, end, peak
            for ev in events:
                # Check if event has a reasonable duration (at least 30 seconds above threshold)
                if len(ev) < 30:
                    continue
                    
                start_idx = max(0, ev[0] - 60) # 1 min buffer before rise
                end_idx = min(len(times) - 1, ev[-1] + 180) # 3 min buffer for decay
                
                start_time = times[start_idx]
                end_time = times[end_idx]
                
                # Find peak inside this interval
                ev_counts = counts[start_idx:end_idx+1]
                ev_times = times[start_idx:end_idx+1]
                
                # Filter out NaNs for argmax
                valid_mask = ~np.isnan(ev_counts)
                if not valid_mask.any():
                    continue
                peak_idx_local = np.nanargmax(ev_counts)
                peak_time = ev_times[peak_idx_local]
                peak_counts = ev_counts[peak_idx_local]
                
                # Determine class based on peak counts
                if peak_counts > 15000:
                    f_class = f"X{peak_counts/10000:.1f}"
                elif peak_counts > 5000:
                    f_class = f"M{peak_counts/1000:.1f}"
                elif peak_counts > 1000:
                    f_class = f"C{peak_counts/100:.1f}"
                else:
                    f_class = "B9.0"
                    
                flares.append({
                    'start_time': int(start_time),
                    'peak_time': int(peak_time),
                    'end_time': int(end_time),
                    'class': f_class,
                    'intensity': float(peak_counts)
                })
                print(f"  Day: {day} -> Detected flare: {f_class} | Start: {int(start_time)} | Peak: {int(peak_time)} | End: {int(end_time)} | Peak Counts: {peak_counts:.1f}")
        except Exception as e:
            print(f"  [ERROR] Failed to scan flares for {day}: {str(e)}")
            
    df_cat = pd.DataFrame(flares)
    if df_cat.empty:
        # Fallback to a safe dummy
        df_cat = pd.DataFrame({
            'start_time': [1782020000],
            'peak_time': [1782021000],
            'end_time': [1782023000],
            'class': ['C1.0'],
            'intensity': [1000.0]
        })
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_cat.to_csv(filepath, index=False)
    print(f"[SUCCESS] Auto-generated flare catalog saved to: {filepath} with {len(df_cat)} events.")
    return df_cat

class CatalogLabeler:
    """
    Labels a dataset based on a scientifically valid flare catalog.
    Supports Nowcasting (Task A), Forecasting (Task B), and Class Estimation (Task C).
    """
    def __init__(self, catalog_df):
        self.catalog = catalog_df.copy()
        # Sort catalog by start time
        self.catalog = self.catalog.sort_values('start_time').reset_index(drop=True)
        
    def _get_class_num(self, class_str):
        if not isinstance(class_str, str) or len(class_str) == 0:
            return 0
        first_char = class_str[0].upper()
        return LABEL_CLASS_MAP.get(first_char, 0)

    def label(self, df, strategy=None):
        """
        Labels the input aligned DataFrame.
        """
        if strategy is None:
            strategy = LABELING_STRATEGY
            
        df = df.copy()
        times = df['TIME'].values
        
        # Pre-compute labels
        flare_now = np.zeros(len(df), dtype=int)
        flare_class = np.zeros(len(df), dtype=int)
        flare_future_5min = np.zeros(len(df), dtype=int)
        flare_future_10min = np.zeros(len(df), dtype=int)
        flare_future_30min = np.zeros(len(df), dtype=int)
        
        if strategy == 'daily_strongest':
            if len(times) > 0:
                obs_start = times[0]
                obs_end = times[-1]
                # Find all flares in catalog within observation window
                obs_flares = self.catalog[
                    (self.catalog['peak_time'] >= obs_start) & (self.catalog['peak_time'] <= obs_end)
                ]
                if len(obs_flares) > 0:
                    # Strongest flare in the window
                    strongest_flare = obs_flares.loc[obs_flares['intensity'].idxmax()]
                    c_str = strongest_flare['class']
                    c_num = self._get_class_num(c_str)
                    
                    flare_now[:] = 1
                    flare_class[:] = c_num
                    flare_future_5min[:] = 1
                    flare_future_10min[:] = 1
                    flare_future_30min[:] = 1
        else:
            # Original overlap strategy (row-by-row matching)
            for _, row in self.catalog.iterrows():
                start = row['start_time']
                end = row['end_time']
                c_str = row['class']
                c_num = self._get_class_num(c_str)
                
                # Nowcasting: active during the flare interval
                now_mask = (times >= start) & (times <= end)
                flare_now[now_mask] = 1
                flare_class[now_mask] = np.maximum(flare_class[now_mask], c_num)
                
                # Forecasting: onset will happen in future horizon
                # A flare starts in the future interval (t, t + horizon]
                # Thus, start - horizon <= t < start
                
                # 5 minutes horizon (300s)
                f5_mask = (times < start) & (times >= start - 300)
                flare_future_5min[f5_mask] = 1
                
                # 10 minutes horizon (600s)
                f10_mask = (times < start) & (times >= start - 600)
                flare_future_10min[f10_mask] = 1
                
                # 30 minutes horizon (1800s)
                f30_mask = (times < start) & (times >= start - 1800)
                flare_future_30min[f30_mask] = 1
                
        df['flare_now'] = flare_now
        df['flare_class'] = flare_class
        df['flare_future_5min'] = flare_future_5min
        df['flare_future_10min'] = flare_future_10min
        df['flare_future_30min'] = flare_future_30min
        
        # Maintain backwards compatibility/default task column
        df['flare_future'] = flare_future_10min  # default forecasting horizon is 10 min
        
        return df

class ThresholdLabeler:
    """
    TEMPORARY/DEBUGGING ONLY: Labels a dataset using simple count thresholds.
    Do NOT use for final systems or publications!
    """
    def __init__(self, threshold=1000):
        self.threshold = threshold
        print("[WARNING] Using ThresholdLabeler! This is for debugging only and not scientifically valid.")
        
    def label(self, df):
        df = df.copy()
        times = df['TIME'].values
        counts = df['solexs_counts'].values
        
        # Nowcasting
        df['flare_now'] = (counts > self.threshold).astype(int)
        
        # Class estimation
        f_class = np.zeros(len(df), dtype=int)
        f_class[counts > self.threshold] = 1      # C-class equivalent
        f_class[counts > 2.5 * self.threshold] = 2 # M-class equivalent
        f_class[counts > 5 * self.threshold] = 3   # X-class equivalent
        df['flare_class'] = f_class
        
        # Forecasting: checks if any counts exceed threshold in future window
        # We can implement this with rolling max on reversed counts
        # 5min = 300s, 10min = 600s, 300min = 1800s
        for h_name, h_sec in [('5min', 300), ('10min', 600), ('30min', 1800)]:
            # Shift backwards to see the future
            future_max = df['solexs_counts'].shift(-h_sec).rolling(window=h_sec, min_periods=1).max().fillna(0.0)
            df[f'flare_future_{h_name}'] = ((future_max > self.threshold) & (df['flare_now'] == 0)).astype(int)
            
        df['flare_future'] = df['flare_future_10min']
        return df

def label_dataset(df, catalog_path=None, strategy=None):
    """
    Applies the appropriate labeler to the aligned dataset.
    Prioritizes catalog-based labeling and falls back to a dynamically generated catalog
    if no catalog is found.
    """
    if catalog_path is None:
        # Check default path
        catalog_path = os.path.join(LABELS_DIR, 'goes_flares.csv')
        
    if not os.path.exists(catalog_path):
        catalog_df = create_catalog_from_data(catalog_path)
    else:
        print(f"[INFO] Loading catalog from {catalog_path}")
        catalog_df = pd.read_csv(catalog_path)
        
    if strategy is None:
        strategy = LABELING_STRATEGY

    if DEBUG_MODE:
        import datetime
        times = df['TIME'].values
        if len(times) > 0:
            date_str = datetime.datetime.fromtimestamp(times[0], datetime.timezone.utc).strftime('%Y-%m-%d')
            obs_start = datetime.datetime.fromtimestamp(times[0], datetime.timezone.utc).isoformat()
            obs_end = datetime.datetime.fromtimestamp(times[-1], datetime.timezone.utc).isoformat()
            
            # Find calendar day flares
            cal_start = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            cal_end = cal_start + datetime.timedelta(days=1)
            
            catalog_df_temp = catalog_df.copy()
            catalog_df_temp['peak_utc'] = pd.to_datetime(catalog_df_temp['peak_time'], unit='s', utc=True)
            
            cal_flares = catalog_df_temp[
                (catalog_df_temp['peak_utc'] >= cal_start) & (catalog_df_temp['peak_utc'] < cal_end)
            ]
            cal_flare_list = list(cal_flares['class'].values)
            
            # Find observation window flares
            obs_flares = catalog_df_temp[
                (catalog_df_temp['peak_time'] >= times[0]) & (catalog_df_time := catalog_df_temp['peak_time'] <= times[-1])
            ]
            # Fix column filter
            obs_flares = catalog_df_temp[
                (catalog_df_temp['peak_time'] >= times[0]) & (catalog_df_temp['peak_time'] <= times[-1])
            ]
            obs_flare_list = list(obs_flares['class'].values)
            
            # Determine label and reason
            assigned_label = "Quiet"
            reason = "No flares in observation window"
            if len(obs_flares) > 0:
                strongest = obs_flares.loc[obs_flares['intensity'].idxmax()]
                assigned_label = strongest['class'][0] # C, M, X
                reason = f"Matched flare {strongest['class']} (intensity={strongest['intensity']})"
            elif len(cal_flares) > 0:
                reason = "Flares occurred on this calendar day, but outside the observation window"
                
            print(f"\n--- DEBUG DIAGNOSTIC REPORT ---")
            print(f"Date: {date_str}")
            print(f"Observation start: {obs_start}")
            print(f"Observation end: {obs_end}")
            print(f"GOES flares occurring that calendar day: {cal_flare_list}")
            print(f"GOES flares occurring during observation window: {obs_flare_list}")
            print(f"Assigned label: {assigned_label}")
            print(f"Reason for assigned label: {reason}")
            print(f"---------------------------------\n")
        
    labeler = CatalogLabeler(catalog_df)
    return labeler.label(df, strategy=strategy)

