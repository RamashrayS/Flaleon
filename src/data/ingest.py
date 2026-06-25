import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from src.utils.config import RAW_DATA_DIR

def read_fits_lc(file_path, ext=None, time_col=None, count_col=None):
    """
    Reads a lightcurve from a FITS file and returns a Pandas DataFrame with TIME and COUNTS.
    Automatically handles conversion of MJD to Unix timestamp if time_col is MJD.
    
    Parameters:
        file_path (str): Path to the FITS file.
        ext (int or str, optional): FITS extension. If None, it attempts auto-detection.
        time_col (str, optional): Name of the time column. Autodetects 'TIME' or 'MJD'.
        count_col (str, optional): Name of the count/flux column. Autodetects 'COUNTS' or 'CTR'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FITS file not found: {file_path}")
        
    with fits.open(file_path) as hdul:
        # Auto-detect extension if not provided
        if ext is None:
            ext = 1  # Default to 1
            for i, hdu in enumerate(hdul):
                if hdu.name:
                    # 'RATE' is for SoLEXS, '1.80KEV_TO_90.00KEV' is the widest band for HEL1OS
                    if '1.80KEV_TO_90.00KEV' in hdu.name.upper() or 'RATE' in hdu.name.upper():
                        ext = i
                        break
        
        data = hdul[ext].data
        if data is None:
            raise ValueError(f"No data found in extension {ext} of {file_path}")
            
        columns = data.names
        
        # Auto-detect time column
        if time_col is None:
            if 'TIME' in columns:
                time_col = 'TIME'
            elif 'MJD' in columns:
                time_col = 'MJD'
            else:
                raise ValueError(f"Could not autodetect time column from {columns} in {file_path}")
                
        # Auto-detect count column
        if count_col is None:
            if 'COUNTS' in columns:
                count_col = 'COUNTS'
            elif 'CTR' in columns:
                count_col = 'CTR'
            else:
                raise ValueError(f"Could not autodetect count column from {columns} in {file_path}")
                
        time_vals = np.asarray(data[time_col]).astype('float64')
        count_vals = np.asarray(data[count_col]).astype('float64')
        
        df = pd.DataFrame({
            'TIME': time_vals,
            'COUNTS': count_vals
        })
        
        # If time is in Modified Julian Date (MJD), convert it to Unix Time
        if time_col == 'MJD':
            df['TIME'] = Time(df['TIME'], format='mjd', scale='utc').unix
            
        return df

def load_solexs_day(date_folder):
    """
    Discovers and loads all SoLEXS lightcurve files (.lc.gz) in a date folder.
    
    Parameters:
        date_folder (str): Path to date directory or a YYYY-MM-DD string.
    """
    # Resolve if date string is provided
    if not os.path.exists(date_folder):
        date_folder = os.path.join(RAW_DATA_DIR, date_folder)
        
    solexs_files = []
    for root, _, files in os.walk(date_folder):
        for file in files:
            # Check for SOLEXS or SLX in filename or directory name
            if file.endswith('.lc.gz') and (
                'SOLEXS' in file.upper() or 'SLX' in file.upper() or 
                'SOLEXS' in root.upper() or 'SLX' in root.upper()
            ):
                solexs_files.append(os.path.join(root, file))
                
    if not solexs_files:
        raise FileNotFoundError(f"No SoLEXS lightcurve (.lc.gz) files found in {date_folder}")
        
    dfs = []
    for filepath in solexs_files:
        df = read_fits_lc(filepath)
        dfs.append(df)
        
    if len(dfs) == 1:
        df_combined = dfs[0]
    else:
        df_combined = pd.concat(dfs, ignore_index=True)
        
    df_combined = df_combined.drop_duplicates(subset=['TIME']).sort_values('TIME').reset_index(drop=True)
    return df_combined

def load_helios_day(date_folder, detector='all'):
    """
    Discovers and loads all HEL1OS lightcurve files for a specific detector or all detectors combined.
    Concatenates multiple parts of the day (e.g. firsthalf and secondhalf).
    
    Parameters:
        date_folder (str): Path to date directory or a YYYY-MM-DD string.
        detector (str): HEL1OS detector to load ('cdte1', 'cdte2', 'czt1', 'czt2', or 'all').
    """
    if detector == 'all':
        # Safely load CDTe detectors
        dfs_cdte = []
        for det in ['cdte1', 'cdte2']:
            try:
                d = load_helios_day(date_folder, detector=det).rename(columns={'COUNTS': det}).drop_duplicates('TIME')
                dfs_cdte.append(d)
            except FileNotFoundError:
                pass
        
        if not dfs_cdte:
            raise FileNotFoundError(f"No CDTe detector files found in {date_folder}")
            
        if len(dfs_cdte) == 2:
            df_cdte = pd.merge(dfs_cdte[0], dfs_cdte[1], on='TIME', how='outer').sort_values('TIME')
            cols = [c for c in df_cdte.columns if c != 'TIME']
            for c in cols:
                df_cdte[c] = df_cdte[c].ffill().bfill().fillna(0.0)
            df_cdte['helios_counts'] = df_cdte[cols].mean(axis=1)
        else:
            df_cdte = dfs_cdte[0]
            col_name = [c for c in df_cdte.columns if c != 'TIME'][0]
            df_cdte['helios_counts'] = df_cdte[col_name]
            
        # Safely load CZT detectors
        dfs_czt = []
        for det in ['czt1', 'czt2']:
            try:
                d = load_helios_day(date_folder, detector=det).rename(columns={'COUNTS': det}).drop_duplicates('TIME')
                dfs_czt.append(d)
            except FileNotFoundError:
                pass
                
        if not dfs_czt:
            # If no CZT files exist, default CZT counts to 0
            df_czt = pd.DataFrame({'TIME': df_cdte['TIME'], 'helios_czt_counts': 0.0})
        else:
            if len(dfs_czt) == 2:
                df_czt = pd.merge(dfs_czt[0], dfs_czt[1], on='TIME', how='outer').sort_values('TIME')
                cols = [c for c in df_czt.columns if c != 'TIME']
                for c in cols:
                    df_czt[c] = df_czt[c].ffill().bfill().fillna(0.0)
                df_czt['helios_czt_counts'] = df_czt[cols].mean(axis=1)
            else:
                df_czt = dfs_czt[0]
                col_name = [c for c in df_czt.columns if c != 'TIME'][0]
                df_czt['helios_czt_counts'] = df_czt[col_name]
                
        # Merge CDTe and CZT averages
        df_combined = pd.merge(
            df_cdte[['TIME', 'helios_counts']], 
            df_czt[['TIME', 'helios_czt_counts']], 
            on='TIME', 
            how='outer'
        ).sort_values('TIME')
        
        df_combined['helios_counts'] = df_combined['helios_counts'].ffill().bfill().fillna(0.0)
        df_combined['helios_czt_counts'] = df_combined['helios_czt_counts'].ffill().bfill().fillna(0.0)
        
        return df_combined

    if not os.path.exists(date_folder):
        date_folder = os.path.join(RAW_DATA_DIR, date_folder)
        
    filename_pattern = f"lightcurve_{detector}.fits"
    helios_files = []
    
    for root, _, files in os.walk(date_folder):
        for file in files:
            # Check for HEL1OS or HLS in parent directories
            if file.lower() == filename_pattern.lower() and (
                'HEL1OS' in root.upper() or 'HLS' in root.upper()
            ):
                helios_files.append(os.path.join(root, file))
                
    if not helios_files:
        raise FileNotFoundError(f"No HEL1OS lightcurve files matching {filename_pattern} found in {date_folder}")
        
    dfs = []
    for filepath in helios_files:
        # Load from HDU 5 (or autodetect 1.80KEV_TO_90.00KEV)
        df = read_fits_lc(filepath)
        dfs.append(df)
        
    if len(dfs) == 1:
        df_combined = dfs[0]
    else:
        df_combined = pd.concat(dfs, ignore_index=True)
        
    df_combined = df_combined.drop_duplicates(subset=['TIME']).sort_values('TIME').reset_index(drop=True)
    return df_combined
