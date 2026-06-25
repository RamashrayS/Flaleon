import os
import json
import datetime
import pandas as pd
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.ingest import load_solexs_day, load_helios_day
from src.preprocessing.alignment import align_payloads
from src.features.engineering import add_physics_features
from src.labeling.labeler import label_dataset

def build_dataset(date_list, base_raw_dir=RAW_DATA_DIR, detector='all'):
    """
    Automated pipeline that:
      1. Discovers and loads SoLEXS and HEL1OS FITS data for a list of dates.
      2. Aligns their timestamps.
      3. Performs feature engineering.
      4. Generates catalog-based labels.
      5. Concatenates all processed data.
      
    Parameters:
        date_list (list of str): Dates to process (e.g. ['2026-06-21']).
        base_raw_dir (str): Path to raw data folder.
        detector (str): HEL1OS detector to load ('cdte1', 'cdte2', 'czt1', 'czt2', or 'all').
    """
    day_dfs = []
    processed_dates = []
    payloads_used = ['SoLEXS', f'HEL1OS_{detector}' if detector != 'all' else 'HEL1OS_combined_all']
    
    checkpoint_dir = os.path.join(PROCESSED_DATA_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for date_str in date_list:
        date_folder = os.path.join(base_raw_dir, date_str)
        if not os.path.exists(date_folder):
            print(f"[WARNING] Date folder {date_folder} does not exist. Skipping.")
            continue
            
        checkpoint_file = os.path.join(checkpoint_dir, f"{date_str}_processed.csv")
        from src.utils.config import FORCE_RERUN
        if not FORCE_RERUN and os.path.exists(checkpoint_file):
            print(f"[CHECKPOINT] Reusing daily checkpoint for {date_str}. Skipping processing.")
            try:
                df_labeled = pd.read_csv(checkpoint_file)
                day_dfs.append(df_labeled)
                processed_dates.append(date_str)
                continue
            except Exception as ce:
                print(f"[WARNING] Failed to load checkpoint for {date_str}: {ce}. Re-processing...")

        print(f"[INFO] Processing date: {date_str}...")
        try:
            # 1. Load Data
            df_solexs = load_solexs_day(date_folder)
            df_helios = load_helios_day(date_folder, detector=detector)
            
            # 2. Align Payloads
            df_aligned = align_payloads(df_solexs, df_helios)
            
            # 3. Engineer Features
            df_features = add_physics_features(df_aligned)
            
            # 4. Generate Labels
            df_labeled = label_dataset(df_features)
            
            # Save checkpoint
            df_labeled.to_csv(checkpoint_file, index=False)
            print(f"[CHECKPOINT] Saved checkpoint for {date_str} to {checkpoint_file}")
            
            day_dfs.append(df_labeled)
            processed_dates.append(date_str)
            print(f"[INFO] Successfully processed {date_str}. Rows: {len(df_labeled)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process date {date_str}: {str(e)}")
            
    if not day_dfs:
        raise ValueError("No datasets were successfully built. Check date paths and files.")
        
    df_final = pd.concat(day_dfs, ignore_index=True)
    df_final = df_final.sort_values('TIME').reset_index(drop=True)
    
    # Create metadata dictionary
    metadata = {
        'generation_timestamp': datetime.datetime.utcnow().isoformat(),
        'dates_processed': processed_dates,
        'payloads_used': payloads_used,
        'row_count': len(df_final),
        'feature_count': len(df_final.columns),
        'columns': list(df_final.columns),
        'missing_data_statistics': df_final.isnull().sum().to_dict()
    }
    
    return df_final, metadata

def save_processed_dataset(df, metadata, base_dir=PROCESSED_DATA_DIR, filename_prefix='dataset'):
    """
    Saves the processed dataset (either as parquet or falls back to csv)
    and exports the dataset metadata to dataset_info.json.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Save metadata
    meta_path = os.path.join(base_dir, f'{filename_prefix}_info.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Dataset metadata saved to: {meta_path}")
    
    # Save dataframe
    csv_path = os.path.join(base_dir, f'{filename_prefix}.csv')
    try:
        # Try writing to Parquet first if pyarrow is installed
        parquet_path = os.path.join(base_dir, f'{filename_prefix}.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"[INFO] Dataset saved to Parquet: {parquet_path}")
        # Save CSV copy as well for portability
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Dataset saved to CSV: {csv_path}")
    except ImportError:
        # Fallback to CSV
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Parquet library not found. Dataset saved to CSV: {csv_path}")

def load_processed_dataset(base_dir=PROCESSED_DATA_DIR, filename_prefix='dataset'):
    """
    Loads the saved dataset, preferring parquet and falling back to csv.
    """
    parquet_path = os.path.join(base_dir, f'{filename_prefix}_processed.parquet')
    csv_path = os.path.join(base_dir, f'{filename_prefix}.csv')
    
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
            
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
        
    raise FileNotFoundError(f"No processed dataset files found in {base_dir}")
