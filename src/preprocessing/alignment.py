import pandas as pd
from src.utils.config import ALIGN_TIME_OFFSET, ALIGN_TOLERANCE

def align_payloads(df_solexs, df_helios, time_offset=ALIGN_TIME_OFFSET, tolerance=ALIGN_TOLERANCE):
    """
    Aligns SoLEXS and HEL1OS observations by applying a time shift to HEL1OS and
    performing a merge_asof based on nearest timestamps.
    
    Parameters:
        df_solexs (DataFrame): SoLEXS data containing TIME and COUNTS columns.
        df_helios (DataFrame): HEL1OS data containing TIME and COUNTS columns.
        time_offset (float): Time offset in seconds to subtract from HEL1OS timestamps.
        tolerance (float): Maximum time tolerance in seconds for aligning rows.
    """
    df_s = df_solexs.copy()
    if 'COUNTS' in df_s.columns:
        df_s = df_s.rename(columns={'COUNTS': 'solexs_counts'})
        
    df_h = df_helios.copy()
    if 'COUNTS' in df_h.columns:
        df_h = df_h.rename(columns={'COUNTS': 'helios_counts'})
        
    # Apply sub-second shift to HEL1OS timestamps to align with SoLEXS
    if time_offset is not None:
        df_h['TIME'] = df_h['TIME'] - time_offset
        
    df_s = df_s.sort_values('TIME')
    df_h = df_h.sort_values('TIME')
    
    # Merge using pd.merge_asof with direction='backward' to avoid future leakage
    df_merged = pd.merge_asof(
        df_s,
        df_h,
        on='TIME',
        direction='backward',
        tolerance=tolerance
    )
    
    # Fill values missing due to lack of overlap
    for col in df_merged.columns:
        if col != 'TIME':
            df_merged[col] = df_merged[col].fillna(0.0)
    
    return df_merged
