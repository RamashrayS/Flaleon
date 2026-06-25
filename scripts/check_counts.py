import os
import numpy as np
import pandas as pd
from astropy.io import fits

RAW_DATA_DIR = 'data/raw'
days = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])

print(f"{'Day':<12} | {'SoLEXS File Found':<18} | {'Max Counts':<12} | {'Min Counts':<12} | {'Mean Counts':<12} | {'Rows':<6}")
print("-" * 80)

for day in days:
    day_path = os.path.join(RAW_DATA_DIR, day)
    lc_file = None
    for r, _, files in os.walk(day_path):
        for f in files:
            if f.endswith('.lc.gz') and ('SOLEXS' in f.upper() or 'SLX' in f.upper()):
                lc_file = os.path.join(r, f)
                break
                
    if not lc_file:
        print(f"{day:<12} | {'No':<18} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<6}")
        continue
        
    try:
        with fits.open(lc_file) as hdul:
            data = hdul[1].data
            counts = np.asarray(data['COUNTS']).astype('float64')
            s_counts = pd.Series(counts).interpolate(limit_direction='both').fillna(0.0)
            max_c = s_counts.max()
            min_c = s_counts.min()
            mean_c = s_counts.mean()
            rows = len(s_counts)
            print(f"{day:<12} | {'Yes':<18} | {max_c:<12.1f} | {min_c:<12.1f} | {mean_c:<12.1f} | {rows:<6}")
    except Exception as e:
        print(f"{day:<12} | {'Error reading':<18} | {str(e)[:30]:<40}")
