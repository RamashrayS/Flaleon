import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

RAW_DATA_DIR = 'data/raw'
days = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])

print(f"{'Day':<12} | {'CDTe1 Max':<10} | {'CDTe2 Max':<10} | {'CZT1 Max':<10} | {'CZT2 Max':<10}")
print("-" * 65)

for day in days:
    day_path = os.path.join(RAW_DATA_DIR, day)
    
    max_vals = {}
    for det in ['cdte1', 'cdte2', 'czt1', 'czt2']:
        filename_pattern = f"lightcurve_{det}.fits"
        h_file = None
        for r, _, files in os.walk(day_path):
            for f in files:
                if f.lower() == filename_pattern.lower() and ('HEL1OS' in r.upper() or 'HLS' in r.upper()):
                    h_file = os.path.join(r, f)
                    break
        if h_file:
            try:
                with fits.open(h_file) as hdul:
                    ext = 1
                    for i, hdu in enumerate(hdul):
                        if hdu.name and '1.80KEV_TO_90.00KEV' in hdu.name.upper():
                            ext = i
                    h_data = hdul[ext].data
                    ctr = np.asarray(h_data['CTR']).astype('float64')
                    max_vals[det] = np.nanmax(ctr)
            except Exception as e:
                max_vals[det] = -999.0
        else:
            max_vals[det] = -1.0
            
    print(f"{day:<12} | {max_vals['cdte1']:<10.1f} | {max_vals['cdte2']:<10.1f} | {max_vals['czt1']:<10.1f} | {max_vals['czt2']:<10.1f}")
