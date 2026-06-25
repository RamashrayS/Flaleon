import os
import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

RAW_DATA_DIR = 'data/raw'
CATALOG_PATH = 'data/labels/goes_flares.csv'

# Load GOES catalog
catalog = pd.read_csv(CATALOG_PATH)
catalog['start_utc'] = pd.to_datetime(catalog['start_time'], unit='s', utc=True)
catalog['peak_utc'] = pd.to_datetime(catalog['peak_time'], unit='s', utc=True)
catalog['end_utc'] = pd.to_datetime(catalog['end_time'], unit='s', utc=True)

dates = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])

print(f"Total dates found: {len(dates)}")
print(f"Total catalog flares: {len(catalog)}")

results = []

for date_str in dates:
    day_path = os.path.join(RAW_DATA_DIR, date_str)
    
    # 1. Find SoLEXS file
    solexs_file = None
    for r, _, files in os.walk(day_path):
        for f in files:
            if f.endswith('.lc.gz') and ('SOLEXS' in f.upper() or 'SLX' in f.upper()):
                solexs_file = os.path.join(r, f)
                break
    
    # 2. Find HEL1OS file (CDTe1)
    helios_file = None
    for r, _, files in os.walk(day_path):
        for f in files:
            if f.lower() == 'lightcurve_cdte1.fits' and ('HEL1OS' in r.upper() or 'HLS' in r.upper()):
                helios_file = os.path.join(r, f)
                break
                
    solexs_start_utc, solexs_end_utc = None, None
    helios_start_utc, helios_end_utc = None, None
    
    # Read SoLEXS times
    if solexs_file:
        try:
            with fits.open(solexs_file) as hdul:
                s_data = hdul[1].data
                s_times = np.asarray(s_data['TIME']).astype('float64')
                solexs_start_utc = datetime.datetime.fromtimestamp(s_times[0], datetime.timezone.utc)
                solexs_end_utc = datetime.datetime.fromtimestamp(s_times[-1], datetime.timezone.utc)
        except Exception as e:
            print(f"Error reading SoLEXS for {date_str}: {e}")
            
    # Read HEL1OS times
    if helios_file:
        try:
            with fits.open(helios_file) as hdul:
                ext = 1
                for i, hdu in enumerate(hdul):
                    if hdu.name and '1.80KEV_TO_90.00KEV' in hdu.name.upper():
                        ext = i
                h_data = hdul[ext].data
                h_mjds = np.asarray(h_data['MJD']).astype('float64')
                h_times_unix = Time(h_mjds, format='mjd', scale='utc').unix
                helios_start_utc = datetime.datetime.fromtimestamp(h_times_unix[0], datetime.timezone.utc)
                helios_end_utc = datetime.datetime.fromtimestamp(h_times_unix[-1], datetime.timezone.utc)
        except Exception as e:
            print(f"Error reading HEL1OS for {date_str}: {e}")
            
    # Observation start and end (aligned window)
    obs_start = None
    obs_end = None
    if solexs_start_utc and helios_start_utc:
        # The alignment uses pd.merge_asof with SoLEXS as left key, so the range is SoLEXS range
        obs_start = solexs_start_utc
        obs_end = solexs_end_utc
    elif solexs_start_utc:
        obs_start = solexs_start_utc
        obs_end = solexs_end_utc
        
    # Get flares occurring on this calendar day in UTC
    cal_start = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    cal_end = cal_start + datetime.timedelta(days=1)
    
    flares_cal_day = catalog[
        (catalog['peak_utc'] >= cal_start) & (catalog['peak_utc'] < cal_end)
    ]
    
    # Get flares occurring during observation window
    flares_obs_win = []
    if obs_start and obs_end:
        flares_obs_win = catalog[
            (catalog['peak_utc'] >= obs_start) & (catalog['peak_utc'] <= obs_end)
        ]
        
    # Compute assigned label for the day (e.g. max flare class during obs window)
    assigned_label = "Quiet"
    max_intensity = -1
    rejection_reason = "No flares in observation window"
    
    if len(flares_obs_win) > 0:
        # Find the strongest flare in obs window
        strongest_flare = flares_obs_win.loc[flares_obs_win['intensity'].idxmax()]
        assigned_label = strongest_flare['class'][0] # 'C', 'M', or 'X'
        rejection_reason = f"Matched flare {strongest_flare['class']} (intensity={strongest_flare['intensity']})"
    else:
        if len(flares_cal_day) > 0:
            rejection_reason = "Flares occurred on this calendar day, but outside the observation window"
            
    results.append({
        'date': date_str,
        'obs_start': obs_start.isoformat() if obs_start else None,
        'obs_end': obs_end.isoformat() if obs_end else None,
        'cal_flares': list(flares_cal_day['class'].values),
        'obs_flares': list(flares_obs_win['class'].values) if len(flares_obs_win) > 0 else [],
        'assigned_label': assigned_label,
        'reason': rejection_reason
    })

df_res = pd.DataFrame(results)
df_res.to_csv('audit_results.csv', index=False)
print("\nAudit completed. Saved to audit_results.csv")
for r in results:
    print(f"Date: {r['date']}")
    print(f"Observation start: {r['obs_start']}")
    print(f"Observation end: {r['obs_end']}")
    print(f"GOES flares occurring that calendar day: {r['cal_flares']}")
    print(f"GOES flares occurring during observation window: {r['obs_flares']}")
    print(f"Assigned label: {r['assigned_label']}")
    print(f"Reason for assigned label: {r['reason']}")
    print("-" * 50)
