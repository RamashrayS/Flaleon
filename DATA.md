# Aditya-L1 FITS Data Documentation

This document describes the structure and format of the FITS (Flexible Image Transport System) data files used in the **Solar Flare Forecasting Project** for both the **SoLEXS** and **HEL1OS** payloads on board Aditya-L1.

---

## 1. SoLEXS (Solar Low Energy X-ray Spectrometer)
SoLEXS observes the Sun in the soft X-ray range (1 keV - 22 keV) and provides high cadence observations.

### Discovered Directory Layout
```
data/raw/<date>/SOLEXS_latest_data_<date>_v1.0/
├── SDD1/
│   └── AL1_SOLEXS_<date>_SDD1_L1.gti.gz
└── SDD2/
    ├── AL1_SOLEXS_<date>_SDD2_L1.gti.gz
    ├── AL1_SOLEXS_<date>_SDD2_L1.lc.gz
    └── AL1_SOLEXS_<date>_SDD2_L1.pi.gz
```

### File Breakdown

#### 1. Light Curve File (`.lc.gz`)
* **Purpose:** Stores the time-series count rates.
* **HDU Extension 1 (`RATE`):**
  * **Dimensions:** `86400R x 2C` (Corresponds to 86,400 seconds or 1 row per second, representing a complete 24-hour day).
  * **Columns:**
    * `TIME` (double): Spacecraft time in seconds.
    * `COUNTS` (double): Soft X-ray photon count rate per second.
  * **Sampling Cadence:** 1.0 second.

#### 2. Spectrum File (`.pi.gz`)
* **Purpose:** Contains the energy/pulse-height spectrum of detected events over time.
* **FITS Structure:** Includes table data mapping energy channels to counts.

#### 3. Good Time Intervals (`.gti.gz`)
* **Purpose:** Details intervals of reliable, noise-free observations.
* **Columns:**
  * `START` (double): Start of good time interval.
  * `STOP` (double): Stop of good time interval.

---

## 2. HEL1OS (High Energy L1 Orbiting X-ray Spectrometer)
HEL1OS observes the Sun in the hard X-ray range (10 keV - 150 keV), capturing highly energetic emissions from solar flares.

### Discovered Directory Layout
```
data/raw/<date>/HEL1OS_data_<date>/06/21/HLS_<date>_<time>_lev1_V111/
├── aux/
│   ├── gticdte1.fits
│   ├── gticdte2.fits
│   ├── gticzt1.fits
│   ├── gticzt2.fits
│   └── hk.fits
├── cdte/
│   ├── hel1os_cdte_spectra_cdte1.fits
│   ├── hel1os_cdte_spectra_cdte2.fits
│   ├── lightcurve_cdte1.fits
│   └── lightcurve_cdte2.fits
├── czt/
│   ├── hel1os_czt_spectra_czt1.fits
│   ├── hel1os_czt_spectra_czt2.fits
│   ├── lightcurve_czt1.fits
│   └── lightcurve_czt2.fits
└── events/
    └── evt.fits
```

### File Breakdown

#### 1. Light Curve Files (`lightcurve_cdte1.fits` / `lightcurve_cdte2.fits`)
* **Purpose:** Stores energy-band-specific count rates.
* **FITS Structure:** Contains multiple extensions corresponding to energy bands.
* **HDU Extensions & Columns:**
  * **HDU 1:** `CDTE1_LC_BAND_5.00KEV_TO_20.00KEV`
    * *Columns:* `MJD` (Modified Julian Date), `ISOT` (ISO timestamp), `CTR` (count rate), `STAT_ERR`.
  * **HDU 2:** `CDTE1_LC_BAND_20.00KEV_TO_30.00KEV`
  * **HDU 3:** `CDTE1_LC_BAND_30.00KEV_TO_40.00KEV`
  * **HDU 4:** `CDTE1_LC_BAND_40.00KEV_TO_60.00KEV`
  * **HDU 5:** `CDTE1_LC_BAND_1.80KEV_TO_90.00KEV` (Full Range / Total Counts)
    * *Columns:* `MJD` (Modified Julian Date), `ISOT` (ISO timestamp), `CTR` (total count rate), `STAT_ERR`.
* **Sampling Cadence:** Approximately 1.0 second.

#### 2. Spectra Files (`hel1os_cdte_spectra_cdte1.fits` / `hel1os_cdte_spectra_cdte2.fits`)
* **Purpose:** Captures the hard X-ray photon energy spectra over short time windows.

#### 3. Event File (`evt.fits`)
* **Purpose:** Event-level list containing individual photon arrivals, energy channel channels, and precise sub-second timings.
* **Size:** Typically large (100–500+ MB per day).

---

## 3. Timestamp Alignment Details
* **Time Scale Difference:** SoLEXS timestamps are recorded in elapsed spacecraft seconds (unix-like format starting at day boundaries), whereas HEL1OS counts are aligned to Modified Julian Date (`MJD`).
* **Alignment Methodology:**
  1. HEL1OS `MJD` values are converted to Unix Epoch timestamps using `astropy.time.Time`.
  2. A sub-second timing offset of **0.322 seconds** is subtracted from the HEL1OS timestamps to account for clock-synchronization differences.
  3. The datasets are merged using a backward-looking `pd.merge_asof` with a strict `nearest` search direction and a `1.0-second` matching tolerance.
