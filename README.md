<p align="center">
<img width="3780" height="1890" alt="banner_flarion(1)" src="https://github.com/user-attachments/assets/4f7a9196-b45c-4d79-ad74-ceb16b169a3d" />
Flaleon is a ML framework to Forecast, Nowcast and Classify solar flares into X,M or C classes

## Data used

The models are trained on:

- SoLEXS (Solar Low Energy X-ray Spectrometer) &
- HEL1OS (High Energy L1 Orbiting X-ray Spectrometer)

sent by [AdityaL1]<https://www.isro.gov.in/Aditya_L1.html>. (available for download from the [ISSDC]<https://pradan1.issdc.gov.in/al1/> portal)

The training set was of 30 days (not continuous, the dates were particularly picked to make sure the models are fed healthy amounts of X,M,C class flares and quiet days)

Out of 30 days:
- X-class days: **6**
- M-class days: **5**
- C-class days: **7**
- Quiet days: **12**

*The dates were picked from 2024, 2025 and 2026 according to the solar flares news on the internet.*

Total Raw data size: **39.8** GB
Total size of final csv datset produced: **6.9** GB
Row_count: **2592000**
Feature_count: **227**

## Models used:

- Random Forest
- XGBoost
- LightGBM
- Catboost

##

*Read ABOUT.md for more scientific details regarding the project.*


