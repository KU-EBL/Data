# Missing-value handling examples for environmental data
# Run in Google Colab or a local Python environment.

import pandas as pd
import numpy as np

BASE = "."

# 1. Only a few missing values -> row removal
df = pd.read_csv(f"{BASE}/01_few_missing_row_removal.csv")
print("Before:", df.shape)
df_clean = df.dropna()
print("After:", df_clean.shape)
print(df_clean)

# 2. Missing values in a numeric variable -> mean or median imputation
df = pd.read_csv(f"{BASE}/02_numeric_mean_median_imputation.csv")
df["soil_moisture_mean"] = df["soil_moisture_pct"].fillna(df["soil_moisture_pct"].mean())
df["soil_moisture_median"] = df["soil_moisture_pct"].fillna(df["soil_moisture_pct"].median())
print(df)

# 3. Missing values in a categorical variable -> mode or "Unknown"
df = pd.read_csv(f"{BASE}/03_categorical_mode_unknown.csv")
mode_value = df["land_use"].mode()[0]
df["land_use_mode"] = df["land_use"].fillna(mode_value)
df["land_use_unknown"] = df["land_use"].fillna("Unknown")
print(df)

# 4. Missing values in time-series data -> interpolation or Kalman smoothing
df = pd.read_csv(f"{BASE}/04_timeseries_interpolation_kalman.csv", parse_dates=["datetime"])
df["linear_interpolation"] = df["temperature_C"].interpolate(method="linear")

# Kalman smoothing example (optional package)
# In Colab, first run: !pip install -q pykalman
try:
    from pykalman import KalmanFilter
    values = df["temperature_C"].to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    kf = KalmanFilter(
        initial_state_mean=np.nanmean(values),
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=0.1
    )
    state_means, _ = kf.smooth(masked_values)
    df["kalman_smoothing"] = state_means.ravel()
except ImportError:
    df["kalman_smoothing"] = np.nan
    print("Install pykalman to run Kalman smoothing: pip install pykalman")

print(df)

# 5. Missing values related to specific groups -> group-wise imputation
df = pd.read_csv(f"{BASE}/05_groupwise_imputation.csv")
group_median = df.groupby("land_use")["soil_moisture_pct"].transform("median")
df["soil_moisture_groupwise"] = df["soil_moisture_pct"].fillna(group_median)
print(df)

# 6. A large proportion of values is missing -> remove the variable or recollect data
df = pd.read_csv(f"{BASE}/06_high_missingness_remove_variable.csv")
missing_ratio = df.isna().mean() * 100
print("Missing ratio (%)")
print(missing_ratio)

threshold = 50  # remove variables with >50% missing values
cols_to_drop = missing_ratio[missing_ratio > threshold].index
df_reduced = df.drop(columns=cols_to_drop)
print("Dropped columns:", list(cols_to_drop))
print(df_reduced)

# 7. Missingness caused by measurement or recording errors -> check source and correct if possible
df = pd.read_csv(f"{BASE}/07_measurement_recording_error.csv")
error_rows = df[df["temperature_C"].isna() | df["rainfall_mm"].isna()]
print("Rows requiring source verification")
print(error_rows)

# Example: only correct values when a reliable external source is available.
# Do NOT invent replacement values.
# df.loc[df["site_id"] == "S03", "temperature_C"] = verified_temperature_value

# 8. Missing values with non-random patterns -> investigate the cause before imputation
df = pd.read_csv(f"{BASE}/08_nonrandom_missingness_investigate.csv")
df["temp_missing"] = df["temperature_C"].isna()

print(pd.crosstab(df["road_access"], df["temp_missing"], normalize="index"))
print(df.groupby("winter_sample_available")["temp_missing"].mean())

# If missingness is concentrated in high-elevation, poorly accessible sites,
# simple mean imputation may bias the dataset. Investigate the sampling process first.
