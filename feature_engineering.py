"""
feature_engineering.py
=======================
Transforms the cleaned HDB Resale dataset (hdb_resale.db) into ML-ready
train / validation / test splits, guided by EDA findings.

EDA-driven decisions
--------------------
Plot 13 & 15 (skewness / log-transform check):
  - resale_price is heavily right-skewed → log1p(price) as target
  - floor_area_sqm is mildly skewed → kept as-is (tree models handle this)

Plot 14 (feature correlations with price):
  - Strong: floor_area_sqm, flat_type, storey_midpoint, remaining_lease, town, year
  - Moderate: flat_model, lease_commence_date, flat_age (derived)
  - Dropped: remaining_lease_months (redundant), block/street_name (too high cardinality)

Plot 17 (outlier analysis):
  - Outliers are genuine anomalies, not valid extremes → rows outside 1.5×IQR
    per flat_type are removed (not capped)

Encoding strategy:
  - flat_type   → ordinal (1-room=1 … multi-generation=7)
  - town        → target encoding (mean log_price per town, fit on train only)
  - flat_model  → target encoding (mean log_price per flat_model, fit on train only)

Split:
  - Train : ≤ 2020
  - Val   : 2021–2022
  - Test  : ≥ 2023

Outputs → model_assets/<run_timestamp>/
  X_train / X_val / X_test  (.parquet)
  y_train / y_val / y_test  (.parquet)
  target_encoders.pkl
  scaler.pkl
  feature_cols.txt
  outlier_bounds.json
  run_manifest.json
  metrics.json (stub — populate with real MAE after model training)

Run:
    python feature_engineering.py
"""

import json
import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH    = "hdb_resale.db"
TABLE_NAME = "resale_prices"
OUTPUT_DIR = "model_assets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Quantiles for prediction interval support.
# Model trained three times using quantile regression:
# Q10 (lower bound), Q50 (point estimate), Q90 (upper bound).
# Supports FR2: system returns a value range, not a single price.
QUANTILES = [0.10, 0.50, 0.90]

# HDB mature estates (official designation)
MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "HOUGANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES",
    "TOA PAYOH",
}

# Ordinal encoding for flat_type (room count order)
FLAT_TYPE_ORDINAL = {
    "1 Room": 1, "2 Room": 2, "3 Room": 3,
    "4 Room": 4, "5 Room": 5,
    "Executive": 6, "Multi-Generation": 7,
}

# Numeric features to scale (excludes binary + ordinal + target-encoded)
SCALE_COLS = [
    "floor_area_sqm",
    "storey_midpoint",
    "flat_age",
    "remaining_lease",
    "lease_commence_date",
    "month_sin",
    "month_cos",
    "year",
    "dist_mrt",    # km to nearest MRT station (from proximity_features.py)
    "dist_cbd",    # km to Raffles Place CBD anchor
    "dist_school", # km to nearest primary school
    "dist_mall",   # km to nearest shopping mall
]

# Model trained three times using quantile regression:
# Q10 (lower bound), Q50 (point estimate), Q90 (upper bound).
# Supports FR2: system returns a value range, not a single price.
# Final feature columns fed to the ML model
FEATURE_COLS = [
    "flat_type_ordinal",   # ordinal int
    "town_enc",            # target encoded (mean log_price)
    "flat_model_enc",      # target encoded (mean log_price)
    "floor_area_sqm",      # scaled
    "storey_midpoint",     # scaled
    "flat_age",            # scaled
    "remaining_lease",     # scaled (years)
    "lease_commence_date", # scaled
    "month_sin",           # scaled cyclical
    "month_cos",           # scaled cyclical
    "year",                # scaled
    "is_mature_estate",    # binary 0/1
    "dist_mrt",            # scaled km — proximity signal
    "dist_cbd",            # scaled km — centrality signal
    "dist_school",         # scaled km — amenity signal
    "dist_mall",           # scaled km — amenity signal
]

TARGET_COL = "log_price"   # log1p(resale_price)


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print(f"Loading from {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    numeric = [
        "floor_area_sqm", "storey_midpoint", "remaining_lease",
        "remaining_lease_months", "lease_commence_date",
        "resale_price", "year", "month_num",
        "latitude", "longitude",
        "dist_mrt", "dist_cbd", "dist_school", "dist_mall",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["town", "flat_type", "flat_model"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"  Loaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# Step 2: Remove outliers (1.5×IQR per flat_type)
#
# EDA Plot 17 showed genuine anomalies above the whiskers in every flat_type
# group — these are removed rather than capped to keep the distribution clean.
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove price outliers using 1.5×IQR per flat_type group.

    EDA Plot 17 showed genuine anomalies above the whiskers in every flat_type
    group — these are removed rather than capped to keep the distribution clean.

    Also serialises IQR bounds to model_assets/outlier_bounds.json for use
    at inference time to apply consistent outlier handling on new predictions.

    Parameters:
        df: Cleaned DataFrame with resale_price and flat_type columns.

    Returns:
        DataFrame with outlier rows removed.
    """
    before = len(df)
    keep = pd.Series(True, index=df.index)
    bounds: dict[str, dict[str, float]] = {}

    for ft, idx in df.groupby("flat_type").groups.items():
        prices = df.loc[idx, "resale_price"]
        q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep[idx] = prices.between(lo, hi)
        bounds[str(ft)] = {"lower": round(float(lo), 2), "upper": round(float(hi), 2)}

    df = df[keep].copy()
    print(f"\nOutlier removal: dropped {before - len(df):,} rows "
          f"({(before - len(df)) / before * 100:.1f}%). "
          f"Remaining: {len(df):,}")

    # Serialise bounds for inference-time outlier handling
    bounds_path = os.path.join(OUTPUT_DIR, "outlier_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"  outlier_bounds.json saved to {bounds_path}")

    return df


# ---------------------------------------------------------------------------
# Step 3: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering features ...")

    # Flat age at time of sale
    df["flat_age"] = (df["year"] - df["lease_commence_date"]).clip(lower=0)

    # Cyclical month encoding — avoids false "Dec→Jan = 11 gap"
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # Mature estate flag
    df["is_mature_estate"] = df["town"].isin(MATURE_ESTATES).astype(int)

    # Ordinal flat_type
    df["flat_type_ordinal"] = df["flat_type"].map(FLAT_TYPE_ORDINAL)
    unmapped = df["flat_type_ordinal"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped:,} unmapped flat_type rows — filling with median.")
        df["flat_type_ordinal"] = df["flat_type_ordinal"].fillna(
            df["flat_type_ordinal"].median()
        )

    # Log-transform target (EDA Plot 15: log price is near-normal)
    df[TARGET_COL] = np.log1p(df["resale_price"])

    print(f"  Features added: flat_age, month_sin, month_cos, "
          f"is_mature_estate, flat_type_ordinal, log_price")
    return df


# ---------------------------------------------------------------------------
# Step 4: Temporal train / val / test split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2020].copy()
    val   = df[(df["year"] >= 2021) & (df["year"] <= 2022)].copy()
    test  = df[df["year"] >= 2023].copy()

    total = len(df)
    print(f"\nTemporal split (total {total:,} rows):")
    print(f"  Train (≤ 2020) : {len(train):>8,}  ({len(train)/total*100:.1f}%)")
    print(f"  Val  (2021–22) : {len(val):>8,}  ({len(val)/total*100:.1f}%)")
    print(f"  Test (≥ 2023)  : {len(test):>8,}  ({len(test)/total*100:.1f}%)")
    return train, val, test


# ---------------------------------------------------------------------------
# Step 5: Target encoding for town and flat_model
#
# EDA Plot 14: both have strong non-linear price signals.
# Target encoding (mean log_price per category) captures this directly
# without inflating dimensionality like one-hot encoding would.
# Fit ONLY on train to prevent leakage.
# ---------------------------------------------------------------------------

def target_encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    print("\nTarget encoding town and flat_model ...")
    global_mean = train[TARGET_COL].mean()
    encoders = {}

    for col in ["town", "flat_model"]:
        means = train.groupby(col)[TARGET_COL].mean()
        encoders[col] = {"means": means, "global_mean": global_mean}

        enc_col = f"{col}_enc"
        train[enc_col] = train[col].map(means).fillna(global_mean)
        val[enc_col]   = val[col].map(means).fillna(global_mean)
        test[enc_col]  = test[col].map(means).fillna(global_mean)

        print(f"  {col}: {len(means)} categories encoded")

    return train, val, test, encoders


# ---------------------------------------------------------------------------
# Step 6: Scale numeric features (fit on train only)
# ---------------------------------------------------------------------------

def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:

    print("\nScaling numeric features ...")
    scaler = StandardScaler()

    train[SCALE_COLS] = scaler.fit_transform(train[SCALE_COLS])
    val[SCALE_COLS]   = scaler.transform(val[SCALE_COLS])
    test[SCALE_COLS]  = scaler.transform(test[SCALE_COLS])

    print(f"  StandardScaler fitted on train, applied to val and test.")
    return train, val, test, scaler


# ---------------------------------------------------------------------------
# Step 7: Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_encoders: dict,
    scaler: StandardScaler,
) -> str:
    """
    Save all ML artefacts into a timestamped versioned run directory.

    Creates model_assets/<YYYYMMDD_HHMMSS>/ for each run and writes
    model_assets/latest.txt pointing to the new directory. Also saves
    run_manifest.json as an audit trail (NFR Transparency & Explainability).

    Parameters:
        train, val, test:   Split DataFrames after encoding and scaling.
        target_encoders:    Dict of target-encoding maps fitted on train.
        scaler:             Fitted StandardScaler instance.

    Returns:
        Path to the created run directory.
    """
    run_dir = os.path.join(OUTPUT_DIR, datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nSaving artefacts to '{run_dir}/' ...")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        X = split[FEATURE_COLS]
        y = split[[TARGET_COL, "resale_price"]]  # keep raw price for evaluation

        X.to_parquet(os.path.join(run_dir, f"X_{name}.parquet"), index=False)
        y.to_parquet(os.path.join(run_dir, f"y_{name}.parquet"), index=False)
        print(f"  X_{name}.parquet {X.shape}   y_{name}.parquet {y.shape}")

    with open(os.path.join(run_dir, "target_encoders.pkl"), "wb") as f:
        pickle.dump(target_encoders, f)
    print("  target_encoders.pkl")

    with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("  scaler.pkl")

    with open(os.path.join(run_dir, "feature_cols.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))
    print("  feature_cols.txt")

    # Run manifest — audit trail for NFR Transparency & Explainability
    manifest = {
        "run_at":       datetime.utcnow().isoformat(),
        "train_rows":   len(train),
        "val_rows":     len(val),
        "test_rows":    len(test),
        "feature_cols": FEATURE_COLS,
        "scale_cols":   SCALE_COLS,
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  run_manifest.json")

    # Stub metrics.json — populate with real MAE after model training step
    metrics_stub = {
        "q10_test_mae": None,
        "q50_test_mae": None,
        "q90_test_mae": None,
        "note": "Populate with actual model metrics after training.",
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_stub, f, indent=2)
    print("  metrics.json (stub)")

    # Update latest.txt to point to this run (overwrite each time)
    latest_path = os.path.join(OUTPUT_DIR, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(run_dir)
    print(f"  latest.txt → {run_dir}")

    return run_dir


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(train: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY (train set)")
    print("=" * 60)
    stats = train[FEATURE_COLS + [TARGET_COL]].describe().T
    print(stats[["mean", "std", "min", "max"]].to_string())
    print(f"\n  Null values: {train[FEATURE_COLS].isnull().sum().sum()} "
          f"(should be 0)")
    print("=" * 60)
    print("\nNext step: run ml_model.py to train XGBoost and Random Forest.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("HDB Resale — Feature Engineering (EDA-guided)")
    print("=" * 60)

    df = load_data()
    df = remove_outliers(df)
    df = engineer_features(df)

    train, val, test = temporal_split(df)
    train, val, test, target_encoders = target_encode(train, val, test)
    train, val, test, scaler = scale_features(train, val, test)

    print_summary(train)
    run_dir = save_artefacts(train, val, test, target_encoders, scaler)

    print(f"\nDone. ML-ready data saved to: {run_dir}/")


if __name__ == "__main__":
    main()
