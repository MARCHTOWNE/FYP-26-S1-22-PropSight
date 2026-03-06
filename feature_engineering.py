import json
import os
import pickle
import sqlite3
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = "hdb_resale.db"
TABLE_NAME = "resale_prices"
OUTPUT_DIR = "model_assets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

QUANTILES = [0.10, 0.50, 0.90]

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "HOUGANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES",
    "TOA PAYOH",
}

FLAT_TYPE_ORDINAL = {
    "1 Room": 1,
    "2 Room": 2,
    "3 Room": 3,
    "4 Room": 4,
    "5 Room": 5,
    "Executive": 6,
    "Multi-Generation": 7,
}

SCALE_COLS = [
    "floor_area_sqm",
    "storey_midpoint",
    "flat_age",
    "remaining_lease",
    "lease_commence_date",
    "month_sin",
    "month_cos",
    "year",
    "dist_mrt",
    "dist_cbd",
    "dist_school",
    "dist_mall",
]

FEATURE_COLS = [
    "flat_type_ordinal",
    "town_enc",
    "flat_model_enc",
    "floor_area_sqm",
    "storey_midpoint",
    "flat_age",
    "remaining_lease",
    "lease_commence_date",
    "month_sin",
    "month_cos",
    "year",
    "is_mature_estate",
    "dist_mrt",
    "dist_cbd",
    "dist_school",
    "dist_mall",
]

TARGET_COL = "log_price"

REQUIRED_MODEL_COLS = [
    "town",
    "flat_type",
    "flat_model",
    "month_num",
    "year",
    "floor_area_sqm",
    "storey_midpoint",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
    "dist_mrt",
    "dist_cbd",
    "dist_school",
    "dist_mall",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print(f"Loading from {DB_PATH} ...")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    numeric_cols = [
        "floor_area_sqm",
        "storey_midpoint",
        "remaining_lease",
        "remaining_lease_months",
        "lease_commence_date",
        "resale_price",
        "year",
        "month_num",
        "latitude",
        "longitude",
        "dist_mrt",
        "dist_cbd",
        "dist_school",
        "dist_mall",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["town", "flat_type", "flat_model"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"  Loaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# Remove outliers
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    keep_mask = pd.Series(True, index=df.index)
    bounds: dict[str, dict[str, float]] = {}

    for flat_type, idx in df.groupby("flat_type").groups.items():
        prices = df.loc[idx, "resale_price"]
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        keep_mask.loc[idx] = prices.between(lower, upper)
        bounds[str(flat_type)] = {
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2),
        }

    df = df[keep_mask].copy()

    dropped = before - len(df)
    print(
        f"\nOutlier removal: dropped {dropped:,} rows "
        f"({dropped / before * 100:.1f}%). Remaining: {len(df):,}"
    )

    bounds_path = os.path.join(OUTPUT_DIR, "outlier_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    print(f"  outlier_bounds.json saved to {bounds_path}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering features ...")

    df["flat_age"] = (df["year"] - df["lease_commence_date"]).clip(lower=0)

    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df["is_mature_estate"] = df["town"].isin(MATURE_ESTATES).astype(int)

    df["flat_type_ordinal"] = df["flat_type"].map(FLAT_TYPE_ORDINAL)

    unmapped = df["flat_type_ordinal"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped:,} unmapped flat_type rows — filling with median.")
        df["flat_type_ordinal"] = df["flat_type_ordinal"].fillna(
            df["flat_type_ordinal"].median()
        )

    df[TARGET_COL] = np.log1p(df["resale_price"])

    print(
        "  Features added: flat_age, month_sin, month_cos, "
        "is_mature_estate, flat_type_ordinal, log_price"
    )
    return df


# ---------------------------------------------------------------------------
# Remove rows with missing model inputs
# ---------------------------------------------------------------------------

def drop_missing_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=REQUIRED_MODEL_COLS).copy()
    dropped = before - len(df)

    print(
        f"\nMissing-value cleanup: dropped {dropped:,} rows with missing "
        f"required model fields. Remaining: {len(df):,}"
    )
    return df


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2020].copy()
    val = df[(df["year"] >= 2021) & (df["year"] <= 2022)].copy()
    test = df[df["year"] >= 2023].copy()

    total = len(df)
    print(f"\nTemporal split (total {total:,} rows):")
    print(f"  Train (≤ 2020) : {len(train):>8,}  ({len(train) / total * 100:.1f}%)")
    print(f"  Val  (2021–22) : {len(val):>8,}  ({len(val) / total * 100:.1f}%)")
    print(f"  Test (≥ 2023)  : {len(test):>8,}  ({len(test) / total * 100:.1f}%)")

    return train, val, test


# ---------------------------------------------------------------------------
# Target encoding
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
        encoders[col] = {
            "means": means,
            "global_mean": global_mean,
        }

        enc_col = f"{col}_enc"
        train[enc_col] = train[col].map(means).fillna(global_mean)
        val[enc_col] = val[col].map(means).fillna(global_mean)
        test[enc_col] = test[col].map(means).fillna(global_mean)

        print(f"  {col}: {len(means)} categories encoded")

    return train, val, test, encoders


# ---------------------------------------------------------------------------
# Scale numeric features
# ---------------------------------------------------------------------------

def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    print("\nScaling numeric features ...")

    scaler = StandardScaler()

    train[SCALE_COLS] = scaler.fit_transform(train[SCALE_COLS])
    val[SCALE_COLS] = scaler.transform(val[SCALE_COLS])
    test[SCALE_COLS] = scaler.transform(test[SCALE_COLS])

    print("  StandardScaler fitted on train, applied to val and test.")
    return train, val, test, scaler


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_encoders: dict,
    scaler: StandardScaler,
) -> str:
    run_dir = os.path.join(
        OUTPUT_DIR,
        datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nSaving artefacts to '{run_dir}/' ...")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        X = split[FEATURE_COLS]
        y = split[[TARGET_COL, "resale_price"]]

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

    manifest = {
        "run_at": datetime.now(UTC).isoformat(),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "feature_cols": FEATURE_COLS,
        "scale_cols": SCALE_COLS,
        "quantiles": QUANTILES,
    }

    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  run_manifest.json")

    metrics_stub = {
        "q10_test_mae": None,
        "q50_test_mae": None,
        "q90_test_mae": None,
        "note": "Populate with actual model metrics after training.",
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_stub, f, indent=2)
    print("  metrics.json (stub)")

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

    null_count = train[FEATURE_COLS].isnull().sum().sum()
    print(f"\n  Null values: {null_count}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("HDB Resale — Feature Engineering")
    print("=" * 60)

    df = load_data()
    df = remove_outliers(df)
    df = engineer_features(df)
    df = drop_missing_model_rows(df)

    train, val, test = temporal_split(df)
    train, val, test, target_encoders = target_encode(train, val, test)
    train, val, test, scaler = scale_features(train, val, test)

    print_summary(train)
    run_dir = save_artefacts(train, val, test, target_encoders, scaler)

    print(f"\nDone. ML-ready data saved to: {run_dir}/")


if __name__ == "__main__":
    main()