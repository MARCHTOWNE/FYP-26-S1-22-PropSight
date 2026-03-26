import json
import os
import pickle
import time
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training_data_source import get_training_data_source_name, load_training_dataframe


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("MODEL_ASSETS_DIR", os.path.join(BASE_DIR, "model_assets"))

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
    "dist_primary_school",
    "dist_major_mall",
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
    "dist_primary_school",
    "dist_major_mall",
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
    "dist_primary_school",
    "dist_major_mall",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, str]:
    source = get_training_data_source_name()
    print(f"Loading training data from {source} ...", flush=True)
    t0 = time.time()
    df, data_source = load_training_dataframe()
    print(f"  Source query completed in {time.time() - t0:.1f}s.")

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
        "dist_primary_school",
        "dist_major_mall",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["town", "flat_type", "flat_model"]:
        # Preserve missing values so required-field cleanup can drop them later.
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    print(f"  Loaded {len(df):,} rows.")
    return df, data_source


# ---------------------------------------------------------------------------
# Outlier diagnostics / conservative cleanup
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    bounds: dict[str, dict[str, float]] = {}

    for flat_type, idx in df.groupby("flat_type").groups.items():
        prices = df.loc[idx, "resale_price"]
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bounds[str(flat_type)] = {
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2),
        }

    # Keep premium but valid transactions in training. With a log-price target,
    # a simple flat_type IQR fence is too aggressive and can remove legitimate
    # high-end sales that the model needs to learn from.
    keep_mask = df["resale_price"] > 0
    df = df[keep_mask].copy()

    dropped = before - len(df)
    print("\nOutlier handling: IQR bounds saved for diagnostics only.")
    print("  High-price transactions are retained to preserve premium-market signal.")
    print(
        f"  Invalid-price cleanup: dropped {dropped:,} rows "
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

def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
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

    # Use a direct log-price target so holdout rows do not depend on a
    # market index estimated from future transactions.
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


# Temporal split
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame, test_months: int = 3) -> dict[str, pd.DataFrame]:
    """
    Split data using a hybrid strategy:
    - Test set: most recent `test_months` months of data (rolling holdout)
    - Train/Val: remaining data split 90/10 via stratified random sampling
      on flat_type for representative coverage.
    """
    # Build a year-month column for recency-based test holdout
    df["_ym"] = df["year"] * 100 + df["month_num"]
    unique_ym = sorted(df["_ym"].dropna().unique())

    if len(unique_ym) <= test_months:
        cutoff_ym = unique_ym[0]
    else:
        cutoff_ym = unique_ym[-test_months]

    test = df[df["_ym"] >= cutoff_ym].copy()
    remaining = df[df["_ym"] < cutoff_ym].copy()

    # Stratified random split of remaining data into train (90%) and val (10%)
    train, val = train_test_split(
        remaining,
        test_size=0.10,
        random_state=42,
        stratify=remaining["flat_type"],
    )
    train = train.copy()
    val = val.copy()

    # Clean up temp column
    for split in (train, val, test):
        split.drop(columns=["_ym"], inplace=True)
    df.drop(columns=["_ym"], inplace=True)

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    total = len(df)
    test_start = f"{int(cutoff_ym // 100)}-{int(cutoff_ym % 100):02d}"
    print(f"\nStratified split with {test_months}-month rolling holdout (total {total:,} rows):")
    print(f"  Train (random 90%) : {len(train):>8,}  ({len(train) / total * 100:.1f}%)")
    print(f"  Val   (random 10%) : {len(val):>8,}  ({len(val) / total * 100:.1f}%)")
    print(f"  Test  (>= {test_start}): {len(test):>8,}  ({len(test) / total * 100:.1f}%)")

    return splits


def build_split_metadata(
    splits: dict[str, pd.DataFrame],
) -> dict[str, int | None]:
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    return {
        "train_rows": len(train),
        "train_min_year": int(train["year"].min()) if not train.empty else None,
        "train_max_year": int(train["year"].max()) if not train.empty else None,
        "val_rows": len(val),
        "val_min_year": int(val["year"].min()) if not val.empty else None,
        "val_max_year": int(val["year"].max()) if not val.empty else None,
        "test_rows": len(test),
        "test_min_year": int(test["year"].min()) if not test.empty else None,
        "test_max_year": int(test["year"].max()) if not test.empty else None,
        "split_strategy": "stratified_random_with_rolling_holdout",
    }


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def target_encode(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict]:
    print("\nTarget encoding town and flat_model ...")

    train = splits["train"]
    global_mean = train[TARGET_COL].mean()
    encoders = {}

    for col in ["town", "flat_model"]:
        means = train.groupby(col)[TARGET_COL].mean()
        encoders[col] = {
            "means": means,
            "global_mean": global_mean,
        }

        enc_col = f"{col}_enc"
        for split in splits.values():
            split[enc_col] = split[col].map(means).fillna(global_mean)

        print(f"  {col}: {len(means)} categories encoded")

    return splits, encoders


# ---------------------------------------------------------------------------
# Scale numeric features
# ---------------------------------------------------------------------------

def scale_features(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    print("\nScaling numeric features ...")

    scaler = StandardScaler()
    train = splits["train"]

    train[SCALE_COLS] = scaler.fit_transform(train[SCALE_COLS])
    for name, split in splits.items():
        if name == "train":
            continue
        split[SCALE_COLS] = scaler.transform(split[SCALE_COLS])

    applied_to = ", ".join(name for name in splits if name != "train")
    print(f"  StandardScaler fitted on train, applied to {applied_to}.")
    return splits, scaler


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    splits: dict[str, pd.DataFrame],
    target_encoders: dict,
    scaler: StandardScaler,
    split_metadata: dict[str, int | None],
    data_source: str,
) -> str:
    run_dir = os.path.join(
        OUTPUT_DIR,
        datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nSaving artefacts to '{run_dir}/' ...")

    for name in ("train", "val", "test", "future_holdout"):
        if name not in splits:
            continue
        split = splits[name]
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
        "train_rows": len(splits["train"]),
        "val_rows": len(splits["val"]),
        "test_rows": len(splits["test"]),
        "feature_cols": FEATURE_COLS,
        "scale_cols": SCALE_COLS,
        "quantiles": QUANTILES,
        "split_metadata": split_metadata,
        "data_source": data_source,
        "target_transform": "log1p_resale_price",
        "outlier_strategy": "diagnostic_iqr_bounds_keep_nonzero_prices",
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
        f.write(os.path.relpath(run_dir, BASE_DIR))
    print(f"  latest.txt → {os.path.relpath(run_dir, BASE_DIR)}")

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

    df, data_source = load_data()
    df = engineer_features(df)
    df = drop_missing_model_rows(df)

    splits = stratified_split(df, test_months=3)
    split_metadata = build_split_metadata(splits)
    splits["train"] = remove_outliers(splits["train"])
    splits, target_encoders = target_encode(splits)
    splits, scaler = scale_features(splits)

    print_summary(splits["train"])
    run_dir = save_artefacts(
        splits,
        target_encoders,
        scaler,
        split_metadata,
        data_source,
    )

    print(f"\nDone. ML-ready data saved to: {run_dir}/")


if __name__ == "__main__":
    main()
