"""
data_pipeline.py
================
Full pipeline for HDB Resale Flat Prices:
  1. Fetch all 5 datasets from data.gov.sg API (1990 – present)
  2. Combine into one unified DataFrame
  3. Clean using Singapore HDB domain logic
  4. Save to SQLite with consistent data types

Datasets (chronological):
  1990-1999  : d_ebc5ab87086db484f88045b47411ebc5
  2000-2012  : d_43f493c6c50d54243cc1eab0df142d6a
  2012-2014  : d_2d5ff9ea31397b66239f245f57751537
  2015-2016  : d_ea9ed51da2787afaf8e51f827c304208
  2017-now   : d_8b84c4ee58e3cfc0ece0d773c8ca6abc

Run:
    python data_pipeline.py
"""

import io
import re
import sqlite3
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_IDS = [
    "d_ebc5ab87086db484f88045b47411ebc5",  # 1990–1999
    "d_43f493c6c50d54243cc1eab0df142d6a",  # 2000–Feb 2012
    "d_2d5ff9ea31397b66239f245f57751537",  # Mar 2012–Dec 2014
    "d_ea9ed51da2787afaf8e51f827c304208",  # Jan 2015–Dec 2016
    "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",  # Jan 2017–present
]

OPEN_API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
DB_PATH = "hdb_resale.db"
TABLE_NAME = "resale_prices"

# Columns kept from raw API data (in order)
RAW_COLUMNS = [
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
]

# Explicit SQLite dtype mapping — all columns are NOT NULL for data integrity
SQLITE_DTYPES = {
    "month":                  "TEXT    NOT NULL",
    "year":                   "INTEGER NOT NULL",
    "month_num":              "INTEGER NOT NULL",
    "town":                   "TEXT    NOT NULL",
    "flat_type":              "TEXT    NOT NULL",
    "block":                  "TEXT    NOT NULL",
    "street_name":            "TEXT    NOT NULL",
    "storey_range":           "TEXT    NOT NULL",
    "storey_midpoint":        "REAL    NOT NULL",
    "floor_area_sqm":         "REAL    NOT NULL",
    "flat_model":             "TEXT    NOT NULL",
    "lease_commence_date":    "INTEGER NOT NULL",
    "remaining_lease":        "REAL    NOT NULL",  # years (e.g. 61.33)
    "remaining_lease_months": "REAL    NOT NULL",
    "resale_price":           "REAL    NOT NULL",
}

# Canonical flat_type values
FLAT_TYPE_ALIASES = {
    "1-room":          "1 Room",
    "2-room":          "2 Room",
    "3-room":          "3 Room",
    "4-room":          "4 Room",
    "5-room":          "5 Room",
    "multi generation":  "Multi-Generation",
    "multi-generation":  "Multi-Generation",
    "multigeneration":   "Multi-Generation",
}

# Known flat_model name variants → canonical form
FLAT_MODEL_ALIASES = {
    "New Gen":               "New Generation",
    "Improved-Maisonette":   "Improved Maisonette",
    "Premium Apartment Loft": "Premium Apartment Loft",
    "Dbss":                  "DBSS",
    "2-Room":                "2-Room",
    "Type S1":               "Type S1",
    "Type S2":               "Type S2",
}

# HDB floor-area sanity bounds (sqm) — used for logging only, rows are kept
FLOOR_AREA_MIN = 28.0   # smaller than any known 1-room unit
FLOOR_AREA_MAX = 320.0  # larger than any known multi-gen unit


# ---------------------------------------------------------------------------
# Step 1: Fetch from API
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, timeout: int = 60, max_retries: int = 5) -> requests.Response:
    """HTTP GET with exponential backoff on 429 Too Many Requests."""
    for attempt in range(max_retries):
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 s
            print(f"    Rate limited (429). Waiting {wait}s ... (retry {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    # Final attempt — let it raise naturally
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def fetch_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Downloads one dataset from data.gov.sg v1 API.
    Initiates an async export, polls until ready, then streams the CSV.
    """
    print(f"  Fetching: {dataset_id}")
    init_resp = _get_with_retry(f"{OPEN_API_BASE}/{dataset_id}/initiate-download")
    download_url = init_resp.json().get("data", {}).get("url")

    if not download_url:
        print("    Export not ready — polling ...")
        for attempt in range(30):
            time.sleep(3)
            poll_resp = _get_with_retry(f"{OPEN_API_BASE}/{dataset_id}/poll-download")
            poll_data = poll_resp.json().get("data", {})
            if poll_data.get("status") == "DOWNLOAD_SUCCESS":
                download_url = poll_data["url"]
                break
            print(f"    Poll attempt {attempt + 1}/30 ...")
        else:
            raise TimeoutError(f"Export for {dataset_id} did not complete.")

    csv_resp = requests.get(download_url, timeout=300)
    csv_resp.raise_for_status()
    df = pd.read_csv(io.StringIO(csv_resp.text))
    print(f"    Rows: {len(df):,}  Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Standardize column names across datasets
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises column names to snake_case and aligns all datasets to the
    same schema.  The 3 older datasets (1990–2014) lack `remaining_lease`;
    it is added as null here and reconstructed during cleaning.
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Column name variants across the 5 datasets
    df = df.rename(columns={
        "floor_area_(sqm)":       "floor_area_sqm",
        "lease_commencement_date": "lease_commence_date",
    })

    if "remaining_lease" not in df.columns:
        df["remaining_lease"] = None

    # Reorder to canonical column list, keeping only known columns
    df = df[[c for c in RAW_COLUMNS if c in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Step 3: Combine
# ---------------------------------------------------------------------------

def combine_datasets(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(dataframes, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows × {combined.shape[1]} columns")
    return combined


# ---------------------------------------------------------------------------
# Step 4: Clean
# ---------------------------------------------------------------------------

def _parse_lease_str(val) -> float:
    """
    Parse '61 years 04 months' → 736.0 (total months).
    Returns NaN if the string is absent, 'none', or unparseable.
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().lower()
    if s in ("", "none", "nan"):
        return float("nan")
    yr_match  = re.search(r"(\d+)\s*year",  s)
    mo_match  = re.search(r"(\d+)\s*month", s)
    years  = int(yr_match.group(1))  if yr_match  else 0
    months = int(mo_match.group(1))  if mo_match  else 0
    if years == 0 and months == 0:
        return float("nan")
    return float(years * 12 + months)


def _derive_lease_months(sale_year, lease_start, sale_month) -> float:
    """
    Compute remaining lease months from first principles.
    All HDB flats have a 99-year lease from lease_commence_date.
        remaining = 99*12 - months_elapsed_since_lease_start
    """
    if pd.isna(sale_year) or pd.isna(lease_start):
        return float("nan")
    months_elapsed = (int(sale_year) - int(lease_start)) * 12 + int(sale_month)
    return float(max(99 * 12 - months_elapsed, 0))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Cleaning data ...")
    original_len = len(df)

    # ------------------------------------------------------------------
    # month → parse to datetime, derive year and month_num, store as text
    # ------------------------------------------------------------------
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    df["year"]      = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["month"]     = df["month"].dt.strftime("%Y-%m")  # back to canonical string

    # ------------------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------------------
    df["floor_area_sqm"]      = pd.to_numeric(df["floor_area_sqm"],      errors="coerce")
    df["resale_price"]        = pd.to_numeric(df["resale_price"],         errors="coerce")
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"],  errors="coerce")

    # Cast year/month_num/lease_commence_date to nullable int so we don't
    # lose non-null rows to float NaN when some rows are null
    for col in ["year", "month_num", "lease_commence_date"]:
        df[col] = df[col].astype("Int64")

    # ------------------------------------------------------------------
    # town — uppercase, strip
    # ------------------------------------------------------------------
    df["town"] = df["town"].astype(str).str.strip().str.upper()

    # ------------------------------------------------------------------
    # flat_type — title-case, normalise known aliases
    # ------------------------------------------------------------------
    df["flat_type"] = df["flat_type"].astype(str).str.strip().str.lower()
    df["flat_type"] = df["flat_type"].replace(FLAT_TYPE_ALIASES)
    df["flat_type"] = df["flat_type"].str.title()

    # ------------------------------------------------------------------
    # block / street_name
    # ------------------------------------------------------------------
    df["block"]       = df["block"].astype(str).str.strip().str.upper()
    df["street_name"] = df["street_name"].astype(str).str.strip().str.title()

    # ------------------------------------------------------------------
    # storey_range — kept as text; derive storey_midpoint
    # ------------------------------------------------------------------
    df["storey_range"] = df["storey_range"].astype(str).str.strip().str.upper()

    def _storey_midpoint(val: str) -> float:
        m = re.search(r"(\d+)\s+TO\s+(\d+)", str(val), re.IGNORECASE)
        if m:
            return (int(m.group(1)) + int(m.group(2))) / 2.0
        return float("nan")

    df["storey_midpoint"] = df["storey_range"].apply(_storey_midpoint)

    # ------------------------------------------------------------------
    # flat_model — title-case, normalise known aliases
    # ------------------------------------------------------------------
    df["flat_model"] = df["flat_model"].astype(str).str.strip().str.title()
    df["flat_model"] = df["flat_model"].replace(FLAT_MODEL_ALIASES)

    # ------------------------------------------------------------------
    # floor_area_sqm — validate HDB-plausible range (log only, keep row)
    # ------------------------------------------------------------------
    out_of_range = (
        (df["floor_area_sqm"] < FLOOR_AREA_MIN) |
        (df["floor_area_sqm"] > FLOOR_AREA_MAX)
    )
    if out_of_range.sum() > 0:
        print(f"    WARNING: {out_of_range.sum():,} rows have floor_area_sqm "
              f"outside [{FLOOR_AREA_MIN}, {FLOOR_AREA_MAX}] sqm — kept as-is.")

    # ------------------------------------------------------------------
    # lease_commence_date — validate plausible range (1960–2025)
    # ------------------------------------------------------------------
    bad_lease = df["lease_commence_date"].notna() & (
        (df["lease_commence_date"] < 1960) | (df["lease_commence_date"] > 2025)
    )
    if bad_lease.sum() > 0:
        print(f"    WARNING: {bad_lease.sum():,} rows have lease_commence_date "
              f"outside [1960, 2025] — kept as-is.")

    # ------------------------------------------------------------------
    # remaining_lease + remaining_lease_months
    #
    # Strategy:
    #   1. Parse from string (datasets 2015+)
    #   2. Derive from lease_commence_date for missing values (datasets 1990–2014)
    #   3. Reconstruct the string for all rows that lacked it
    # ------------------------------------------------------------------

    # Step 1: parse numeric value from the raw string
    df["remaining_lease_months"] = df["remaining_lease"].apply(_parse_lease_str)

    # Step 2: fill missing numeric values by calculation
    missing_mask = df["remaining_lease_months"].isna()
    if missing_mask.sum() > 0:
        df.loc[missing_mask, "remaining_lease_months"] = df[missing_mask].apply(
            lambda r: _derive_lease_months(r["year"], r["lease_commence_date"], r["month_num"]),
            axis=1,
        )
        filled = missing_mask.sum() - df["remaining_lease_months"].isna().sum()
        print(f"    remaining_lease_months: derived {filled:,} missing values from lease_commence_date.")

    # Step 3: remaining_lease in years (REAL) — simply divide months by 12
    df["remaining_lease"] = (df["remaining_lease_months"] / 12).round(2)

    still_null = df["remaining_lease_months"].isna().sum()
    if still_null > 0:
        print(f"    WARNING: {still_null:,} rows still have null remaining_lease_months "
              f"(missing both string and lease_commence_date).")

    # ------------------------------------------------------------------
    # Drop rows with no resale price — these cannot be used for any purpose
    # ------------------------------------------------------------------
    df = df.dropna(subset=["resale_price"])
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"    Dropped {dropped:,} rows with null resale_price.")

    # ------------------------------------------------------------------
    # Final column order (original columns + derived columns)
    # ------------------------------------------------------------------
    final_cols = [
        "month", "year", "month_num",
        "town", "flat_type", "block", "street_name",
        "storey_range", "storey_midpoint",
        "floor_area_sqm", "flat_model",
        "lease_commence_date", "remaining_lease", "remaining_lease_months",
        "resale_price",
    ]
    df = df[final_cols]

    # Cast integer columns cleanly (convert Int64 → plain int where possible)
    for col in ["year", "month_num", "lease_commence_date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    print(f"  Clean complete: {len(df):,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 5: Enforce no nulls before saving
# ---------------------------------------------------------------------------

def _enforce_not_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees every column is non-null before writing to SQLite.
    - storey_midpoint: filled with per-flat_type median (rare parse failures)
    - All other columns must already be complete; raises if not.
    """
    # storey_midpoint can be NaN when storey_range is malformed
    null_storey = df["storey_midpoint"].isna()
    if null_storey.sum() > 0:
        median_by_type = df.groupby("flat_type")["storey_midpoint"].transform("median")
        df.loc[null_storey, "storey_midpoint"] = median_by_type[null_storey]
        print(f"    storey_midpoint: filled {null_storey.sum():,} nulls with per-flat_type median.")

    # Any remaining nulls in any column are a pipeline error — fail loudly
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if not remaining_nulls.empty:
        raise ValueError(
            f"Columns still have nulls before save:\n{remaining_nulls.to_string()}\n"
            "Fix the cleaning logic above."
        )
    return df


# ---------------------------------------------------------------------------
# Step 6: Save to SQLite
# ---------------------------------------------------------------------------

def save_to_sqlite(df: pd.DataFrame) -> None:
    print(f"\n  Saving to {DB_PATH} → table '{TABLE_NAME}' ...")
    conn = sqlite3.connect(DB_PATH)

    # Build CREATE TABLE with explicit column types for consistency
    col_defs = ", ".join(
        f'"{col}" {SQLITE_DTYPES.get(col, "TEXT")}' for col in df.columns
    )
    conn.execute(f'DROP TABLE IF EXISTS "{TABLE_NAME}"')
    conn.execute(f'CREATE TABLE "{TABLE_NAME}" ({col_defs})')
    conn.commit()

    # Write data
    df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

    # Indexes for fast querying
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_month ON "{TABLE_NAME}"(month)')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_town  ON "{TABLE_NAME}"(town)')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_flat  ON "{TABLE_NAME}"(flat_type)')
    conn.commit()
    conn.close()
    print(f"  Saved {len(df):,} rows.")


# ---------------------------------------------------------------------------
# Step 6: Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total rows        : {len(df):,}")
    print(f"  Date range        : {df['month'].min()} → {df['month'].max()}")
    print(f"  Towns             : {df['town'].nunique()} unique")
    print(f"  Flat types        : {sorted(df['flat_type'].dropna().unique())}")
    print(f"\n  Missing values per column:")
    missing = df.isnull().sum()
    for col, n in missing.items():
        flag = " ✓" if n == 0 else f" ← {n:,} missing"
        print(f"    {col:<28}{flag}")
    print(f"\n  Data types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<28}{dtype}")
    print(f"\n  Price range       : SGD {df['resale_price'].min():,.0f} – {df['resale_price'].max():,.0f}")
    print(f"  Remaining lease   : {df['remaining_lease'].min():.2f} – "
          f"{df['remaining_lease'].max():.2f} years")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("HDB Resale — Data Pipeline")
    print("=" * 60)

    # Fetch all datasets with a brief pause between requests
    raw_frames = []
    for i, dataset_id in enumerate(DATASET_IDS):
        df = fetch_dataset(dataset_id)
        df = standardize_columns(df)
        raw_frames.append(df)
        if i < len(DATASET_IDS) - 1:
            time.sleep(3)

    combined = combine_datasets(raw_frames)
    cleaned  = clean_data(combined)
    cleaned  = _enforce_not_null(cleaned)

    save_to_sqlite(cleaned)
    print_summary(cleaned)

    print(f"\nDone. Database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()