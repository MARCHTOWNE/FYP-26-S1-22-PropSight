"""
data_pipeline.py
================
Single responsibility: read raw CSVs from /raw, consolidate, clean,
validate, and write the authoritative dataset to hdb_resale.db.

Design decisions:
  - All API fetching has been moved to api_fetcher.py.
  - full_address is derived here and stored in the DB for use by geocoding.py.
  - Nullable columns (latitude, longitude, dist_mrt, dist_cbd,
    dist_primary_school, dist_major_mall) are created in this step so
    downstream scripts can UPDATE them without altering the table schema.
  - geocode_cache and upload_audit tables are never dropped on re-runs.
  - district_summary is dropped and rebuilt each run for accuracy.

Execution order context:
  Step 2 of 6. Reads: raw/*.csv. Writes: hdb_resale.db (resale_prices).
  Previous step: api_fetcher.py. Next step: geocoding.py.

Run:
    python data_pipeline.py
"""

import datetime
import os
import re
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR    = "raw"
DB_PATH    = "hdb_resale.db"
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

# Explicit SQLite dtype mapping
# Non-nullable columns enforce data integrity; nullable columns are filled
# by downstream scripts (geocoding.py, proximity_features.py).
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
    "full_address":           "TEXT",              # nullable — derived from block + street_name
    "latitude":               "REAL",              # nullable — filled by geocoding.py
    "longitude":              "REAL",              # nullable — filled by geocoding.py
    "dist_mrt":               "REAL",              # nullable — filled by proximity_features.py
    "dist_cbd":               "REAL",              # nullable — filled by proximity_features.py
    "dist_primary_school":    "REAL",              # nullable — filled by proximity_features.py
    "dist_major_mall":        "REAL",              # nullable — filled by proximity_features.py
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

# Columns intentionally left nullable (filled by external processes)
NULLABLE_COLS: set[str] = {
    "full_address",
    "latitude", "longitude",
    "dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall",
}

# HDB floor-area sanity bounds (sqm) — used for logging only, rows are kept
FLOOR_AREA_MIN = 28.0   # smaller than any known 1-room unit
FLOOR_AREA_MAX = 320.0  # larger than any known multi-gen unit

# Key columns used to detect duplicate records during admin ingestion
_ADMIN_DEDUP_KEY = ["month", "block", "street_name", "flat_type", "storey_range"]

# Validation thresholds for validate_data()
PRICE_PER_SQM_MIN = 1_000    # SGD — below this is almost certainly corrupt data
PRICE_PER_SQM_MAX = 20_000   # SGD — above this is almost certainly corrupt data

FLAT_TYPE_AREA_BOUNDS = {
    "1 Room":           (28,  45),
    "2 Room":           (45,  60),
    "3 Room":           (60,  90),
    "4 Room":           (90,  115),
    "5 Room":           (110, 150),
    "Executive":        (130, 200),
    "Multi-Generation": (155, 320),
}

VALID_TOWNS = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN",
}

LEASE_MISMATCH_TOLERANCE = 2    # years — accounts for rounding in source data
HDB_FIRST_YEAR           = 1960  # no HDB flats existed before this year


# ---------------------------------------------------------------------------
# Step 1: Load raw CSVs from /raw
# ---------------------------------------------------------------------------

def load_raw_csvs(raw_dir: str = RAW_DIR) -> list[pd.DataFrame]:
    """
    Scan RAW_DIR for all .csv files, read each into a DataFrame, and apply
    standardize_columns() to normalise schema across the 5 source datasets.

    Parameters:
        raw_dir: Path to the directory containing raw CSVs from api_fetcher.py.

    Returns:
        List of DataFrames, one per CSV file found.

    Raises:
        FileNotFoundError: If RAW_DIR is missing or contains no .csv files.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"RAW_DIR '{raw_dir}' does not exist. "
            "Run api_fetcher.py first to download the raw data."
        )

    csv_files = sorted(
        f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"RAW_DIR '{raw_dir}' contains no .csv files. "
            "Run api_fetcher.py first to download the raw data."
        )

    frames: list[pd.DataFrame] = []
    for filename in csv_files:
        path = os.path.join(raw_dir, filename)
        df = pd.read_csv(path)
        df = standardize_columns(df)
        print(f"  Loaded: {filename}  ({len(df):,} rows)")
        frames.append(df)

    return frames


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
    """
    Concatenate all raw DataFrames and remove cross-dataset duplicates.

    Duplicate detection uses the minimal set of transaction-identifying
    fields; rows that are identical on all seven are considered the same
    transaction and only the first occurrence is kept.

    Parameters:
        dataframes: List of standardized DataFrames from load_raw_csvs().

    Returns:
        Single combined DataFrame with duplicates removed.
    """
    combined = pd.concat(dataframes, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows × {combined.shape[1]} columns")

    # Drop rows that are identical across all transaction-identifying fields
    _DEDUP_COLS = [
        "month", "block", "street_name", "flat_type",
        "storey_range", "floor_area_sqm", "resale_price",
    ]
    before = len(combined)
    combined = combined.drop_duplicates(subset=_DEDUP_COLS)
    removed = before - len(combined)
    print(f"  Duplicates removed: {removed:,}  Rows remaining: {len(combined):,}")
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
    """
    Apply domain-specific cleaning rules to the combined raw DataFrame.

    Derives year, month_num, storey_midpoint, remaining_lease_months,
    full_address, and initialises nullable downstream columns. Drops rows
    with a null resale_price only — all other nulls are handled or logged.

    Parameters:
        df: Combined raw DataFrame from combine_datasets() or standardize_columns().

    Returns:
        Cleaned DataFrame with the canonical column set.
    """
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
    # full_address — derived for OneMap geocoding lookups
    # ------------------------------------------------------------------
    df["full_address"] = (
        df["block"].str.strip() + " " + df["street_name"].str.strip() + " SINGAPORE"
    )

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

    # Step 3: cap at the maximum possible HDB lease (99 years = 1 188 months),
    # then convert to years. Values above 1188 indicate a bad source string or
    # a lease_commence_date before the sale month — both are data errors.
    _MAX_LEASE_MONTHS = 99 * 12  # 1188
    over_cap = (df["remaining_lease_months"] > _MAX_LEASE_MONTHS).sum()
    if over_cap > 0:
        print(f"    WARNING: {over_cap:,} rows had remaining_lease_months > 1188 — capped at 99 years.")
    df["remaining_lease_months"] = df["remaining_lease_months"].clip(upper=_MAX_LEASE_MONTHS)
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
    # Initialise nullable columns filled by downstream scripts
    # ------------------------------------------------------------------
    df["latitude"]   = None   # filled by geocoding.py
    df["longitude"]  = None   # filled by geocoding.py
    df["dist_mrt"]            = None  # filled by proximity_features.py
    df["dist_cbd"]            = None  # filled by proximity_features.py
    df["dist_primary_school"] = None  # filled by proximity_features.py
    df["dist_major_mall"]     = None  # filled by proximity_features.py

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
        "full_address",
        "latitude", "longitude",
        "dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall",
    ]
    df = df[final_cols]

    # Cast integer columns cleanly (convert Int64 → plain int where possible)
    for col in ["year", "month_num", "lease_commence_date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    print(f"  Clean complete: {len(df):,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 4b: Domain validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific hard and soft validation rules to the cleaned DataFrame.

    Hard rules (H1–H7): rows that are physically or logically impossible are
    dropped. Each rule is applied independently and its drop count is logged.

    Soft rules (S1–S5): rows that are suspicious but possibly valid are flagged
    with a WARNING count in the summary. No rows are removed by soft rules.

    Parameters:
        df: Cleaned DataFrame from clean_data().

    Returns:
        DataFrame with hard-rule violations removed. Soft-rule rows are kept.
    """
    print("\n  Validating data ...")
    before = len(df)

    def _bool_mask(series: pd.Series) -> pd.Series:
        """Return a plain bool Series with nulls filled as False."""
        return series.fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # Pre-compute helper series used by multiple rules
    # ------------------------------------------------------------------
    _year = pd.to_numeric(df["year"], errors="coerce").astype("float64")
    _lcd = pd.to_numeric(df["lease_commence_date"], errors="coerce").astype("float64")
    _remaining_lease = pd.to_numeric(df["remaining_lease"], errors="coerce").astype("float64")
    _floor_area = pd.to_numeric(df["floor_area_sqm"], errors="coerce").astype("float64")
    _resale_price = pd.to_numeric(df["resale_price"], errors="coerce").astype("float64")

    # Parse storey_range "XX TO YY" into lower and upper integers for H7
    _storey = df["storey_range"].str.extract(r"(\d+)\s+TO\s+(\d+)", expand=True)
    storey_lo = pd.to_numeric(_storey[0], errors="coerce").astype("float64")
    storey_hi = pd.to_numeric(_storey[1], errors="coerce").astype("float64")

    # ------------------------------------------------------------------
    # HARD RULES — build boolean drop mask
    # ------------------------------------------------------------------
    h1_mask = _bool_mask(_remaining_lease > 99)
    h1_count = int(h1_mask.sum())

    h2_mask = _bool_mask(_remaining_lease < 0)
    h2_count = int(h2_mask.sum())

    h3_mask = _bool_mask(_lcd.notna() & _year.notna() & (_lcd > _year))
    h3_count = int(h3_mask.sum())

    h4_mask = _bool_mask(_lcd.notna() & (_lcd < HDB_FIRST_YEAR))
    h4_count = int(h4_mask.sum())

    h5_mask = _bool_mask(_resale_price <= 0)
    h5_count = int(h5_mask.sum())

    h6_mask = _bool_mask(_floor_area <= 0)
    h6_count = int(h6_mask.sum())

    h7_mask = _bool_mask(storey_hi.notna() & storey_lo.notna() & (storey_hi < storey_lo))
    h7_count = int(h7_mask.sum())

    drop_mask = _bool_mask(h1_mask | h2_mask | h3_mask | h4_mask | h5_mask | h6_mask | h7_mask)
    df = df.loc[~drop_mask].copy()
    total_hard_dropped = before - len(df)

    # Recompute helpers after filtering so indices align exactly with df
    _year = pd.to_numeric(df["year"], errors="coerce").astype("float64")
    _lcd = pd.to_numeric(df["lease_commence_date"], errors="coerce").astype("float64")
    _remaining_lease = pd.to_numeric(df["remaining_lease"], errors="coerce").astype("float64")
    _floor_area = pd.to_numeric(df["floor_area_sqm"], errors="coerce").astype("float64")
    _resale_price = pd.to_numeric(df["resale_price"], errors="coerce").astype("float64")

    # ------------------------------------------------------------------
    # SOFT RULES — log WARNING counts, no rows removed
    # ------------------------------------------------------------------
    _expected_lease = (99 - (_year - _lcd)).astype("float64")
    valid_lease_mask = _bool_mask(_year.notna() & _lcd.notna() & _remaining_lease.notna())
    s1_mask = _bool_mask(
        valid_lease_mask &
        ((_remaining_lease - _expected_lease).abs() > LEASE_MISMATCH_TOLERANCE)
    )
    s1_count = int(s1_mask.sum())

    _price_per_sqm = (_resale_price / _floor_area).astype("float64")
    s2_mask = _bool_mask(
        _price_per_sqm.notna() &
        ((_price_per_sqm < PRICE_PER_SQM_MIN) | (_price_per_sqm > PRICE_PER_SQM_MAX))
    )
    s2_count = int(s2_mask.sum())

    s3_mask = pd.Series(False, index=df.index, dtype=bool)
    for flat_type, (lo, hi) in FLAT_TYPE_AREA_BOUNDS.items():
        type_rows = _bool_mask(df["flat_type"] == flat_type)
        out_of_bounds = _bool_mask(type_rows & ((_floor_area < lo) | (_floor_area > hi)))
        s3_mask |= out_of_bounds
    s3_count = int(s3_mask.sum())

    s4_mask = _bool_mask(~df["town"].isin(VALID_TOWNS))
    s4_count = int(s4_mask.sum())

    s5_mask = _bool_mask(_year.notna() & (_year < HDB_FIRST_YEAR))
    s5_count = int(s5_mask.sum())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep = "  " + "─" * 50
    print(sep)
    print("  Validation Summary")
    print(sep)
    print("  HARD drops:")
    print(f"    H1 remaining_lease > 99 yrs       : {h1_count:>8,} rows dropped")
    print(f"    H2 remaining_lease < 0            : {h2_count:>8,} rows dropped")
    print(f"    H3 lease_commence > sale_year     : {h3_count:>8,} rows dropped")
    print(f"    H4 lease_commence < {HDB_FIRST_YEAR}          : {h4_count:>8,} rows dropped")
    print(f"    H5 resale_price <= 0              : {h5_count:>8,} rows dropped")
    print(f"    H6 floor_area_sqm <= 0            : {h6_count:>8,} rows dropped")
    print(f"    H7 storey range corrupt           : {h7_count:>8,} rows dropped")
    print(f"    Total hard dropped                : {total_hard_dropped:>8,} rows")
    print("  SOFT flags (kept):")
    print(f"    S1 remaining_lease mismatch > {LEASE_MISMATCH_TOLERANCE}yr : {s1_count:>8,} rows flagged")
    print(f"    S2 price/sqm outside {PRICE_PER_SQM_MIN // 1000}k–{PRICE_PER_SQM_MAX // 1000}k      : {s2_count:>8,} rows flagged")
    print(f"    S3 floor area outside type bounds : {s3_count:>8,} rows flagged")
    print(f"    S4 unrecognised town              : {s4_count:>8,} rows flagged")
    print(f"    S5 sale year before {HDB_FIRST_YEAR}          : {s5_count:>8,} rows flagged")
    print(f"\n  Rows before validation : {before:>9,}")
    print(f"  Rows after  validation : {len(df):>9,}")
    print(sep)

    return df


# ---------------------------------------------------------------------------
# Step 5: Enforce no nulls before saving
# ---------------------------------------------------------------------------

def _enforce_not_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees every non-nullable column is non-null before writing to SQLite.
    - storey_midpoint: filled with per-flat_type median (rare parse failures)
    - All other non-nullable columns must already be complete; raises if not.
    """
    # storey_midpoint can be NaN when storey_range is malformed
    null_storey = df["storey_midpoint"].isna()
    if null_storey.sum() > 0:
        median_by_type = df.groupby("flat_type")["storey_midpoint"].transform("median")
        df.loc[null_storey, "storey_midpoint"] = median_by_type[null_storey]
        print(f"    storey_midpoint: filled {null_storey.sum():,} nulls with per-flat_type median.")

    # Any remaining nulls in non-nullable columns are a pipeline error — fail loudly
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    remaining_nulls = remaining_nulls[~remaining_nulls.index.isin(NULLABLE_COLS)]
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
    """
    Write the cleaned DataFrame to hdb_resale.db and maintain all
    auxiliary tables required by the pipeline.

    Tables created/updated:
      - resale_prices     : dropped and recreated each run
      - geocode_cache     : created if absent; never dropped
      - district_summary  : dropped and recreated each run
      - pipeline_meta     : created if absent; rows upserted each run
      - upload_audit      : created if absent; never dropped

    Parameters:
        df: Cleaned DataFrame from clean_data() + _enforce_not_null().
    """
    print(f"\n  Saving to {DB_PATH} → table '{TABLE_NAME}' ...")
    conn = sqlite3.connect(DB_PATH)

    # Build CREATE TABLE with explicit column types for consistency
    col_defs = ", ".join(
        f'"{col}" {SQLITE_DTYPES.get(col, "TEXT")}' for col in df.columns
    )
    conn.execute(f'DROP TABLE IF EXISTS "{TABLE_NAME}"')
    conn.execute(f'CREATE TABLE "{TABLE_NAME}" ({col_defs})')
    conn.commit()

    # geocode_cache — never dropped; survives pipeline re-runs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geocode_cache (
            full_address  TEXT PRIMARY KEY,
            latitude      REAL NOT NULL,
            longitude     REAL NOT NULL,
            fetched_at    TEXT NOT NULL
        )
    """)
    conn.commit()

    # upload_audit — permanent audit trail for FR8 admin uploads; never dropped
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_audit (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            uploaded_at   TEXT NOT NULL,
            uploaded_by   TEXT NOT NULL,
            filename      TEXT NOT NULL,
            rows_inserted INTEGER NOT NULL,
            status        TEXT NOT NULL
        )
    """)
    conn.commit()

    # Write resale_prices data
    df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

    # Indexes for fast querying
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_month ON "{TABLE_NAME}"(month)')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_town  ON "{TABLE_NAME}"(town)')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_flat  ON "{TABLE_NAME}"(flat_type)')
    conn.commit()
    print(f"  Saved {len(df):,} rows.")

    # district_summary — drop and recreate each run for accuracy
    conn.execute("DROP TABLE IF EXISTS district_summary")
    summary = (
        df.groupby(["town", "flat_type", "year"])
        .agg(
            median_price=("resale_price", "median"),
            avg_price=("resale_price", "mean"),
            transaction_count=("resale_price", "count"),
            avg_floor_area=("floor_area_sqm", "mean"),
            avg_remaining_lease=("remaining_lease", "mean"),
        )
        .reset_index()
    )
    summary.to_sql("district_summary", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ds_town_year ON district_summary(town, year)"
    )
    conn.commit()
    print(f"  district_summary: {len(summary):,} aggregated rows saved.")

    # pipeline_meta — persists across runs; upserted on each run
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pipeline_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    meta_entries = [
        ("last_fetched_month", str(df["month"].max())),
        ("last_run_at",        datetime.datetime.utcnow().isoformat()),
        ("total_rows",         str(len(df))),
    ]
    for key, value in meta_entries:
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
    conn.commit()
    conn.close()


def get_pipeline_meta(db_path: str) -> dict:
    """
    Reads all rows from the pipeline_meta table and returns them as a plain dict.

    Used by retrain_pipeline.py to decide whether new data exists without
    running the full pipeline.

    Parameters:
        db_path: Path to the SQLite database file.

    Returns:
        A dict mapping each key to its value string (e.g. {'total_rows': '892345'}).
        Returns an empty dict if the table does not yet exist (pipeline not run).
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT key, value FROM pipeline_meta").fetchall()
        return dict(rows)
    except sqlite3.OperationalError:
        # pipeline_meta table absent — pipeline has never been run
        return {}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 7: Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable pipeline summary to stdout."""
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
# Admin upload path (FR8)
# ---------------------------------------------------------------------------

def ingest_admin_csv(csv_path: str, db_path: str, uploaded_by: str) -> int:
    """
    Controlled entry point for FR8 (Admin Data Management).

    Reads a CSV uploaded via the admin panel, cleans it using the same logic
    as the main pipeline, and appends only genuinely new records to the DB.
    Rebuilds district_summary and pipeline_meta after each successful insert.
    Writes a row to upload_audit for every call (success or partial).

    Steps:
        a. Reads csv_path using standardize_columns().
        b. Runs clean_data() on it.
        c. Runs _enforce_not_null() on it.
        d. Appends only rows where (month, block, street_name, flat_type,
           storey_range) does not already exist in the DB.
        e. Updates district_summary and pipeline_meta.
        f. Writes a row to upload_audit.
        g. Returns number of new rows inserted.

    Parameters:
        csv_path:    Filesystem path to the CSV file to ingest.
        db_path:     Path to the SQLite database file to write to.
        uploaded_by: Identifier of the admin user performing the upload
                     (written to the upload_audit table for traceability).

    Returns:
        Number of new rows inserted into resale_prices (0 if none were new).

    Raises:
        Exception: Re-raises any I/O or database error after logging it.
    """
    filename = os.path.basename(csv_path)
    uploaded_at = datetime.datetime.utcnow().isoformat()

    # a. Read and standardize
    try:
        raw = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [ingest_admin_csv] ERROR reading '{csv_path}': {exc}")
        raise

    df = standardize_columns(raw)

    # b. Clean
    df = clean_data(df)

    # c. Enforce not-null (nullable cols are intentionally skipped)
    df = _enforce_not_null(df)

    # d. Check which rows already exist in the DB
    conn = sqlite3.connect(db_path)
    try:
        try:
            existing = pd.read_sql(
                f'SELECT {", ".join(_ADMIN_DEDUP_KEY)} FROM "{TABLE_NAME}"', conn
            )
            existing = existing.drop_duplicates(subset=_ADMIN_DEDUP_KEY)
        except Exception:
            # resale_prices table absent — all rows are new
            existing = pd.DataFrame(columns=_ADMIN_DEDUP_KEY)

        merged = df.merge(existing, on=_ADMIN_DEDUP_KEY, how="left", indicator=True)
        new_mask = (merged["_merge"] == "left_only").values
        new_rows = df[new_mask].copy()
    except Exception as exc:
        conn.close()
        print(f"  [ingest_admin_csv] ERROR checking existing rows: {exc}")
        raise

    inserted = len(new_rows)
    print(
        f"  [ingest_admin_csv] New rows to insert: {inserted:,}  "
        f"(skipped {len(df) - inserted:,} already-existing rows)"
    )

    status = "success"
    if inserted > 0:
        # Append only the new rows
        new_rows.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        # e. Rebuild district_summary from the full updated table
        all_df = pd.read_sql(f'SELECT * FROM "{TABLE_NAME}"', conn)
        conn.execute("DROP TABLE IF EXISTS district_summary")
        summary = (
            all_df.groupby(["town", "flat_type", "year"])
            .agg(
                median_price=("resale_price", "median"),
                avg_price=("resale_price", "mean"),
                transaction_count=("resale_price", "count"),
                avg_floor_area=("floor_area_sqm", "mean"),
                avg_remaining_lease=("remaining_lease", "mean"),
            )
            .reset_index()
        )
        summary.to_sql("district_summary", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ds_town_year ON district_summary(town, year)"
        )

        # Update pipeline_meta to reflect the new totals
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        meta_entries = [
            ("last_fetched_month", str(all_df["month"].max())),
            ("last_run_at",        datetime.datetime.utcnow().isoformat()),
            ("total_rows",         str(len(all_df))),
        ]
        for key, value in meta_entries:
            conn.execute(
                "INSERT OR REPLACE INTO pipeline_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    # f. Write upload_audit row (always — even when inserted == 0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS upload_audit ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  uploaded_at TEXT NOT NULL,"
        "  uploaded_by TEXT NOT NULL,"
        "  filename    TEXT NOT NULL,"
        "  rows_inserted INTEGER NOT NULL,"
        "  status      TEXT NOT NULL"
        ")"
    )
    conn.execute(
        "INSERT INTO upload_audit (uploaded_at, uploaded_by, filename, rows_inserted, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (uploaded_at, uploaded_by, filename, inserted, status),
    )
    conn.commit()
    conn.close()

    print(
        f"  [ingest_admin_csv] Inserted {inserted:,} rows. "
        f"district_summary and pipeline_meta updated. "
        f"upload_audit row written (uploaded_by={uploaded_by!r})."
    )
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load raw CSVs from /raw, consolidate, clean, validate, and write to SQLite.
    Requires api_fetcher.py to have been run first.
    """
    print("=" * 60)
    print("HDB Resale — Data Pipeline")
    print("=" * 60)
    print(f"\n  Reading raw CSVs from: {RAW_DIR}/")

    raw_frames = load_raw_csvs()
    combined = combine_datasets(raw_frames)
    cleaned  = clean_data(combined)
    cleaned  = validate_data(cleaned)
    cleaned  = _enforce_not_null(cleaned)

    save_to_sqlite(cleaned)
    print_summary(cleaned)

    print(f"\nDone. Database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
