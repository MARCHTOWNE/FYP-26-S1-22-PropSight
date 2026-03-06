
# HDB Resale Price Analytics Platform

A data pipeline for Singapore HDB resale price analytics and future price prediction.

This project builds a clean, enriched, analysis-ready HDB resale dataset from public transaction data. It is designed to support a web platform for historical trends, map-based analytics, and future price prediction features.

---

## Project Overview

The pipeline:

1. fetches HDB resale transaction data from data.gov.sg
2. consolidates and cleans the raw CSV files
3. stores the cleaned data in a SQLite database
4. geocodes HDB addresses using OneMap
5. computes proximity-based location features
6. prepares a clean feature-ready dataset for future machine learning use

The current project scope focuses on:

- HDB resale data only
- clean and reliable data preparation
- location enrichment
- feature engineering
- support for future Streamlit-based analytics and prediction tools

---

## Architecture

Each file has a single responsibility.

```text
api_fetcher.py          → Fetch raw CSVs from data.gov.sg API → /raw
data_pipeline.py        → Consolidate, clean, validate → hdb_resale.db
geocoding.py            → Resolve latitude/longitude using OneMap API
proximity_features.py   → Compute MRT/CBD/school/mall distances
feature_engineering.py  → Build clean feature-ready datasets
eda_visualisation.py    → Optional exploratory analysis
````

---

## Database Schema

Main database: `hdb_resale.db`

### Tables

| Table              | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `resale_prices`    | Main transaction table, one row per resale transaction   |
| `district_summary` | Aggregated summary table for analytics and visualisation |
| `geocode_cache`    | Cached OneMap geocoding results                          |
| `pipeline_meta`    | Stores metadata about pipeline runs                      |

---

## `resale_prices` Key Columns

| Column                   | Type    | Description                        |
| ------------------------ | ------- | ---------------------------------- |
| `month`                  | TEXT    | Transaction month                  |
| `year`                   | INTEGER | Transaction year                   |
| `month_num`              | INTEGER | Transaction month number           |
| `town`                   | TEXT    | HDB town                           |
| `flat_type`              | TEXT    | Flat type                          |
| `block`                  | TEXT    | Block number                       |
| `street_name`            | TEXT    | Street name                        |
| `storey_range`           | TEXT    | Original storey range              |
| `storey_midpoint`        | REAL    | Derived midpoint of storey range   |
| `floor_area_sqm`         | REAL    | Floor area in square metres        |
| `flat_model`             | TEXT    | HDB flat model                     |
| `lease_commence_date`    | INTEGER | Lease commencement year            |
| `remaining_lease`        | REAL    | Remaining lease in years           |
| `remaining_lease_months` | REAL    | Remaining lease in months          |
| `resale_price`           | REAL    | Resale price in SGD                |
| `full_address`           | TEXT    | Derived full address               |
| `latitude`               | REAL    | Geocoded latitude                  |
| `longitude`              | REAL    | Geocoded longitude                 |
| `dist_mrt`               | REAL    | Distance to nearest MRT/LRT        |
| `dist_cbd`               | REAL    | Distance to CBD anchor point       |
| `dist_school`            | REAL    | Distance to nearest primary school |
| `dist_mall`              | REAL    | Distance to nearest shopping mall  |

---

## Execution Order

### Full data pipeline run

```bash
python api_fetcher.py
python data_pipeline.py
python geocoding.py
python proximity_features.py
python feature_engineering.py
```

### Optional EDA

```bash
python eda_visualisation.py
```

---

## File Reference

### `api_fetcher.py`

Fetches HDB resale datasets from data.gov.sg and saves raw CSV files into `/raw`.

Main purpose:

* download raw source data only
* no cleaning or transformation

---

### `data_pipeline.py`

Reads all raw CSV files, consolidates them, cleans the data, validates records, and writes the results to `hdb_resale.db`.

Main responsibilities:

* standardise column types
* derive helper columns such as `storey_midpoint`, `full_address`, and `remaining_lease_months`
* apply hard and soft validation rules
* save cleaned data to SQLite
* build `district_summary`
* update pipeline metadata

---

### `geocoding.py`

Geocodes unique HDB addresses using the OneMap API and updates `latitude` and `longitude` in `resale_prices`.

Main responsibilities:

* use cache-first lookup through `geocode_cache`
* avoid repeated API calls
* batch-save geocoding results
* update all matching rows in `resale_prices`

---

### `proximity_features.py`

Computes proximity-based location features for geocoded transactions.

Generated features:

* `dist_mrt`
* `dist_cbd`
* `dist_school`
* `dist_mall`

These features are written back to `resale_prices`.

---

### `feature_engineering.py`
Transforms the enriched HDB resale database into clean, feature-ready train/val/test splits.

Split strategy:
- Train: ≤ 2020
- Val: 2021–2022
- Test: ≥ 2023

Key decisions:
- `log1p(resale_price)` as target to reduce right skew
- Ordinal encoding for `flat_type`
- Target encoding for `town` and `flat_model` (fit on train only to reduce leakage)
- `StandardScaler` for numeric features (fit on train only)
- IQR outlier removal by `flat_type`, with bounds saved to `outlier_bounds.json`
- Cyclical encoding for month using `month_sin` and `month_cos`
- Removal of rows with missing required fields so saved datasets contain no null values
- Saves processed artefacts to `model_assets/<YYYYMMDD_HHMMSS>/`

Outputs:
- `X_train.parquet`, `X_val.parquet`, `X_test.parquet`
- `y_train.parquet`, `y_val.parquet`, `y_test.parquet`
- `scaler.pkl`
- `target_encoders.pkl`
- `outlier_bounds.json`
- `feature_cols.txt`
- `run_manifest.json`
- `metrics.json` (stub)
---

### `eda_visualisation.py`

Optional exploratory analysis script for understanding distributions, trends, and feature relationships in the dataset.

---

## Data Validation Rules

Validation is handled in `validate_data()` in `data_pipeline.py`.

### Hard rules — dropped rows

| Rule | Condition                    |
| ---- | ---------------------------- |
| H1   | `remaining_lease > 99`       |
| H2   | `remaining_lease < 0`        |
| H3   | `lease_commence_date > year` |
| H4   | `lease_commence_date < 1960` |
| H5   | `resale_price <= 0`          |
| H6   | `floor_area_sqm <= 0`        |
| H7   | Invalid storey range bounds  |

### Soft rules — flagged but kept

| Rule | Condition                                       |
| ---- | ----------------------------------------------- |
| S1   | Remaining lease mismatch greater than tolerance |
| S2   | Price per sqm outside expected range            |
| S3   | Floor area outside expected flat-type bounds    |
| S4   | Unrecognised town                               |
| S5   | Sale year before 1960                           |

---

## Processed Assets Structure

Generated by `feature_engineering.py`:

```text
model_assets/
├── latest.txt
└── YYYYMMDD_HHMMSS/
    ├── X_train.parquet
    ├── X_val.parquet
    ├── X_test.parquet
    ├── y_train.parquet
    ├── y_val.parquet
    ├── y_test.parquet
    ├── scaler.pkl
    ├── target_encoders.pkl
    ├── outlier_bounds.json
    ├── feature_cols.txt
    ├── run_manifest.json
    └── metrics.json
```

---

## Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy requests pyarrow
```

---

## Viewing the Database

You can open `hdb_resale.db` in VS Code using the **SQLite Viewer** extension.

You can also inspect it using SQLite in the terminal:

```bash
sqlite3 hdb_resale.db
```

Useful commands:

```sql
.tables
SELECT COUNT(*) FROM resale_prices;
SELECT COUNT(*) FROM geocode_cache;
```

---

## Data Sources

| Source                        | Usage                           |
| ----------------------------- | ------------------------------- |
| data.gov.sg HDB resale prices | Main transaction dataset        |
| OneMap Singapore API          | Address geocoding               |
| MRT/LRT reference data        | Nearest station distance        |
| School reference data         | Nearest primary school distance |
| Mall reference data           | Nearest shopping mall distance  |

---

