# HDB Resale Price Analytics Platform

A full machine learning pipeline for Singapore HDB resale price prediction, market analytics, and AI-assisted insights. Built for a web platform with role-based access (Guest, General, Premium, Admin).

---

## Project Overview

This pipeline fetches HDB resale transaction data from data.gov.sg, cleans and consolidates it into a SQLite database, enriches each transaction with geocoded coordinates and proximity features, then trains quantile regression models to produce property price ranges — not single-point estimates.

The platform supports:
- **FR2** — Property price prediction with value ranges (Q10–Q90)
- **FR3/FR4** — Pin-level and district-level heatmaps
- **FR5** — Graphical market analysis and district comparisons
- **FR6** — SHAP-based AI explanations (Premium users)
- **FR7** — Role-based access control
- **FR8** — Admin data upload and audit trail

---

## Architecture

Each file has a single responsibility. No file does more than one job.

```
api_fetcher.py          → Fetch raw CSVs from data.gov.sg API → /raw
data_pipeline.py        → Consolidate, clean, validate → hdb_resale.db
geocoding.py            → Resolve lat/lng via OneMap API → resale_prices
proximity_features.py   → Compute MRT/CBD/school/mall distances → resale_prices
feature_engineering.py  → Build ML-ready splits → model_assets/<timestamp>/
ml_model.py             → Train XGBoost + RF, SHAP explanations → model_assets/<timestamp>/
retrain_pipeline.py     → Orchestrate full pipeline end-to-end
eda_visualisation.py    → One-time exploratory analysis → eda_plots/
```

---

## Database Schema

**`hdb_resale.db`**

| Table | Purpose |
|---|---|
| `resale_prices` | Core transaction table — one row per transaction |
| `district_summary` | Precomputed aggregates for fast frontend queries (FR4/FR5) |
| `geocode_cache` | Caches OneMap API results — prevents redundant API calls |
| `pipeline_meta` | Tracks last run timestamp and row counts for retrain logic |
| `upload_audit` | Permanent audit trail for admin CSV uploads (FR8) |

**`resale_prices` key columns:**

| Column | Type | Filled by |
|---|---|---|
| month, year, town, flat_type, block, street_name | TEXT/INT | data_pipeline.py |
| floor_area_sqm, storey_midpoint, resale_price | REAL | data_pipeline.py |
| lease_commence_date, remaining_lease | INT/REAL | data_pipeline.py |
| full_address | TEXT | data_pipeline.py |
| latitude, longitude | REAL nullable | geocoding.py |
| dist_mrt, dist_cbd, dist_school, dist_mall | REAL nullable | proximity_features.py |

---

## Execution Order

### First run (full pipeline)

```bash
# Step 1 — Fetch raw data from data.gov.sg API
python api_fetcher.py

# Step 2 — Consolidate, clean, validate → SQLite
python data_pipeline.py

# Step 3 — Geocode every address via OneMap API
python geocoding.py

# Step 4 — Compute proximity distances
python proximity_features.py

# Step 5 — Feature engineering and train/val/test splits
python feature_engineering.py

# Step 6 — Train models and generate SHAP explanations
python ml_model.py
```

### Subsequent runs (automated retraining)

```bash
# Checks for new data, runs full pipeline only if new months exist
python retrain_pipeline.py --trigger cron

# On-demand trigger (e.g. from backend API)
python retrain_pipeline.py --trigger api
```

### One-time EDA (run after data_pipeline.py)

```bash
python eda_visualisation.py
```

---

## File Reference

### `api_fetcher.py`
Fetches all 5 HDB resale datasets from data.gov.sg and saves each as a raw CSV to `/raw`. No cleaning or transformation — raw data only.

- Datasets: 1990–1999, 2000–2012, 2012–2014, 2015–2016, 2017–present
- Handles rate limiting with exponential backoff (HTTP 429)
- `get_latest_available_month()` — used by retrain pipeline to detect new data without downloading

### `data_pipeline.py`
Reads CSVs from `/raw`, consolidates all 5 datasets, cleans, validates, and writes to `hdb_resale.db`.

Key functions:
- `load_raw_csvs()` — reads from `/raw` folder
- `clean_data()` — standardises types, derives full_address, storey_midpoint, remaining_lease
- `validate_data()` — enforces HDB domain logic (see validation rules below)
- `save_to_sqlite()` — writes all tables including district_summary and audit tables
- `ingest_admin_csv()` — FR8 admin upload entry point, deduplicates before insert
- `get_pipeline_meta()` — reads pipeline state for retrain logic

### `geocoding.py`
Resolves every unique `full_address` in `resale_prices` to `(latitude, longitude)` using the OneMap Singapore API. Results are cached in `geocode_cache`.

- Processes in batches of 50 with cache flush after each batch
- Cache-first approach — never re-queries already resolved addresses
- Single SQL UPDATE writes all results back to `resale_prices`

### `proximity_features.py`
Computes four distance features for every geocoded transaction using the Haversine formula. Writes directly back to `resale_prices`.

| Feature | Description |
|---|---|
| `dist_mrt` | km to nearest MRT/LRT station |
| `dist_cbd` | km to Raffles Place MRT (CBD anchor) |
| `dist_school` | km to nearest MOE primary school |
| `dist_mall` | km to nearest shopping mall |

> Reference data (MRT stations, schools, malls) is embedded as module constants. Replace with full datasets before production use.

### `feature_engineering.py`
Transforms the cleaned DB into ML-ready train/val/test splits using EDA-guided decisions.

Split strategy:
- Train: ≤ 2020
- Val: 2021–2022
- Test: ≥ 2023

Key decisions:
- `log1p(resale_price)` as target — corrects right skew
- Ordinal encoding for flat_type
- Target encoding for town and flat_model (fit on train only, prevents leakage)
- StandardScaler for numeric features (fit on train only)
- IQR outlier removal per flat_type — bounds serialised to `outlier_bounds.json`
- Cyclical encoding for month (sin/cos)
- `QUANTILES = [0.10, 0.50, 0.90]` — supports FR2 value range output

Outputs to `model_assets/<YYYYMMDD_HHMMSS>/`:
```
X_train.parquet, X_val.parquet, X_test.parquet
y_train.parquet, y_val.parquet, y_test.parquet
scaler.pkl
target_encoders.pkl
outlier_bounds.json
feature_cols.txt
run_manifest.json
```

### `ml_model.py`
Trains XGBoost quantile models (Q10, Q50, Q90) and a Random Forest baseline. Selects the best Q50 model by validation MAE. Computes SHAP explanations for Premium AI insights (FR6).

Evaluation metrics (all in original SGD scale):
- MAE, RMSE, MAPE, R²

Outputs to same `model_assets/<timestamp>/` run directory:
```
best_model.pkl
xgb_q10.pkl, xgb_q90.pkl
metrics.json
shap_importance.json
shap_summary.png
feature_importance.json
```

### `retrain_pipeline.py`
Orchestrates the full pipeline. Checks for new data before running — exits cleanly if nothing has changed.

```
Step 1 — check_for_new_data()
Step 2 — api_fetcher.run_fetch()
Step 3 — data_pipeline.main()
Step 4 — geocoding.run_geocoding()
Step 5 — proximity_features.run_proximity_features()
Step 6 — feature_engineering.main()
Step 7 — compare metrics, promote model only if better
```

Always writes a structured report to `logs/retrain_YYYYMMDD_HHMMSS.json` — even on failure.

### `eda_visualisation.py`
One-time exploratory analysis. Run after `data_pipeline.py` to understand the data before modelling. Outputs 17 plots to `eda_plots/`.

Key plots:
- Price distribution, price over time, volume over time
- Price by flat type, town, storey, remaining lease
- Correlation heatmap, feature correlations with price
- Skewness and log-transform checks
- Outlier analysis per flat type

---

## Data Validation Rules

`validate_data()` in `data_pipeline.py` enforces HDB domain logic on every row.

### Hard rules — row dropped

| Rule | Condition |
|---|---|
| H1 | `remaining_lease > 99 years` |
| H2 | `remaining_lease < 0` |
| H3 | `lease_commence_date > sale_year` |
| H4 | `lease_commence_date < 1960` |
| H5 | `resale_price <= 0` |
| H6 | `floor_area_sqm <= 0` |
| H7 | Storey range upper bound < lower bound |

### Soft rules — WARNING logged, row kept

| Rule | Condition |
|---|---|
| S1 | `remaining_lease` differs from derived value by > 2 years |
| S2 | Price per sqm outside SGD 1,000–20,000 |
| S3 | Floor area outside expected bounds for flat type |
| S4 | Town not in the 26 known HDB towns |
| S5 | Sale year before 1960 |

---

## Model Assets Structure

```
model_assets/
├── latest.txt                  ← path to current best run directory
└── 20240115_143022/            ← example versioned run
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
    ├── best_model.pkl
    ├── xgb_q10.pkl
    ├── xgb_q90.pkl
    ├── metrics.json
    ├── shap_importance.json
    ├── shap_summary.png
    └── feature_importance.json
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy requests
```

---

## Viewing the Database

Install the **SQLite Viewer** extension in VS Code (by Florian Klampfer). Click `hdb_resale.db` in the file explorer to browse tables and run queries directly.

---

## Logs

Every retrain run writes a structured JSON report to `logs/`:

```json
{
  "triggered_by": "cron",
  "started_at": "2024-01-15T14:30:22Z",
  "finished_at": "2024-01-15T16:45:10Z",
  "new_data_found": true,
  "steps_run": ["api_fetcher", "data_pipeline", "geocoding", "proximity_features", "feature_engineering", "ml_model"],
  "old_metrics": { "test_mae": 28450.0 },
  "new_metrics": { "test_mae": 26830.0 },
  "promoted": true
}
```

---

## Data Sources

| Source | Usage |
|---|---|
| [data.gov.sg HDB Resale Prices](https://data.gov.sg/dataset/resale-flat-prices) | Transaction data 1990–present |
| [OneMap Singapore API](https://www.onemap.gov.sg/apidocs/) | Address geocoding |
| LTA DataMall | MRT/LRT station coordinates |
| MOE | Primary school locations |