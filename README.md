
# PropSight

A full-stack data pipeline and web application for Singapore HDB resale property valuation and market analytics.

This project fetches HDB resale transaction data from public sources, cleans and enriches it with geocoding and proximity features, trains machine learning models to predict resale prices, and serves an interactive web platform for market analysis, price predictions, and transaction visualisation. The current codebase uses **Supabase** (PostgreSQL) as the authoritative runtime database for the web app and as the canonical ML training source, while local SQLite remains a staging layer for ETL and migration into Supabase.

Current README revision created on: `2026-03-27`

---

## Project Overview

The pipeline:

1. fetches HDB resale transaction data from data.gov.sg
2. consolidates and cleans the raw CSV files
3. stores the cleaned data in a SQLite database
4. geocodes HDB addresses using OneMap
5. fetches one-time reference datasets for MRT stations, primary schools, and major malls
6. computes proximity-based location features
7. prepares a clean feature-ready dataset for future machine learning use

The current project scope focuses on:

- HDB resale data only
- clean and reliable data preparation
- location enrichment
- feature engineering
- a Flask web application with interactive maps, charts, and ML-powered price predictions
- Supabase (PostgreSQL) as the cloud database backend with normalised schema and RPC functions

In plain English, this repo does 3 big jobs:

1. builds and cleans the HDB resale dataset
2. trains a time-aware resale price prediction model
3. serves that model inside a Flask website backed by Supabase

## Start Here

Choose the path that matches what you want to do:

- Run the website only:
  `python webapp/app.py`
- Run only the ML pipeline:
  `./.venv/bin/python scripts/run_ml_pipeline.py`
- Run only model training against the latest prepared features:
  `./.venv/bin/python ML/model_training.py`
- Sync the latest data + deployed model metadata to Supabase:
  `./.venv/bin/python scripts/sync_to_supabase.py`
- Run the full retrain + deploy workflow:
  `./.venv/bin/python scripts/retrain_and_deploy.py`

If you are new to the project, the fastest way to understand it is:

1. read the Architecture section below
2. read Execution Order
3. read the Feature Engineering, Model Training, and Latest Model Results sections
4. then open `scripts/pipeline_orchestration.py` to see how the pieces connect

---

## Architecture

```text
hdb_resale/
├── Data Preprocessing/
│   ├── api_fetcher.py           # Fetch raw HDB CSVs from data.gov.sg
│   ├── data_pipeline.py         # Build the local SQLite staging database
│   ├── geocoding.py             # Cache-first geocoding via OneMap
│   ├── fetch_reference_data.py  # Build MRT / school / mall reference JSON
│   ├── proximity_features.py    # Compute block-level distance features
│   ├── eda_visualisation.py     # Optional EDA plots
│   ├── raw/                     # Active raw CSV fetch location
│   ├── raw hdb data/            # Checked-in raw CSV snapshot
│   └── reference_data/          # Checked-in MRT / school / mall JSON
│
├── ML/
│   ├── feature_engineering.py   # Build train / val / test artefacts
│   ├── training_data_source.py  # Supabase-backed training extract helpers
│   ├── model_training.py        # Train XGBoost, LightGBM, RF, ensemble
│   └── model_assets/            # Checked-in model runs and artefacts
│
├── Database/                    # Supabase schema + SQLite -> Supabase migration
│   ├── supabase_schema.sql      # Checked-in Supabase PostgreSQL schema
│   ├── migrate_to_supabase.py   # SQLite → Supabase migration script
│
├── scripts/                     # Shared entry points for preprocessing / ML / deploy
│   ├── pipeline_orchestration.py
│   ├── run_data_preprocessing.py
│   ├── run_ml_pipeline.py
│   ├── sync_to_supabase.py
│   └── retrain_and_deploy.py
│
├── webapp/
│   ├── app.py                   # Flask app, auth, APIs, predictions
│   ├── templates/               # Jinja2 templates
│   └── static/                  # CSS and images
│
├── Data Preprocessing/hdb_resale.db   # Local SQLite staging DB created by ETL
└── README.md
```

Most ETL and ML scripts use relative paths such as `raw/`, `reference_data/`, `hdb_resale.db`, and `model_assets/`, so their outputs depend on the working directory you run them from.

---

## Tech Stack

| Layer | Technologies |
| ----- | ------------ |
| Backend | Python, Flask, SQLite, PostgreSQL (Supabase) |
| Frontend | HTML5, Jinja2, Bootstrap 5.3, Leaflet.js, Chart.js |
| ML | XGBoost, LightGBM, scikit-learn, Optuna |
| Data | Pandas, NumPy, SciPy |
| APIs | data.gov.sg, OneMap Singapore, Supabase REST & Auth |
| Auth | Supabase Auth + Werkzeug password hashing |

---

## Subscription Tiers

The platform uses a freemium model with two subscription tiers:

| Feature | General (Free) | Premium ($4.90/month) |
| ------- | -------------- | --------------------- |
| Price predictions | Unlimited | Unlimited |
| Save predictions | Up to 3 | Unlimited |
| Interactive map | 3 views/week | Unlimited |
| Analytics dashboard | 3 views/week | Unlimited |
| Comparison tool | 3 views/week | Unlimited |

- New users default to the **General** tier.
- The `subscription_tier` column on the `users` table is constrained to `'general'` or `'premium'`.
- Weekly view limits for Map, Analytics, and Comparison are tracked via the `feature_view_log` table.
- The `/pricing` page shows the plan comparison; logged-in general users can upgrade via the Upgrade button.

---

## Web Application

The Flask app in `webapp/app.py` loads `.env` from the project root, reads trained model artefacts at startup, and uses Supabase REST/RPC plus Supabase Auth for the live website. The current runtime will raise a startup error if `SUPABASE_URL` and either `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_KEY` are missing.

At startup, the app resolves model artefacts from:

- `MODEL_ASSETS_DIR` if set
- `./ML/model_assets`

The active run is chosen from `latest.txt` when present, otherwise the newest run directory under the artefact root.

User sessions store:

- `user_id`
- `username`
- `email`
- `access_token`
- `subscription_tier`

Feature limits are enforced on page routes for General users:

- Map: 3 views per week
- Analytics: 3 views per week
- Comparison: 3 views per week

Premium users get unlimited feature views, unlimited saved predictions, and up to 5 comparison panels. General users are limited to 3 saved predictions and 2 comparison panels.

### Page Routes

| Route | Access | Purpose |
| ----- | ------ | ------- |
| `/` | Public | Landing page with total transaction count, model MAPE, public teaser data, and popular or personalized prediction cards |
| `/register` | Public (`GET`, `POST`) | Create a Supabase Auth account and provision a matching row in `public.users` |
| `/login` | Public (`GET`, `POST`) | Sign in with Supabase password auth and hydrate the Flask session |
| `/forgot-password` | Public (`GET`, `POST`) | Trigger Supabase password recovery email |
| `/logout` | Session route | Clear the current login session and attempt Supabase logout |
| `/pricing` | Public | Pricing and subscription comparison page |
| `/upgrade` | Login required (`POST`) | Patch the current user to `premium` in `public.users` |
| `/predict` | Login required (`GET`, `POST`) | Prediction form with optional query-param prefill, recent similar transactions, and 5-year forecast |
| `/save_prediction` | Login required (`POST`) | Save the current prediction to `saved_predictions` |
| `/my_predictions` | Login required | List saved predictions |
| `/delete_prediction/<int:pred_id>` | Login required (`POST`) | Delete one saved prediction |
| `/my_predictions/bulk_delete` | Login required (`POST`) | Delete multiple saved predictions |
| `/comparison` | Login required (`GET`, `POST`) | Compare 2 properties for General users or up to 5 for Premium users |
| `/comparison/select/<int:pred_id>` | Login required | Push a saved prediction into the comparison-session state |
| `/map` | Login required | Interactive map with transaction pins and predicted heatmap modes |
| `/analytics` | Login required | Analytics dashboard with trend charts, context cards, and prediction-linked deep links |

### JSON Endpoints

Authenticated endpoints:

- `GET /api/transactions` — transaction rows for the map pin layer
- `GET /api/district_summary` — town-level summary rows with coordinates for heatmap-style views
- `GET /api/predicted_heatmap` — per-town model predictions for the map heatmap mode
- `GET /api/price_trend` — yearly trend data normalized to `avg_price`, `q1`, `q3`, and `txn_count`
- `GET /api/price_trend_simple` — yearly average/min/max trend data with optional town, flat type, street, and block filters
- `GET /api/street_price_trend` — yearly street-level trend data inside a town
- `GET /api/district_comparison` — latest per-town comparison data
- `GET /api/flat_type_breakdown` — average price, transaction count, and average floor area by flat type
- `GET /api/monthly_volume` — monthly transaction counts and average price
- `GET /api/available_models` — valid flat models for a town + flat type pair
- `GET /api/available_storey_ranges` — valid storey ranges for a town + flat type pair
- `GET /api/floor_area_stats` — min / max / average floor area for a town + flat type pair
- `GET /api/lease_year_range` — min / max / average lease commence year for a town
- `GET /api/available_streets` — street list for a town
- `GET /api/available_blocks` — block list for a town + street pair
- `GET /api/prediction_context` — lease-decay data plus recent comparable transactions for analytics
- `GET /api/future_prediction` — 5-year forward prediction timeline as JSON

Public endpoints:

- `GET /api/public/location_summary` — guest teaser map with blurred town-level price buckets
- `GET /api/public/recent_ticker` — 20 most recent transactions for the homepage ticker

### Start the Website (No ETL / No Model Training)

Use this path if you only want to run the Flask website. The current app reads its runtime data from Supabase, so you do not need a local website database.

You do not need to run:

- `Data Preprocessing/api_fetcher.py`
- `Data Preprocessing/data_pipeline.py`
- `Data Preprocessing/geocoding.py`
- `Data Preprocessing/fetch_reference_data.py`
- `Data Preprocessing/proximity_features.py`
- `ML/feature_engineering.py`
- `ML/model_training.py`

Minimum requirements:

- trained model artefacts under `ML/model_assets/`
- a `.env` file with `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_KEY`), and `SECRET_KEY`

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install flask python-dotenv werkzeug xgboost lightgbm pandas numpy scikit-learn scipy pyarrow "numexpr>=2.10.2" "bottleneck>=1.4.2"
```

Download or copy the trained model artefacts so this structure exists:

```text
ML/model_assets/
├── latest.txt
└── <run_dir>/
    ├── xgboost_model.pkl or lgbm_model.pkl or rf_model.pkl or ensemble_model.pkl
    ├── scaler.pkl
    ├── target_encoders.pkl
    └── metrics.json
```

Set the required variables in `.env`:

```bash
SECRET_KEY=<flask-session-secret>
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

Then start the app:

```bash
python webapp/app.py
```

Notes:

- `webapp/app.py` automatically loads `.env` from the project root.
- The app will fail at startup if `SUPABASE_URL` and a Supabase key are not set.
- The app loads the active run from `ML/model_assets/latest.txt`, so retraining is not required just to run the website.
- The app serves the winner declared in `metrics.json` when present and reads performance metadata dynamically from the same run.
- For current runs, `price_index.pkl` is optional because the newer target is `log1p(resale_price)` rather than RPI-adjusted log price.
- `hdb_resale.db` is still used by the local ETL and migration flow, but it is not used by the website runtime.
- On macOS, `xgboost` may require OpenMP. If import fails with `libomp.dylib` not found, run `brew install libomp`.
- The default dev server port is `5001`. Override it with `FLASK_PORT=<port> python webapp/app.py` if `5001` is already in use.

---

## Supabase Backend (PostgreSQL)

The checked-in PostgreSQL schema lives in `Database/supabase_schema.sql`. `Database/migrate_to_supabase.py` migrates rows from the local SQLite `resale_prices` table into the normalized Supabase tables.

At runtime, `webapp/app.py` requires both `SUPABASE_URL` and either `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_KEY`. Supabase is the authoritative application database for the website and the canonical source used by the current ML feature engineering step.

### Normalised Schema

| Table | Purpose |
| ----- | ------- |
| `towns` | Dimension table — unique HDB town names |
| `flat_types` | Dimension table — flat type categories |
| `flat_models` | Dimension table — HDB flat model types |
| `blocks` | Address table — block + street + town with pre-computed distances |
| `transactions` | Fact table — resale transactions with foreign keys to dimension tables |
| `users` | Public app-user table with `subscription_tier` column (`'general'` or `'premium'`) |
| `saved_predictions` | Saved prediction rows written by `/save_prediction` |
| `feature_view_log` | Tracks per-user weekly views of Map, Analytics, and Comparison features |
| `model_versions` | Model metadata table; not currently read by `webapp/app.py` |

### RPC Functions

The current SQL file defines the RPC functions used by `webapp/app.py`, including:

- Lookup RPCs: `rpc_get_towns`, `rpc_get_flat_models`, `rpc_get_town_avg_distances`, `rpc_available_streets`, `rpc_available_blocks`, `rpc_block_distances`
- Analytics RPCs: `rpc_api_transactions`, `rpc_api_district_summary`, `rpc_api_price_trend_simple`, `rpc_api_district_comparison`, `rpc_api_flat_type_breakdown`, `rpc_api_monthly_volume`, `rpc_lease_decay`, `rpc_recent_similar_transactions`
- Prediction RPCs: `rpc_predict_trend`, `rpc_predict_benchmarks`, `rpc_resolve_floor_area`, `rpc_resolve_lease_commence`
- UI support RPCs: `rpc_count_transactions`, `rpc_api_available_models`, `rpc_api_available_storey_ranges`, `rpc_api_floor_area_stats`, `rpc_api_lease_year_range`, `rpc_api_public_location_summary`, `rpc_api_public_recent_ticker`

### Environment Variables

Relevant `.env` values used by the current code:

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
SUPABASE_KEY=<optional alternate key name accepted by webapp/app.py>
SUPABASE_DB_URL=postgresql://...
SECRET_KEY=<flask-session-secret>
MODEL_ASSETS_DIR=<optional model artefacts override>
SUPABASE_USERS_TABLE=users
SUPABASE_PREDICTIONS_TABLE=saved_predictions
```

---

## Database Schema (PostgreSQL / Supabase)

The production database uses a normalised PostgreSQL schema hosted on Supabase. The full schema is defined in `Database/supabase_schema.sql`.

### Dimension Tables

#### `towns`

| Column             | Type    | Description                     |
| ------------------ | ------- | ------------------------------- |
| `id`               | SERIAL  | Primary key                     |
| `name`             | TEXT    | Town name (unique)              |
| `is_mature_estate` | BOOLEAN | Whether the town is mature      |

#### `flat_types`

| Column    | Type    | Description                     |
| --------- | ------- | ------------------------------- |
| `id`      | SERIAL  | Primary key                     |
| `name`    | TEXT    | Flat type name (unique)         |
| `ordinal` | INTEGER | Sort order (1R=1, 2R=2, etc.)  |

#### `flat_models`

| Column | Type   | Description                |
| ------ | ------ | -------------------------- |
| `id`   | SERIAL | Primary key                |
| `name` | TEXT   | Flat model name (unique)   |

### Address / Location Table

#### `blocks`

One row per unique block + street combination, with pre-computed location features.

| Column              | Type             | Description                        |
| ------------------- | ---------------- | ---------------------------------- |
| `id`                | SERIAL           | Primary key                        |
| `block`             | TEXT             | Block number                       |
| `street_name`       | TEXT             | Street name                        |
| `town_id`           | INTEGER          | FK → `towns(id)`                   |
| `full_address`      | TEXT             | Derived full address               |
| `latitude`          | DOUBLE PRECISION | Geocoded latitude                  |
| `longitude`         | DOUBLE PRECISION | Geocoded longitude                 |
| `dist_mrt`          | DOUBLE PRECISION | Distance to nearest MRT/LRT (km)  |
| `dist_cbd`          | DOUBLE PRECISION | Distance to CBD anchor point (km)  |
| `dist_primary_school` | DOUBLE PRECISION | Distance to nearest primary school |
| `dist_major_mall`   | DOUBLE PRECISION | Distance to nearest major mall     |

### Fact Table

#### `transactions`

One row per resale transaction, with foreign keys to dimension tables.

| Column                 | Type             | Description                      |
| ---------------------- | ---------------- | -------------------------------- |
| `id`                   | SERIAL           | Primary key                      |
| `block_id`             | INTEGER          | FK → `blocks(id)`               |
| `flat_type_id`         | INTEGER          | FK → `flat_types(id)`           |
| `flat_model_id`        | INTEGER          | FK → `flat_models(id)`          |
| `storey_range`         | TEXT             | Original storey range            |
| `storey_midpoint`      | DOUBLE PRECISION | Derived midpoint of storey range |
| `floor_area_sqm`       | DOUBLE PRECISION | Floor area in square metres      |
| `lease_commence_date`  | INTEGER          | Lease commencement year          |
| `remaining_lease`      | DOUBLE PRECISION | Remaining lease in years         |
| `remaining_lease_months` | DOUBLE PRECISION | Remaining lease in months      |
| `resale_price`         | DOUBLE PRECISION | Resale price in SGD              |
| `month`                | TEXT             | Transaction month (YYYY-MM)      |
| `month_num`            | INTEGER          | Transaction month number         |
| `year`                 | INTEGER          | Transaction year                 |

### User Tables

#### `users`

| Column          | Type                     | Description             |
| --------------- | ------------------------ | ----------------------- |
| `id`                | SERIAL                   | Primary key             |
| `username`          | TEXT                     | Unique username         |
| `email`             | TEXT                     | Unique email address    |
| `password_hash`     | TEXT                     | Hashed password         |
| `subscription_tier` | TEXT                     | `'general'` or `'premium'` (default `'general'`) |
| `created_at`        | TIMESTAMP WITH TIME ZONE | Account creation time   |

#### `saved_predictions`

| Column            | Type                     | Description                        |
| ----------------- | ------------------------ | ---------------------------------- |
| `id`              | SERIAL                   | Primary key                        |
| `user_id`         | INTEGER                  | FK → `users(id)` (cascade delete)  |
| `town`            | TEXT                     | Predicted town                     |
| `flat_type`       | TEXT                     | Predicted flat type                |
| `flat_model`      | TEXT                     | Predicted flat model               |
| `floor_area`      | DOUBLE PRECISION         | Floor area used                    |
| `storey_range`    | TEXT                     | Storey range used                  |
| `lease_commence`  | INTEGER                  | Lease commence year used           |
| `predicted_price` | DOUBLE PRECISION         | Predicted resale price             |
| `price_low`       | DOUBLE PRECISION         | Lower bound (80% confidence)       |
| `price_high`      | DOUBLE PRECISION         | Upper bound (80% confidence)       |
| `created_at`      | TIMESTAMP WITH TIME ZONE | Prediction save time               |

#### `feature_view_log`

| Column       | Type                     | Description                                |
| ------------ | ------------------------ | ------------------------------------------ |
| `id`         | SERIAL                   | Primary key                                |
| `user_id`    | INTEGER                  | FK → `users(id)` (cascade delete)          |
| `feature`    | TEXT                     | Feature name (`'map'`, `'analytics'`, `'comparison'`) |
| `created_at` | TIMESTAMP WITH TIME ZONE | View timestamp                             |

#### `model_versions`

| Column      | Type                     | Description                  |
| ----------- | ------------------------ | ---------------------------- |
| `id`        | SERIAL                   | Primary key                  |
| `version`   | TEXT                     | Model version label          |
| `run_dir`   | TEXT                     | Model assets run directory   |
| `trained_at` | TIMESTAMP WITH TIME ZONE | Training timestamp           |
| `test_mape` | DOUBLE PRECISION         | Test MAPE score              |
| `test_rmse` | DOUBLE PRECISION         | Test RMSE score              |
| `test_r2`   | DOUBLE PRECISION         | Test R² score                |
| `notes`     | TEXT                     | Optional notes               |
| `is_active` | BOOLEAN                  | Whether this is the serving model |

### Index and Constraint Notes

- `towns.name`, `flat_types.name`, `flat_models.name`, `users.username`, and `users.email` are unique.
- `blocks` has `UNIQUE (block, street_name)` plus indexes on `town_id` and `(latitude, longitude)`.
- `transactions` has indexes on `year`, `block_id`, `flat_type_id`, and `(year, block_id)`.
- `saved_predictions.user_id` uses `ON DELETE CASCADE`.
- `is_mature_estate`, `ordinal`, `created_at`, `trained_at`, and `is_active` all have SQL defaults in `supabase_schema.sql`.

---

## Local SQLite Database

The current codebase still depends on SQLite for local preprocessing and for staging rows before they are migrated into Supabase. The website runtime does not use SQLite, and the ML training pipeline now reads from Supabase.

### Main Analytics DB (`hdb_resale.db`)

| Table | Purpose |
| ----- | ------- |
| `resale_prices` | Transaction table used by local ETL and as the source for SQLite → Supabase migration |
| `district_summary` | Rebuilt aggregate summary table for town / flat_type / year analytics |
| `geocode_cache` | Cached OneMap geocoding results |
| `pipeline_meta` | Key-value metadata written by the ETL pipeline |
| `upload_audit` | Append-only audit trail for ingestion runs |

---

## Execution Order

### Website only

If you only want to run the website, skip the full pipeline and start the Flask app directly:

```bash
python webapp/app.py
```

This requires Supabase credentials in `.env` and model artefacts in `ML/model_assets/`.

### Recommended script entry points

Use the `scripts/` entry points from the project root whenever possible:

```bash
./.venv/bin/python scripts/run_data_preprocessing.py
./.venv/bin/python scripts/run_ml_pipeline.py
./.venv/bin/python scripts/sync_to_supabase.py
./.venv/bin/python scripts/retrain_and_deploy.py
```

These wrappers call the shared orchestration logic in `scripts/pipeline_orchestration.py` and are easier to run consistently than invoking every lower-level file manually.

### Manual end-to-end pipeline run

```bash
python "Data Preprocessing/api_fetcher.py"
python "Data Preprocessing/data_pipeline.py"
python "Data Preprocessing/geocoding.py"
python "Data Preprocessing/fetch_reference_data.py"
python "Data Preprocessing/proximity_features.py"
python Database/migrate_to_supabase.py
python ML/feature_engineering.py
python ML/model_training.py
```

Important:

* `fetch_reference_data.py` is a one-time setup step.
* Run it before the first run of `proximity_features.py`.
* You do not need to run it again on every pipeline run unless you want to refresh the MRT, school, or major mall reference datasets.
* These scripts use relative paths such as `raw/`, `reference_data/`, `hdb_resale.db`, and `model_assets/`, so the exact output location depends on the working directory.

Geocoding note:

* `geocoding.py` is rerunnable and incremental, not one-time.
* Run it after `data_pipeline.py`.
* Run it again whenever new resale rows are added or when you want to retry unresolved addresses.

### Optional EDA

```bash
python "Data Preprocessing/eda_visualisation.py"
```

---

## File Reference

### `api_fetcher.py`

Fetches HDB resale datasets from data.gov.sg and saves raw CSV files into a relative `raw/` directory.

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
* rebuild `district_summary`
* upsert `pipeline_meta`
* preserve `geocode_cache`
* preserve `upload_audit`

---

### `geocoding.py`

Geocodes unique HDB addresses using the OneMap API and updates `latitude` and `longitude` in `resale_prices`.

Main responsibilities:

* use cache-first lookup through `geocode_cache`
* avoid repeated API calls
* batch-save geocoding results
* bulk-update all matching rows in `resale_prices`

Important usage note:

* this script is safe to rerun
* it is not a one-time setup script
* rerun it whenever the database contains new ungeocoded addresses

---

### `fetch_reference_data.py`

Fetches and saves the reference datasets used by `proximity_features.py`.

Main responsibilities:

* fetch MRT station reference data from LTA static geospatial data
* fetch primary school names from data.gov.sg and geocode them with OneMap
* geocode a curated major mall list with OneMap
* save JSON files to a relative `reference_data/` directory

Important usage note:

* run this script once before running `proximity_features.py`
* rerun only if you want to refresh the reference datasets

---

### `proximity_features.py`

Computes proximity-based location features for geocoded transactions.

Generated features:

* `dist_mrt`
* `dist_cbd`
* `dist_primary_school`
* `dist_major_mall`

These features are written back to `resale_prices`.

---

### `feature_engineering.py`
Transforms the canonical Supabase training extract into clean, feature-ready train/val/test splits.

Split strategy:
- Train: random 90% sample from rows earlier than the rolling test cutoff
- Val: random 10% sample from the same pre-cutoff window, stratified by `flat_type`
- Test: most recent `test_months` of data by `year-month` (current default: 3 months)
- Future holdout: optional only for older run formats; not produced by the current rolling-holdout flow

Key decisions:
- newer runs use `log1p(resale_price)` as the training target
- legacy runs with `price_index` are still supported
- Ordinal encoding for `flat_type`
- Target encoding for `town` and `flat_model` (fit on train only to reduce leakage)
- `StandardScaler` for numeric features (fit on train only)
- high-price but valid transactions are kept in training
- `outlier_bounds.json` is now diagnostic only
- Cyclical encoding for month using `month_sin` and `month_cos`
- Removal of rows with missing required fields so saved datasets contain no null values
- feature engineering now reads from the canonical Supabase training source and stores split metadata in `run_manifest.json`
- Saves processed artefacts to `model_assets/<YYYYMMDD_HHMMSS>/`

Outputs:
- `X_train.parquet`, `X_val.parquet`, `X_test.parquet`, `X_future_holdout.parquet` when available
- `y_train.parquet`, `y_val.parquet`, `y_test.parquet`, `y_future_holdout.parquet` when available
- `scaler.pkl`
- `target_encoders.pkl`
- `outlier_bounds.json`
- `feature_cols.txt`
- `run_manifest.json`
- `metrics.json` (stub)
---

### `model_training.py`
Trains and evaluates machine learning models using the pre-split artefacts from `feature_engineering.py`.

Models trained:
- XGBoost (Optuna-tuned)
- LightGBM (Optuna-tuned)
- Random Forest (baseline)
- Stacked ensemble (Ridge meta-learner over base model predictions)

Key decisions:
- Reads split files from `model_assets/latest.txt`
- Uses train split for fitting, val split for tuning/early stopping, and keeps test locked for final reporting only
- XGBoost and LightGBM are tuned on a deterministic train subset for speed, then retrained on the full train split
- current defaults are `XGB_TRIALS=25`, `LGBM_TRIALS=25`, `TUNING_SAMPLE_SIZE=250000`, `XGB_N_ESTIMATORS=2000`, `LGBM_N_ESTIMATORS=2000`, and `RF_ESTIMATORS=300`
- Optuna studies can be resumed from saved `.pkl` files, and `FRESH_TUNING=1` forces a clean retune
- Reports metrics in SGD space (`RMSE`, `MAE`, `R2`, `MAPE`) by inverse-transforming `log1p` predictions
- Winner selection now chooses the best **base model** on validation RMSE
- The ensemble is reported separately and excluded from production winner selection until stacking is rebuilt with out-of-fold predictions or a dedicated meta-validation split
- Writes all training outputs into the same timestamped run folder created by `feature_engineering.py`

Outputs written to `model_assets/<YYYYMMDD_HHMMSS>/`:
- `xgboost_model.pkl`
- `lgbm_model.pkl`
- `rf_model.pkl`
- `ensemble_model.pkl`
- `optuna_study_xgb.pkl`
- `optuna_study_lgbm.pkl`
- `metrics.json` (updated with full model results and winner)
- `model_comparison.json`
- `training_report.txt`
---

## Latest Model Training Results

Latest run:
- Run directory: `ML/model_assets/20260326_182112/`
- Model date (timekeeping): `2026-03-26` (derived from run folder timestamp `YYYYMMDD_HHMMSS`)
- Recommended serving model: `lgbm_model.pkl`

Split used in this run:

- Train: random 90% of rows before `2026-01`
- Val: random 10% of rows before `2026-01`
- Test: `>= 2026-01` rolling 3-month holdout

Model comparison:

| Model      | Val RMSE | Test RMSE | Test R2 | Test MAPE |
| ---------- | -------- | --------- | ------- | --------- |
| XGBoost    | 19,092   | 34,020    | 0.9740  | 3.68%     |
| LightGBM   | 18,868   | 34,142    | 0.9738  | 3.69%     |
| RF         | 21,572   | 40,400    | 0.9633  | 4.14%     |
| Ensemble   | 18,678   | 33,781    | 0.9743  | 3.63%     |

Winner:
- `lgbm` is the current serving recommendation because it is the strongest production-safe base model on validation RMSE.
- The ensemble scored slightly better on the locked test holdout, but it is not selected for serving because its Ridge meta-learner was fit on validation predictions.

Important interpretation note:

- These scores come from a stricter rolling holdout, so the test set is a better proxy for forward-looking drift than a random split alone.
- Validation remains useful for tuning and production-safe base-model selection, but it is easier than the January 2026+ test window because the current validation split is still random within the pre-cutoff period.
- The ensemble currently looks strongest numerically, but it is intentionally excluded from production winner selection until stacking becomes leakage-safe.

Upcoming ensemble fix:

- rebuild the ensemble using out-of-fold stacking on the training set
- re-evaluate it fairly against base models
- make it deployable as a first-class website serving model if it still wins

Timekeeping note:
- Use the run folder timestamp (for example `20260326_151359`) as the model version/date key.
- `ML/model_assets/latest.txt` currently points to the checked-in active run.
- `run_manifest.json` stores run metadata, and `training_report.txt` stores per-run training/evaluation details.
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

The checked-in artefacts currently live under `ML/model_assets/`. The ML scripts default to writing there as well, because they run from the `ML/` directory unless `MODEL_ASSETS_DIR` overrides that path.

```text
model_assets/
├── latest.txt
├── outlier_bounds.json
└── YYYYMMDD_HHMMSS/
    ├── X_train.parquet
    ├── X_val.parquet
    ├── X_test.parquet
    ├── X_future_holdout.parquet
    ├── y_train.parquet
    ├── y_val.parquet
    ├── y_test.parquet
    ├── y_future_holdout.parquet
    ├── scaler.pkl
    ├── target_encoders.pkl
    ├── feature_cols.txt
    ├── run_manifest.json
    ├── metrics.json
    ├── xgboost_model.pkl
    ├── lgbm_model.pkl
    ├── rf_model.pkl
    ├── ensemble_model.pkl
    ├── optuna_study_xgb.pkl
    ├── optuna_study_lgbm.pkl
    ├── model_comparison.json
    └── training_report.txt
```

---

## Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy requests pyarrow xgboost lightgbm optuna flask python-dotenv werkzeug psycopg2-binary "numexpr>=2.10.2" "bottleneck>=1.4.2"
```


## Data Sources

| Source                        | Usage                           |
| ----------------------------- | ------------------------------- |
| data.gov.sg HDB resale prices | Main transaction dataset        |
| OneMap Singapore API          | Address geocoding               |
| MRT/LRT reference data        | Nearest station distance        |
| School reference data         | Nearest primary school distance |
| Major mall reference data     | Nearest major mall distance     |

---
