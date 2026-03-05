# HDB Resale Flat Price Prediction

Final Year Project (FYP) — SIM-UOW
Predicting Singapore HDB resale flat prices using machine learning.

---

## Project Overview

This project builds an end-to-end pipeline to:
1. Fetch historical HDB resale transaction data from the Singapore government API
2. Clean and prepare the dataset with HDB domain logic
3. Perform exploratory data analysis (EDA) to guide feature selection
4. Preprocess features for ML modelling
5. Train and evaluate models to predict resale flat prices

**Dataset:** [HDB Resale Flat Prices — data.gov.sg](https://data.gov.sg/collections/189/view)
**Coverage:** 1990 – present (~972,000 transactions)

---

## Project Structure

```
├── data_pipeline.py        # Fetch from API → clean → save to SQLite
├── preprocessing.py        # Feature engineering → encode → scale → train/val/test split
├── eda_visualisation.py    # 17 EDA plots for feature selection analysis
├── README.md
└── .gitignore
```

> **Note:** `hdb_resale.db` and generated folders (`eda_plots/`, `processed_data/`) are excluded from the repository. Regenerate them by running the scripts in order.

---

## Setup

```bash
pip install pandas requests matplotlib seaborn scipy scikit-learn
```

---

## How to Run

### 1. Fetch and clean the dataset
```bash
python data_pipeline.py
```
Fetches all 5 datasets from data.gov.sg, cleans them, and saves to `hdb_resale.db`.

### 2. Explore the data
```bash
python eda_visualisation.py
```
Generates 17 plots in `eda_plots/` covering distributions, correlations, outliers, and feature-selection insights.

### 3. Preprocess for ML
```bash
python preprocessing.py
```
Engineers features, encodes categoricals, scales numerics, and saves train/val/test splits to `processed_data/`.

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `month` | TEXT | Sale month (`YYYY-MM`) |
| `year` | INTEGER | Sale year |
| `month_num` | INTEGER | Sale month number (1–12) |
| `town` | TEXT | HDB town (e.g. `ANG MO KIO`) |
| `flat_type` | TEXT | Flat type (e.g. `4 Room`, `Executive`) |
| `block` | TEXT | Block number |
| `street_name` | TEXT | Street name |
| `storey_range` | TEXT | Storey range (e.g. `07 TO 09`) |
| `storey_midpoint` | REAL | Numeric midpoint of storey range |
| `floor_area_sqm` | REAL | Floor area in square metres |
| `flat_model` | TEXT | Flat model (e.g. `New Generation`, `DBSS`) |
| `lease_commence_date` | INTEGER | Year the 99-year lease started |
| `remaining_lease` | REAL | Remaining lease in years (e.g. `61.33`) |
| `remaining_lease_months` | REAL | Remaining lease in months |
| `resale_price` | REAL | **Target variable** — transaction price in SGD |

---

## EDA Plots

| Plot | Description |
|------|-------------|
| 01 | Missing values per column |
| 02 | Resale price distribution |
| 03 | Median price over time (quarterly) |
| 04 | Transaction volume per year |
| 05 | Price by flat type |
| 06 | Median price by town |
| 07 | Price vs floor area |
| 08 | Median price by storey level |
| 09 | Price vs remaining lease (years) |
| 10 | Correlation heatmap |
| 11 | Flat type mix over time |
| 12 | Price trend by flat type |
| 13 | Numeric feature distributions + skewness |
| 14 | Feature correlations with resale price |
| 15 | Log transform check (price & floor area) |
| 16 | Price by flat model |
| 17 | Outlier analysis per flat type |
