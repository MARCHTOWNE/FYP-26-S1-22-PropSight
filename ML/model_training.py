"""
model_training.py
=================
Single responsibility: train, tune, and evaluate ML models for HDB resale
price prediction using the feature-ready datasets produced by
feature_engineering.py.

Design decisions:
  - Reads pre-split parquets from the run directory pointed to by
    model_assets/latest.txt. No feature engineering is done here.
  - Target is log1p(resale_price / price_index), i.e. RPI-adjusted
    log price. Predictions are denormalized back to SGD via
    expm1(pred) * price_index. This removes macro price trends so
    models learn structural property value.
  - Four models: XGBoost, LightGBM, Random Forest, and a stacked
    ensemble (XGB + LGBM + RF base learners, Ridge meta-learner).
  - Optuna tunes XGBoost and LightGBM (100 trials each) on val RMSE.
  - Random Forest uses a sensible fixed configuration (no tuning).
  - Train set is for fitting; val set is for tuning and early stopping;
    test set is touched once at the very end.
  - All random states are seeded to 42 for reproducibility.
  - Runs a data quality diagnostic before training and includes the
    findings in the training report.

Execution order context:
  Step 7 of 7. Reads: model_assets/<run_dir>/*.parquet.
  Writes: model_assets/<run_dir>/{models, metrics, reports}.
  Previous step: feature_engineering.py.

Run:
    python model_training.py
"""

import json
import os
import pickle
import sqlite3
import sys
import time
from textwrap import dedent

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

_MISSING = []
try:
    import xgboost as xgb
except ImportError:
    _MISSING.append("xgboost")
try:
    import lightgbm as lgb
except ImportError:
    _MISSING.append("lightgbm")
try:
    import optuna
except ImportError:
    _MISSING.append("optuna")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if _MISSING:
    print(f"Missing required packages: {', '.join(_MISSING)}")
    print(f"Install with:  pip install {' '.join(_MISSING)}")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH        = "hdb_resale.db"
OUTPUT_DIR     = "model_assets"
LATEST_FILE    = os.path.join(OUTPUT_DIR, "latest.txt")
SEED           = 42
XGB_TRIALS     = 100
LGBM_TRIALS    = 100
N_RF_ESTIMATORS = 500

# Flat type labels used for error breakdown
FLAT_TYPES = [
    "1 Room", "2 Room", "3 Room", "4 Room",
    "5 Room", "Executive", "Multi-Generation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir() -> str:
    """
    Read model_assets/latest.txt and return the absolute run directory path.

    Returns:
        Absolute path to the latest feature engineering run directory.

    Raises:
        FileNotFoundError: If latest.txt is missing or the directory does not exist.
    """
    if not os.path.isfile(LATEST_FILE):
        raise FileNotFoundError(
            f"'{LATEST_FILE}' not found. Run feature_engineering.py first."
        )
    with open(LATEST_FILE) as f:
        run_dir = f.read().strip()
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")
    return run_dir


def _load_splits(run_dir: str) -> tuple:
    """
    Load X and y parquets for train, val, and test splits.

    Parameters:
        run_dir: Path to the feature engineering run directory.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) DataFrames.
    """
    splits = {}
    for name in ("train", "val", "test"):
        X = pd.read_parquet(os.path.join(run_dir, f"X_{name}.parquet"))
        y = pd.read_parquet(os.path.join(run_dir, f"y_{name}.parquet"))
        splits[name] = (X, y)
    return (
        splits["train"][0], splits["train"][1],
        splits["val"][0],   splits["val"][1],
        splits["test"][0],  splits["test"][1],
    )


def _inv_transform(y_log: np.ndarray, price_index: np.ndarray) -> np.ndarray:
    """Convert RPI-adjusted log1p(price) back to actual SGD.

    predicted_sgd = expm1(log_prediction) * price_index
    """
    return np.expm1(y_log) * price_index


def _compute_metrics(y_true_sgd: np.ndarray, y_pred_sgd: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, R-squared, and MAPE in actual SGD.

    Parameters:
        y_true_sgd: Ground truth resale prices in SGD.
        y_pred_sgd: Predicted resale prices in SGD.

    Returns:
        Dict with keys rmse, mae, r2, mape.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true_sgd, y_pred_sgd)))
    mae  = float(mean_absolute_error(y_true_sgd, y_pred_sgd))
    r2   = float(r2_score(y_true_sgd, y_pred_sgd))
    mape = float(np.mean(np.abs((y_true_sgd - y_pred_sgd) / y_true_sgd)) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def _fmt_metrics(m: dict) -> str:
    """Format a metrics dict as a one-line summary string."""
    return (
        f"RMSE={m['rmse']:,.0f}  MAE={m['mae']:,.0f}  "
        f"R2={m['r2']:.4f}  MAPE={m['mape']:.2f}%"
    )


def _error_by_flat_type(
    y_true_sgd: np.ndarray,
    y_pred_sgd: np.ndarray,
    flat_types_series: pd.Series,
) -> dict:
    """
    Compute MAE breakdown by flat_type.

    Parameters:
        y_true_sgd:       Actual prices in SGD.
        y_pred_sgd:       Predicted prices in SGD.
        flat_types_series: Series of flat_type labels aligned with predictions.

    Returns:
        Dict mapping flat_type to {mae, count}.
    """
    result = {}
    for ft in flat_types_series.unique():
        mask = flat_types_series.values == ft
        if mask.sum() == 0:
            continue
        mae = float(mean_absolute_error(y_true_sgd[mask], y_pred_sgd[mask]))
        result[str(ft)] = {"mae": round(mae, 2), "count": int(mask.sum())}
    return result


# ---------------------------------------------------------------------------
# Data quality diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(run_dir: str) -> str:
    """
    Query hdb_resale.db and the parquet splits to produce a data quality
    diagnostic report. Returns the report as a string.

    Parameters:
        run_dir: Path to the feature engineering run directory.

    Returns:
        Multi-line diagnostic report string.
    """
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("DATA QUALITY DIAGNOSTIC REPORT")
    lines.append(sep)

    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM resale_prices").fetchone()[0]
    lines.append(f"\n1. Total rows in resale_prices: {total:,}")

    # Geocoding coverage
    null_lat = conn.execute(
        "SELECT COUNT(*) FROM resale_prices WHERE latitude IS NULL"
    ).fetchone()[0]
    geocode_pct = (total - null_lat) / total * 100
    lines.append(f"\n2. Geocoding coverage:")
    lines.append(f"   Null latitude/longitude: {null_lat:,} ({null_lat/total*100:.1f}%)")
    lines.append(f"   Geocoded:                {total - null_lat:,} ({geocode_pct:.1f}%)")
    if geocode_pct < 80:
        lines.append(
            "   *** WARNING: Geocoding coverage below 80%! "
            "Proximity features will have limited signal."
        )
        lines.append("   *** Recommend rerunning geocoding.py first.")

    # Null proximity features
    for col in ["dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"]:
        null_c = conn.execute(
            f"SELECT COUNT(*) FROM resale_prices WHERE {col} IS NULL"
        ).fetchone()[0]
        lines.append(f"   Null {col}: {null_c:,} ({null_c/total*100:.1f}%)")

    # Parquet survival rates
    lines.append(f"\n3. Parquet survival rates vs raw DB ({total:,} rows):")
    for name in ("train", "val", "test"):
        X = pd.read_parquet(os.path.join(run_dir, f"X_{name}.parquet"))
        lines.append(f"   X_{name}: {len(X):,} rows ({len(X)/total*100:.1f}%)")

    # Test set by year
    lines.append(f"\n4. Test set row counts by year (from DB, year >= 2023):")
    rows = conn.execute(
        "SELECT year, COUNT(*) FROM resale_prices "
        "WHERE year >= 2023 GROUP BY year ORDER BY year"
    ).fetchall()
    for yr, cnt in rows:
        lines.append(f"   {yr}: {cnt:,}")

    # Null counts in X_train
    X_train = pd.read_parquet(os.path.join(run_dir, "X_train.parquet"))
    lines.append(f"\n5. Null counts per feature in X_train:")
    nulls = X_train.isnull().sum()
    all_ok = True
    for col, n in nulls.items():
        if n > 0:
            lines.append(f"   {col}: {n:,} nulls!")
            all_ok = False
    if all_ok:
        lines.append("   All features: 0 nulls")

    # Price distribution train vs test
    lines.append(f"\n6. Resale price distribution (actual SGD):")
    y_train = pd.read_parquet(os.path.join(run_dir, "y_train.parquet"))
    y_test  = pd.read_parquet(os.path.join(run_dir, "y_test.parquet"))
    for label, y in [("train", y_train), ("test", y_test)]:
        p = y["resale_price"]
        lines.append(
            f"   {label:5s}: min={p.min():>10,.0f}  median={p.median():>10,.0f}  "
            f"mean={p.mean():>10,.0f}  max={p.max():>10,.0f}"
        )
    drift_ratio = y_test["resale_price"].median() / y_train["resale_price"].median()
    if drift_ratio > 1.3 or drift_ratio < 0.7:
        lines.append(
            f"   *** TEMPORAL DRIFT WARNING: test median is {drift_ratio:.2f}x "
            f"train median. Expect higher test errors."
        )

    # Flat type breakdown
    lines.append(f"\n7. Flat type breakdown (DB train <= 2020 vs test >= 2023):")
    train_ft = pd.read_sql(
        "SELECT flat_type, COUNT(*) as n FROM resale_prices "
        "WHERE year <= 2020 GROUP BY flat_type ORDER BY flat_type", conn
    )
    test_ft = pd.read_sql(
        "SELECT flat_type, COUNT(*) as n FROM resale_prices "
        "WHERE year >= 2023 GROUP BY flat_type ORDER BY flat_type", conn
    )
    merged = train_ft.merge(
        test_ft, on="flat_type", how="outer", suffixes=("_train", "_test")
    ).fillna(0)
    for _, row in merged.iterrows():
        lines.append(
            f"   {row['flat_type']:20s}  "
            f"train: {int(row['n_train']):>8,}  test: {int(row['n_test']):>8,}"
        )

    conn.close()
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model training: XGBoost with Optuna
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_val_sgd: np.ndarray,
    y_val_pi: np.ndarray,
    run_dir: str,
) -> tuple:
    """
    Train XGBoost with Optuna hyperparameter tuning (100 trials).
    Early stopping on val set prevents overfitting.

    Parameters:
        X_train:   Training features.
        y_train:   Training target (RPI-adjusted log1p scale).
        X_val:     Validation features.
        y_val:     Validation target (RPI-adjusted log1p scale).
        y_val_sgd: Validation actual prices in SGD (for RMSE optimisation).
        y_val_pi:  Validation price index values (for denormalization).
        run_dir:   Directory to save the Optuna study.

    Returns:
        Tuple of (trained model, best params dict).
    """
    print("\n  Tuning XGBoost with Optuna ...")

    def objective(trial):
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "random_state": SEED,
            "tree_method": "hist",
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred_log = model.predict(X_val)
        pred_sgd = _inv_transform(pred_log, y_val_pi)
        rmse = float(np.sqrt(mean_squared_error(y_val_sgd, pred_sgd)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=XGB_TRIALS, show_progress_bar=True)

    # Save study
    with open(os.path.join(run_dir, "optuna_study_xgb.pkl"), "wb") as f:
        pickle.dump(study, f)

    # Retrain best model
    best = study.best_params
    best.update({
        "n_estimators": 5000,
        "random_state": SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    })
    print(f"  Best val RMSE: SGD {study.best_value:,.0f}")

    model = xgb.XGBRegressor(**best)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model, best


# ---------------------------------------------------------------------------
# Model training: LightGBM with Optuna
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_val_sgd: np.ndarray,
    y_val_pi: np.ndarray,
    run_dir: str,
) -> tuple:
    """
    Train LightGBM with Optuna hyperparameter tuning (100 trials).
    Early stopping on val set prevents overfitting.

    Parameters:
        X_train:   Training features.
        y_train:   Training target (RPI-adjusted log1p scale).
        X_val:     Validation features.
        y_val:     Validation target (RPI-adjusted log1p scale).
        y_val_sgd: Validation actual prices in SGD (for RMSE optimisation).
        y_val_pi:  Validation price index values (for denormalization).
        run_dir:   Directory to save the Optuna study.

    Returns:
        Tuple of (trained model, best params dict).
    """
    print("\n  Tuning LightGBM with Optuna ...")

    def objective(trial):
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": SEED,
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        pred_log = model.predict(X_val)
        pred_sgd = _inv_transform(pred_log, y_val_pi)
        rmse = float(np.sqrt(mean_squared_error(y_val_sgd, pred_sgd)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=LGBM_TRIALS, show_progress_bar=True)

    # Save study
    with open(os.path.join(run_dir, "optuna_study_lgbm.pkl"), "wb") as f:
        pickle.dump(study, f)

    # Retrain best model
    best = study.best_params
    best.update({
        "n_estimators": 5000,
        "random_state": SEED,
        "verbose": -1,
    })
    print(f"  Best val RMSE: SGD {study.best_value:,.0f}")

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    model = lgb.LGBMRegressor(**best)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    return model, best


# ---------------------------------------------------------------------------
# Model training: Random Forest (baseline)
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    """
    Train a Random Forest regressor with fixed hyperparameters as a baseline.

    Parameters:
        X_train: Training features.
        y_train: Training target (log1p scale).

    Returns:
        Trained RandomForestRegressor.
    """
    print("\n  Training Random Forest (baseline) ...")
    model = RandomForestRegressor(
        n_estimators=N_RF_ESTIMATORS,
        max_depth=20,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Stacked ensemble
# ---------------------------------------------------------------------------

def train_ensemble(
    models: dict,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Ridge:
    """
    Build a stacked ensemble: XGBoost + LightGBM + Random Forest as base
    learners, with Ridge regression as the meta-learner trained on val set
    predictions.

    Parameters:
        models:  Dict of {name: trained_model} for base learners.
        X_val:   Validation features for generating meta-features.
        y_val:   Validation target (log1p scale) for fitting meta-learner.
        X_train: Training features (unused, kept for interface consistency).
        y_train: Training target (unused, kept for interface consistency).

    Returns:
        Trained Ridge meta-learner.
    """
    print("\n  Building stacked ensemble (Ridge meta-learner on val predictions) ...")

    # Generate meta-features from val predictions
    meta_features = np.column_stack([
        models[name].predict(X_val) for name in sorted(models.keys())
    ])

    meta_model = Ridge(alpha=1.0, random_state=SEED)
    meta_model.fit(meta_features, y_val)

    # Store base learner order for consistent prediction
    meta_model.base_learner_order_ = sorted(models.keys())
    return meta_model


def _ensemble_predict(
    models: dict,
    meta_model: Ridge,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate ensemble predictions by stacking base learner outputs and
    passing them through the Ridge meta-learner.

    Parameters:
        models:     Dict of {name: trained_model} for base learners.
        meta_model: Trained Ridge meta-learner.
        X:          Features to predict on.

    Returns:
        Predicted log1p(resale_price) values.
    """
    meta_features = np.column_stack([
        models[name].predict(X) for name in meta_model.base_learner_order_
    ])
    return meta_model.predict(meta_features)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    models: dict,
    meta_model: Ridge,
    X_val: pd.DataFrame,
    y_val_sgd: np.ndarray,
    y_val_log: np.ndarray,
    y_val_pi: np.ndarray,
    X_test: pd.DataFrame,
    y_test_sgd: np.ndarray,
    y_test_log: np.ndarray,
    y_test_pi: np.ndarray,
    flat_type_val: pd.Series,
    flat_type_test: pd.Series,
) -> dict:
    """
    Evaluate all models on val and test sets. Returns a results dict
    containing metrics for every model on both splits, plus flat_type
    error breakdowns.

    Parameters:
        models:         Dict of {name: trained_model} for base learners.
        meta_model:     Trained Ridge meta-learner.
        X_val:          Validation features.
        y_val_sgd:      Validation actual prices (SGD).
        y_val_log:      Validation RPI-adjusted log1p prices.
        y_val_pi:       Validation price index values.
        X_test:         Test features.
        y_test_sgd:     Test actual prices (SGD).
        y_test_log:     Test RPI-adjusted log1p prices.
        y_test_pi:      Test price index values.
        flat_type_val:  Flat type labels for val set.
        flat_type_test: Flat type labels for test set.

    Returns:
        Dict with structure {model_name: {val: metrics, test: metrics, ...}}.
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    all_results = {}
    all_names = list(models.keys()) + ["ensemble"]

    for name in all_names:
        results = {}

        for split_name, X, y_sgd, y_log, pi, ft_series in [
            ("val",  X_val,  y_val_sgd,  y_val_log,  y_val_pi,  flat_type_val),
            ("test", X_test, y_test_sgd, y_test_log, y_test_pi, flat_type_test),
        ]:
            if name == "ensemble":
                pred_log = _ensemble_predict(models, meta_model, X)
            else:
                pred_log = models[name].predict(X)

            pred_sgd = _inv_transform(pred_log, pi)
            metrics = _compute_metrics(y_sgd, pred_sgd)
            ft_breakdown = _error_by_flat_type(y_sgd, pred_sgd, ft_series)
            results[split_name] = {**metrics, "flat_type_breakdown": ft_breakdown}

        all_results[name] = results

        # Print summary
        val_m  = results["val"]
        test_m = results["test"]
        print(f"\n  {name.upper()}")
        print(f"    Val:  {_fmt_metrics(val_m)}")
        print(f"    Test: {_fmt_metrics(test_m)}")

        # Temporal drift warning
        if test_m["rmse"] > val_m["rmse"] * 1.2:
            print(
                f"    *** DRIFT WARNING: test RMSE is {test_m['rmse']/val_m['rmse']:.2f}x "
                f"val RMSE — likely temporal distribution shift."
            )

    return all_results


# ---------------------------------------------------------------------------
# Determine winner
# ---------------------------------------------------------------------------

def determine_winner(results: dict) -> dict:
    """
    Compare all models on test RMSE and declare a winner.

    Parameters:
        results: Full evaluation results dict from evaluate_all().

    Returns:
        Dict with winner name, test metrics, and justification string.
    """
    best_name = None
    best_rmse = float("inf")

    for name, res in results.items():
        if res["test"]["rmse"] < best_rmse:
            best_rmse = res["test"]["rmse"]
            best_name = name

    justification = (
        f"{best_name} achieved the lowest test RMSE of SGD {best_rmse:,.0f} "
        f"(R2={results[best_name]['test']['r2']:.4f}, "
        f"MAPE={results[best_name]['test']['mape']:.2f}%). "
    )

    # Check runner-up
    runner_up = sorted(
        [(n, r["test"]["rmse"]) for n, r in results.items() if n != best_name],
        key=lambda x: x[1],
    )[0]
    justification += (
        f"Runner-up: {runner_up[0]} with RMSE SGD {runner_up[1]:,.0f}."
    )

    return {
        "winner": best_name,
        "test_rmse": round(best_rmse, 2),
        "test_r2": round(results[best_name]["test"]["r2"], 6),
        "test_mape": round(results[best_name]["test"]["mape"], 4),
        "justification": justification,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    run_dir: str,
    models: dict,
    meta_model: Ridge,
    results: dict,
    winner: dict,
    xgb_params: dict,
    lgbm_params: dict,
) -> None:
    """
    Save all training artefacts to the run directory.

    Parameters:
        run_dir:           Feature engineering run directory path.
        models:            Dict of trained base learner models.
        meta_model:        Trained Ridge meta-learner.
        results:           Full evaluation results dict.
        winner:            Winner declaration dict.
        xgb_params:        Best XGBoost hyperparameters.
        lgbm_params:       Best LightGBM hyperparameters.
    """
    print(f"\n  Saving outputs to '{run_dir}/' ...")

    # Models
    for name, model in models.items():
        path = os.path.join(run_dir, f"{name}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"    {name}_model.pkl")

    with open(os.path.join(run_dir, "ensemble_model.pkl"), "wb") as f:
        pickle.dump({"base_models": {n: None for n in models}, "meta_model": meta_model}, f)
    print("    ensemble_model.pkl")

    # Metrics — update existing metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Convert flat_type_breakdown values for JSON serialization
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {}
        for split_name, split_metrics in model_results.items():
            serializable_results[model_name][split_name] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in split_metrics.items()
            }

    metrics["model_results"] = serializable_results
    metrics["winner"] = winner
    metrics["xgb_best_params"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in xgb_params.items()}
    metrics["lgbm_best_params"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in lgbm_params.items()}

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("    metrics.json (updated)")

    # Model comparison
    comparison_path = os.path.join(run_dir, "model_comparison.json")
    comparison = {
        "winner": winner,
        "all_models": {
            name: {
                "val_rmse": round(res["val"]["rmse"], 2),
                "val_r2": round(res["val"]["r2"], 6),
                "test_rmse": round(res["test"]["rmse"], 2),
                "test_r2": round(res["test"]["r2"], 6),
                "test_mape": round(res["test"]["mape"], 4),
            }
            for name, res in results.items()
        },
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print("    model_comparison.json")

    # Training report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("HDB RESALE PRICE PREDICTION — TRAINING REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("MODEL RESULTS")
    report_lines.append("=" * 70)

    for name in results:
        report_lines.append(f"\n--- {name.upper()} ---")
        for split in ("val", "test"):
            m = results[name][split]
            report_lines.append(f"  {split.upper()}: {_fmt_metrics(m)}")
            report_lines.append(f"  {split.upper()} flat_type breakdown:")
            for ft, info in sorted(m["flat_type_breakdown"].items()):
                report_lines.append(
                    f"    {ft:20s}  MAE=SGD {info['mae']:>10,.0f}  (n={info['count']:,})"
                )

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("WINNER")
    report_lines.append("=" * 70)
    report_lines.append(f"  {winner['justification']}")
    report_lines.append("")

    report_path = os.path.join(run_dir, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("    training_report.txt")


# ---------------------------------------------------------------------------
# Flat type lookup for error breakdown
# ---------------------------------------------------------------------------

def _load_flat_type_series(run_dir: str) -> tuple[pd.Series, pd.Series]:
    """
    Load flat_type labels for val and test sets from the database,
    matching the rows that survived into the parquet splits.

    We use flat_type_ordinal from X_val/X_test and reverse-map it to
    flat_type strings for the error breakdown.

    Parameters:
        run_dir: Path to the feature engineering run directory.

    Returns:
        Tuple of (flat_type_val, flat_type_test) as pd.Series.
    """
    ordinal_to_flat = {v: k for k, v in {
        "1 Room": 1, "2 Room": 2, "3 Room": 3, "4 Room": 4,
        "5 Room": 5, "Executive": 6, "Multi-Generation": 7,
    }.items()}

    results = []
    for name in ("val", "test"):
        X = pd.read_parquet(os.path.join(run_dir, f"X_{name}.parquet"))
        ft_series = X["flat_type_ordinal"].map(ordinal_to_flat).fillna("Unknown")
        results.append(ft_series)

    return results[0], results[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full training pipeline: diagnostic, train 4 models, evaluate,
    determine winner, and save all outputs.
    """
    print("=" * 70)
    print("HDB Resale — Model Training")
    print("=" * 70)

    # Resolve run directory
    run_dir = _resolve_run_dir()
    print(f"  Run directory: {run_dir}")

    # Load data
    print("\n--- LOADING DATA ---")
    X_train, y_train_df, X_val, y_val_df, X_test, y_test_df = _load_splits(run_dir)

    y_train_log = y_train_df["log_price"].values
    y_val_log   = y_val_df["log_price"].values
    y_test_log  = y_test_df["log_price"].values

    y_val_sgd  = y_val_df["resale_price"].values
    y_test_sgd = y_test_df["resale_price"].values

    y_val_pi   = y_val_df["price_index"].values
    y_test_pi  = y_test_df["price_index"].values

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # Load flat type labels for error breakdown
    flat_type_val, flat_type_test = _load_flat_type_series(run_dir)

    # Step 1: Train XGBoost
    print("\n--- STEP 1: XGBOOST ---")
    t0 = time.time()
    xgb_model, xgb_params = train_xgboost(
        X_train, y_train_log, X_val, y_val_log, y_val_sgd, y_val_pi, run_dir,
    )
    print(f"  XGBoost training time: {time.time() - t0:.1f}s")

    # Step 2: Train LightGBM
    print("\n--- STEP 2: LIGHTGBM ---")
    t0 = time.time()
    lgbm_model, lgbm_params = train_lightgbm(
        X_train, y_train_log, X_val, y_val_log, y_val_sgd, y_val_pi, run_dir,
    )
    print(f"  LightGBM training time: {time.time() - t0:.1f}s")

    # Step 3: Train Random Forest
    print("\n--- STEP 3: RANDOM FOREST ---")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train_log)
    print(f"  Random Forest training time: {time.time() - t0:.1f}s")

    # Step 4: Build ensemble
    print("\n--- STEP 4: STACKED ENSEMBLE ---")
    base_models = {"xgboost": xgb_model, "lgbm": lgbm_model, "rf": rf_model}
    meta_model = train_ensemble(
        base_models, X_val, y_val_log, X_train, y_train_log,
    )

    # Step 5: Evaluate all
    print("\n--- STEP 5: EVALUATION ---")
    results = evaluate_all(
        base_models, meta_model,
        X_val, y_val_sgd, y_val_log, y_val_pi,
        X_test, y_test_sgd, y_test_log, y_test_pi,
        flat_type_val, flat_type_test,
    )

    # Step 6: Determine winner
    winner = determine_winner(results)
    print(f"\n  WINNER: {winner['winner'].upper()}")
    print(f"  {winner['justification']}")

    # Step 7: Save outputs
    print("\n--- STEP 7: SAVING OUTPUTS ---")
    save_outputs(
        run_dir, base_models, meta_model, results, winner,
        xgb_params, lgbm_params,
    )

    # Final comparison table
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<12} {'Val RMSE':>12} {'Test RMSE':>12} {'Test R2':>10} {'Test MAPE':>12}")
    print("  " + "-" * 58)
    for name in results:
        v = results[name]["val"]
        t = results[name]["test"]
        marker = " <-- WINNER" if name == winner["winner"] else ""
        print(
            f"  {name:<12} {v['rmse']:>12,.0f} {t['rmse']:>12,.0f} "
            f"{t['r2']:>10.4f} {t['mape']:>11.2f}%{marker}"
        )
    print("=" * 70)

    print(f"\nDone. All outputs saved to: {run_dir}/")


if __name__ == "__main__":
    main()
