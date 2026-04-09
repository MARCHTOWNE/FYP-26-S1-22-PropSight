"""
model_training.py
=================
Single responsibility: train, tune, and evaluate ML models for HDB resale
price prediction using the feature-ready datasets produced by
feature_engineering.py.

Design decisions:
  - Reads pre-split parquets from the run directory pointed to by
    model_assets/latest.txt. No feature engineering is done here.
  - New feature runs use log1p(resale_price) as the target to avoid
    holdout leakage from a target normalization index computed on the
    transaction table. Legacy runs with price_index are still supported.
  - Four models: XGBoost, LightGBM, CatBoost, Random Forest.
  - Optuna tunes XGBoost, LightGBM, and CatBoost on val RMSE using a
    deterministic training subset for speed, then retrains the best params
    on the full train split. Trial counts and sample sizes are configurable
    via env vars.
  - CatBoost uses cat_features for flat_type_ordinal and is_mature_estate.
  - Random Forest uses a sensible fixed configuration (no tuning).
  - Train set is for fitting; val set is for tuning and early stopping.
  - Winner selection uses validation RMSE across XGBoost, LightGBM, and
    CatBoost, keeping the test set locked for final reporting.
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
import sys
import time
from textwrap import dedent

import numpy as np
import pandas as pd
from training_data_source import get_training_source_summary

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
    import catboost as cb
except ImportError:
    _MISSING.append("catboost")
try:
    import optuna
except ImportError:
    _MISSING.append("optuna")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if _MISSING:
    print(f"Missing required packages: {', '.join(_MISSING)}")
    print(f"Install with:  pip install {' '.join(_MISSING)}")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.environ.get("MODEL_ASSETS_DIR", os.path.join(BASE_DIR, "model_assets"))
LATEST_FILE    = os.path.join(OUTPUT_DIR, "latest.txt")
SEED           = 42


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


XGB_TRIALS     = int(os.environ.get("XGB_TRIALS", "25"))
LGBM_TRIALS    = int(os.environ.get("LGBM_TRIALS", "25"))
CATBOOST_TRIALS = int(os.environ.get("CATBOOST_TRIALS", "25"))
XGB_N_ESTIMATORS = int(os.environ.get("XGB_N_ESTIMATORS", "2000"))
LGBM_N_ESTIMATORS = int(os.environ.get("LGBM_N_ESTIMATORS", "2000"))
CATBOOST_N_ESTIMATORS = int(os.environ.get("CATBOOST_N_ESTIMATORS", "2000"))
EARLY_STOPPING_ROUNDS = int(os.environ.get("GBM_EARLY_STOPPING_ROUNDS", "50"))
TUNING_SAMPLE_SIZE = int(os.environ.get("TUNING_SAMPLE_SIZE", "250000"))
N_RF_ESTIMATORS = int(os.environ.get("RF_ESTIMATORS", "300"))
MODEL_N_JOBS   = int(os.environ.get("MODEL_N_JOBS", "-1"))
FRESH_TUNING   = _env_flag("FRESH_TUNING", False)
DECAY_RATE     = float(os.environ.get("DECAY_RATE", "0.006"))
COMPUTE_SHAP   = _env_flag("COMPUTE_SHAP", False)

DEFAULT_TRAINING_CONFIG = {
    "xgb_trials": 25,
    "lgbm_trials": 25,
    "catboost_trials": 25,
    "xgb_n_estimators": 2000,
    "lgbm_n_estimators": 2000,
    "catboost_n_estimators": 2000,
    "early_stopping_rounds": 50,
    "tuning_sample_size": 250000,
    "rf_estimators": 300,
    "model_n_jobs": -1,
    "fresh_tuning": False,
}

# Flat type labels used for error breakdown
FLAT_TYPES = [
    "1 Room", "2 Room", "3 Room", "4 Room",
    "5 Room", "Executive", "Multi-Generation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _training_config_snapshot() -> dict[str, int | bool]:
    """Return the effective training configuration for this run."""
    return {
        "xgb_trials": XGB_TRIALS,
        "lgbm_trials": LGBM_TRIALS,
        "catboost_trials": CATBOOST_TRIALS,
        "xgb_n_estimators": XGB_N_ESTIMATORS,
        "lgbm_n_estimators": LGBM_N_ESTIMATORS,
        "catboost_n_estimators": CATBOOST_N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "tuning_sample_size": TUNING_SAMPLE_SIZE,
        "rf_estimators": N_RF_ESTIMATORS,
        "model_n_jobs": MODEL_N_JOBS,
        "fresh_tuning": FRESH_TUNING,
    }


def _training_config_notes(config: dict[str, int | bool]) -> list[str]:
    """Return human-readable notes when a run uses unusually fast settings."""
    notes: list[str] = []

    if int(config["xgb_trials"]) < DEFAULT_TRAINING_CONFIG["xgb_trials"]:
        notes.append(
            f"XGBoost Optuna trials reduced to {config['xgb_trials']} "
            f"(default {DEFAULT_TRAINING_CONFIG['xgb_trials']})."
        )
    if int(config["lgbm_trials"]) < DEFAULT_TRAINING_CONFIG["lgbm_trials"]:
        notes.append(
            f"LightGBM Optuna trials reduced to {config['lgbm_trials']} "
            f"(default {DEFAULT_TRAINING_CONFIG['lgbm_trials']})."
        )
    if int(config["catboost_trials"]) < DEFAULT_TRAINING_CONFIG["catboost_trials"]:
        notes.append(
            f"CatBoost Optuna trials reduced to {config['catboost_trials']} "
            f"(default {DEFAULT_TRAINING_CONFIG['catboost_trials']})."
        )
    if int(config["xgb_n_estimators"]) < 100:
        notes.append(
            f"XGBoost n_estimators is only {config['xgb_n_estimators']}; "
            "this is suitable for smoke tests, not production tuning."
        )
    if int(config["lgbm_n_estimators"]) < 100:
        notes.append(
            f"LightGBM n_estimators is only {config['lgbm_n_estimators']}; "
            "this is suitable for smoke tests, not production tuning."
        )
    if int(config["catboost_n_estimators"]) < 100:
        notes.append(
            f"CatBoost iterations is only {config['catboost_n_estimators']}; "
            "this is suitable for smoke tests, not production tuning."
        )
    if int(config["tuning_sample_size"]) < 50000:
        notes.append(
            f"Tuning sample size reduced to {int(config['tuning_sample_size']):,}; "
            "best params may be less stable than a full run."
        )
    if int(config["model_n_jobs"]) == 1:
        notes.append(
            "MODEL_N_JOBS=1; this often indicates a constrained or verification run."
        )

    return notes

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
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(BASE_DIR, run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")
    return run_dir


def _load_splits(run_dir: str) -> tuple:
    """
    Load X and y parquets for the required train/val/test splits and the
    optional future_holdout split when present.

    Parameters:
        run_dir: Path to the feature engineering run directory.

    Returns:
        Dict mapping split name to (X_df, y_df).
    """
    splits = {}
    for name in ("train", "val", "test", "future_holdout"):
        X_path = os.path.join(run_dir, f"X_{name}.parquet")
        y_path = os.path.join(run_dir, f"y_{name}.parquet")
        if not (os.path.exists(X_path) and os.path.exists(y_path)):
            if name in {"train", "val", "test"}:
                raise FileNotFoundError(
                    f"Missing required split files for '{name}' in '{run_dir}'."
                )
            continue
        splits[name] = (pd.read_parquet(X_path), pd.read_parquet(y_path))
    return splits


def _format_year_window(start_year: int | None, end_year: int | None) -> str:
    if start_year is None and end_year is None:
        return "unavailable"
    if start_year == end_year:
        return str(start_year)
    if start_year is None:
        return f"<= {end_year}"
    if end_year is None:
        return f">= {start_year}"
    return f"{start_year}-{end_year}"


def _build_tuning_subset(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    y_train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a training subset for hyperparameter tuning using the most recent rows.

    Tuning on the most-recent data keeps Optuna focused on the current market
    regime while staying responsive. Final best params are retrained on the
    full train split.
    """
    if TUNING_SAMPLE_SIZE <= 0 or len(X_train) <= TUNING_SAMPLE_SIZE:
        print(f"  Tuning subset: using full training set ({len(X_train):,} rows).")
        return X_train, y_train

    if y_train_df is not None and "year_month_raw" in y_train_df.columns:
        sorted_idx = np.argsort(y_train_df["year_month_raw"].values)
        sample_idx = sorted_idx[-TUNING_SAMPLE_SIZE:]
        ym_vals = y_train_df["year_month_raw"].values[sample_idx]
        ym_min, ym_max = int(ym_vals.min()), int(ym_vals.max())
        date_range = (
            f"{ym_min // 100}-{ym_min % 100:02d} – "
            f"{ym_max // 100}-{ym_max % 100:02d}"
        )
        print(
            f"  Tuning subset: {TUNING_SAMPLE_SIZE:,}/{len(X_train):,} rows "
            f"(most recent: {date_range})."
        )
    else:
        rng = np.random.default_rng(SEED)
        sample_idx = np.sort(
            rng.choice(len(X_train), size=TUNING_SAMPLE_SIZE, replace=False)
        )
        print(
            f"  Tuning subset: {TUNING_SAMPLE_SIZE:,}/{len(X_train):,} rows "
            f"(random — year_month_raw not available)."
        )

    X_tune = X_train.iloc[sample_idx].copy()
    y_tune = y_train[sample_idx].copy()
    return X_tune, y_tune


def _inv_transform(
    y_log: np.ndarray,
    price_index: np.ndarray | None = None,
) -> np.ndarray:
    """Convert the model target back to actual SGD."""
    if price_index is None:
        return np.expm1(y_log)
    return np.expm1(y_log) * price_index


class EnsembleModel:
    """Weighted average of two trained regressors, optimised on val MAPE."""

    def __init__(self, models: list, weights: np.ndarray) -> None:
        self.models = models
        self.weights = weights  # shape (n_models,), sums to 1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        return preds @ self.weights

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute weighted SHAP values for the ensemble.

        SHAP values are additive, so blending them by ensemble weight gives
        the correct attribution for the blended prediction.

        Requires `shap` to be installed:  pip install shap
        """
        try:
            import shap as shap_lib
        except ImportError:
            raise ImportError("pip install shap")

        blended = None
        for model, w in zip(self.models, self.weights):
            explainer = shap_lib.TreeExplainer(model)
            sv = explainer.shap_values(X)
            blended = sv * w if blended is None else blended + sv * w
        return blended


def build_ensemble(
    models_dict: dict,
    X_val: pd.DataFrame,
    y_val_sgd: np.ndarray,
    y_val_pi: np.ndarray | None = None,
    eligible: tuple = ("catboost", "lgbm"),
) -> tuple["EnsembleModel | None", dict]:
    """
    Grid-search the optimal blend weight (w) for two models on val MAPE.

    Returns:
        (EnsembleModel, weights_dict) or (None, {}) if fewer than 2 are present.
    """
    present = [n for n in eligible if n in models_dict]
    if len(present) < 2:
        return None, {}

    name_a, name_b = present[0], present[1]
    pred_a = _inv_transform(models_dict[name_a].predict(X_val), y_val_pi)
    pred_b = _inv_transform(models_dict[name_b].predict(X_val), y_val_pi)

    best_mape, best_w = float("inf"), 0.5
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * pred_a + (1.0 - w) * pred_b
        mape = float(np.mean(np.abs((y_val_sgd - blend) / y_val_sgd)) * 100)
        if mape < best_mape:
            best_mape, best_w = mape, float(w)

    weights = np.array([best_w, 1.0 - best_w])
    ensemble = EnsembleModel([models_dict[name_a], models_dict[name_b]], weights)
    print(
        f"\n  Ensemble ({name_a}×{best_w:.2f} + {name_b}×{1-best_w:.2f}) "
        f"→ val MAPE {best_mape:.2f}%"
    )
    return ensemble, {name_a: round(best_w, 4), name_b: round(1.0 - best_w, 4)}


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


def _save_optuna_study(
    study: "optuna.study.Study",
    study_path: str,
    meta: dict | None = None,
) -> None:
    """Persist an Optuna study snapshot to disk, with optional sidecar meta JSON."""
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    if meta is not None:
        meta_path = study_path.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def _check_study_meta_invalidation(saved: dict, current: dict) -> str | None:
    """Return a reason string if the cached study should be discarded, else None."""
    if saved.get("split_strategy") != current.get("split_strategy"):
        return (
            f"split_strategy changed "
            f"({saved.get('split_strategy')} → {current.get('split_strategy')})"
        )
    if saved.get("feature_cols") != current.get("feature_cols"):
        return "feature_cols list changed"
    if saved.get("optuna_objective") != current.get("optuna_objective"):
        return (
            f"optuna_objective changed "
            f"({saved.get('optuna_objective')} → {current.get('optuna_objective')})"
        )
    saved_n = saved.get("n_train_rows", 0)
    current_n = current.get("n_train_rows", 0)
    if saved_n > 0 and abs(current_n - saved_n) / saved_n > 0.05:
        return f"n_train_rows changed >5% ({saved_n:,} → {current_n:,})"
    return None


def _load_or_create_study(
    study_path: str,
    label: str,
    total_trials: int,
    current_meta: dict | None = None,
) -> tuple["optuna.study.Study", bool]:
    """
    Load a previously saved Optuna study when available, otherwise create one.

    If current_meta is provided and a sidecar _meta.json exists, the saved
    metadata is compared against current_meta. Mismatches in split_strategy,
    feature_cols, or n_train_rows (>5% change) trigger auto-discard so stale
    hyperparameters are not reused.  FRESH_TUNING=1 is still a manual override.

    Returns:
        Tuple of (study, resumed_existing_study).
    """
    if FRESH_TUNING:
        if os.path.exists(study_path):
            print(
                f"  {label}: FRESH_TUNING=1, ignoring existing "
                f"{os.path.basename(study_path)} and starting a new {total_trials}-trial search."
            )
        else:
            print(
                f"  {label}: FRESH_TUNING=1, starting a new {total_trials}-trial search."
            )
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        return study, False

    if os.path.exists(study_path):
        # Check sidecar meta for auto-invalidation before loading the study.
        if current_meta is not None:
            meta_path = study_path.replace(".pkl", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    saved_meta = json.load(f)
                reason = _check_study_meta_invalidation(saved_meta, current_meta)
                if reason:
                    print(
                        f"  {label}: discarding cached study — {reason}. "
                        f"Starting a new {total_trials}-trial search."
                    )
                    study = optuna.create_study(
                        direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=SEED),
                    )
                    return study, False

        with open(study_path, "rb") as f:
            study = pickle.load(f)
        completed = len(
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        )
        print(
            f"  {label}: resuming {os.path.basename(study_path)} with "
            f"{len(study.trials):,}/{total_trials:,} saved trials "
            f"({completed:,} complete)."
        )
        return study, True

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    print(f"  {label}: starting a new {total_trials}-trial Optuna search.")
    return study, False


def _best_completed_trial(study: "optuna.study.Study") -> "optuna.trial.FrozenTrial":
    """Return the best completed trial or raise when none succeeded."""
    complete_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise ValueError("No completed Optuna trials available.")
    return min(complete_trials, key=lambda trial: float(trial.value))


def _optimize_study(
    study: "optuna.study.Study",
    objective,
    total_trials: int,
    study_path: str,
    label: str,
    meta: dict | None = None,
) -> tuple["optuna.trial.FrozenTrial", bool]:
    """
    Run or resume Optuna optimization, saving progress on success or interrupt.

    Returns:
        Tuple of (best_completed_trial, interrupted_flag).
    """
    completed_before = len(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    )
    existing_trials = len(study.trials)
    remaining_trials = max(total_trials - existing_trials, 0)
    if remaining_trials == 0:
        print(
            f"  {label}: skipping search because {existing_trials:,} saved trials "
            f"already meet or exceed the requested {total_trials:,}."
        )
        best_trial = _best_completed_trial(study)
        _save_optuna_study(study, study_path, meta=meta)
        return best_trial, False

    print(
        f"  {label}: running {remaining_trials:,} more trial(s) "
        f"to reach {total_trials:,} total."
    )

    interrupted = False
    try:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\n  {label} tuning interrupted after {len(study.trials):,} trials. "
            "Saving partial study and using the best completed trial so far."
        )
    finally:
        _save_optuna_study(study, study_path, meta=meta)

    completed_after = len(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    )
    new_completed = completed_after - completed_before
    print(
        f"  {label} study snapshot saved to {os.path.basename(study_path)} "
        f"({completed_after:,} complete, {new_completed:+,} new complete trials)."
    )
    best_trial = _best_completed_trial(study)
    return best_trial, interrupted


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
    Query the canonical training source and the parquet splits to produce
    a data quality diagnostic report. Returns the report as a string.

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

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest: dict[str, object] = {}
    split_years: dict[str, int | None] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        split_years = manifest.get("split_years", {}) or {}

    test_start_year = split_years.get("test_start_year")
    test_end_year = split_years.get("test_end_year")
    train_end_year = split_years.get("train_end_year")
    future_holdout_start_year = split_years.get("future_holdout_start_year")
    source_summary = get_training_source_summary(
        test_start_year=int(test_start_year) if test_start_year is not None else None,
        test_end_year=int(test_end_year) if test_end_year is not None else None,
        train_end_year=int(train_end_year) if train_end_year is not None else None,
    )
    total = int(source_summary.get("total_rows", 0))
    lines.append(f"\n1. Total rows in {source_summary['source']}: {total:,}")
    lines.append(f"   Source used for feature engineering: {manifest.get('data_source', source_summary['source'])}")

    # Geocoding coverage
    geocoded_rows = int(source_summary.get("geocoded_rows", 0))
    null_lat = max(total - geocoded_rows, 0)
    geocode_pct = (geocoded_rows / total * 100) if total else 0.0
    lines.append(f"\n2. Geocoding coverage:")
    null_lat_pct = (null_lat / total * 100) if total else 0.0
    lines.append(f"   Null latitude/longitude: {null_lat:,} ({null_lat_pct:.1f}%)")
    lines.append(f"   Geocoded:                {geocoded_rows:,} ({geocode_pct:.1f}%)")
    if geocode_pct < 80:
        lines.append(
            "   *** WARNING: Geocoding coverage below 80%! "
            "Proximity features will have limited signal."
        )
        lines.append("   *** Recommend rerunning geocoding.py first.")

    # Null proximity features
    proximity_rows = int(source_summary.get("proximity_rows", 0))
    null_proximity = max(total - proximity_rows, 0)
    for col in ["dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"]:
        pct_null = (null_proximity / total * 100) if total else 0.0
        lines.append(f"   Null {col}: {null_proximity:,} ({pct_null:.1f}%)")

    # Parquet survival rates
    lines.append(f"\n3. Parquet survival rates vs source ({total:,} rows):")
    for name in ("train", "val", "test", "future_holdout"):
        X_path = os.path.join(run_dir, f"X_{name}.parquet")
        if not os.path.exists(X_path):
            continue
        X = pd.read_parquet(X_path)
        pct = (len(X) / total * 100) if total else 0.0
        lines.append(f"   X_{name}: {len(X):,} rows ({pct:.1f}%)")

    # Test set by year
    if test_start_year is not None:
        test_window = _format_year_window(
            int(test_start_year),
            int(test_end_year) if test_end_year is not None else None,
        )
        lines.append(
            f"\n4. Test set row counts by year (from source, full-year test window {test_window}):"
        )
        for row in source_summary.get("test_rows_by_year", []):
            lines.append(f"   {row['year']}: {row['count']:,}")
        if future_holdout_start_year is not None:
            lines.append(f"   Future holdout starts at: {int(future_holdout_start_year)}")
    else:
        lines.append("\n4. Test set row counts by year: unavailable")

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
    distributions = [("train", y_train), ("test", y_test)]
    future_holdout_path = os.path.join(run_dir, "y_future_holdout.parquet")
    if os.path.exists(future_holdout_path):
        y_future_holdout = pd.read_parquet(future_holdout_path)
        distributions.append(("future_holdout", y_future_holdout))
    for label, y in distributions:
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
    if train_end_year is not None and test_start_year is not None:
        test_window = _format_year_window(
            int(test_start_year),
            int(test_end_year) if test_end_year is not None else None,
        )
        lines.append(
            f"\n7. Flat type breakdown (source train <= {train_end_year} vs test {test_window}):"
        )
        for row in source_summary.get("flat_type_breakdown", []):
            lines.append(
                f"   {row['flat_type']:20s}  "
                f"train: {int(row['n_train']):>8,}  test: {int(row['n_test']):>8,}"
            )
    else:
        lines.append("\n7. Flat type breakdown: unavailable")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model training: XGBoost with Optuna
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_val_sgd: np.ndarray,
    y_val_pi: np.ndarray | None,
    run_dir: str,
    sample_weight: np.ndarray | None = None,
    current_meta: dict | None = None,
) -> tuple:
    """
    Train XGBoost with Optuna hyperparameter tuning.
    Early stopping on val set prevents overfitting.

    Parameters:
        X_train:       Full training features used for final refit.
        y_train:       Full training target on the model log scale.
        X_tune:        Sampled training features used during Optuna search.
        y_tune:        Sampled training target used during Optuna search.
        X_val:         Validation features.
        y_val:         Validation target on the model log scale.
        y_val_sgd:     Validation actual prices in SGD (for RMSE optimisation).
        y_val_pi:      Optional validation price index values for legacy runs.
        run_dir:       Directory to save the Optuna study.
        sample_weight: Optional per-row weights for the final refit (time-decay).
        current_meta:  Study invalidation metadata for auto-discard logic.

    Returns:
        Tuple of (trained model, best params dict).
    """
    print("\n  Tuning XGBoost with Optuna ...")
    study_path = os.path.join(OUTPUT_DIR, "optuna_study_xgb.pkl")

    def objective(trial):
        params = {
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 3.0, log=True),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
            "n_jobs": MODEL_N_JOBS,
            "random_state": SEED,
            "tree_method": "hist",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tune, y_tune,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred_log = model.predict(X_val)
        pred_sgd = _inv_transform(pred_log, y_val_pi)
        rmse = float(np.sqrt(mean_squared_error(y_val_sgd, pred_sgd)))
        return rmse

    study, _ = _load_or_create_study(study_path, "XGBoost", XGB_TRIALS, current_meta=current_meta)
    best_trial, interrupted = _optimize_study(
        study,
        objective,
        total_trials=XGB_TRIALS,
        study_path=study_path,
        label="XGBoost",
        meta=current_meta,
    )

    # Retrain best model
    best = dict(best_trial.params)
    best.update({
        "n_estimators": XGB_N_ESTIMATORS,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "n_jobs": MODEL_N_JOBS,
        "random_state": SEED,
        "tree_method": "hist",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    })
    if interrupted:
        print(
            f"  Continuing with best completed XGBoost trial at val RMSE SGD "
            f"{float(best_trial.value):,.0f}."
        )
    else:
        print(f"  Best val RMSE: SGD {float(best_trial.value):,.0f}")

    model = xgb.XGBRegressor(**best)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        sample_weight=sample_weight,
    )
    return model, best


# ---------------------------------------------------------------------------
# Model training: LightGBM with Optuna
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_val_sgd: np.ndarray,
    y_val_pi: np.ndarray | None,
    run_dir: str,
    sample_weight: np.ndarray | None = None,
    current_meta: dict | None = None,
) -> tuple:
    """
    Train LightGBM with Optuna hyperparameter tuning.
    Early stopping on val set prevents overfitting.

    Parameters:
        X_train:       Full training features used for final refit.
        y_train:       Full training target on the model log scale.
        X_tune:        Sampled training features used during Optuna search.
        y_tune:        Sampled training target used during Optuna search.
        X_val:         Validation features.
        y_val:         Validation target on the model log scale.
        y_val_sgd:     Validation actual prices in SGD (for RMSE optimisation).
        y_val_pi:      Optional validation price index values for legacy runs.
        run_dir:       Directory to save the Optuna study.
        sample_weight: Optional per-row weights for the final refit (time-decay).
        current_meta:  Study invalidation metadata for auto-discard logic.

    Returns:
        Tuple of (trained model, best params dict).
    """
    print("\n  Tuning LightGBM with Optuna ...")
    study_path = os.path.join(OUTPUT_DIR, "optuna_study_lgbm.pkl")

    def objective(trial):
        params = {
            "n_estimators": LGBM_N_ESTIMATORS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 160),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample_freq": 1,
            "force_col_wise": True,
            "n_jobs": MODEL_N_JOBS,
            "random_state": SEED,
            "verbose": -1,
        }
        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(-1),
        ]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tune, y_tune,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        pred_log = model.predict(X_val)
        pred_sgd = _inv_transform(pred_log, y_val_pi)
        rmse = float(np.sqrt(mean_squared_error(y_val_sgd, pred_sgd)))
        return rmse

    study, _ = _load_or_create_study(study_path, "LightGBM", LGBM_TRIALS, current_meta=current_meta)
    best_trial, interrupted = _optimize_study(
        study,
        objective,
        total_trials=LGBM_TRIALS,
        study_path=study_path,
        label="LightGBM",
        meta=current_meta,
    )

    # Retrain best model
    best = dict(best_trial.params)
    best.update({
        "n_estimators": LGBM_N_ESTIMATORS,
        "subsample_freq": 1,
        "force_col_wise": True,
        "n_jobs": MODEL_N_JOBS,
        "random_state": SEED,
        "verbose": -1,
    })
    if interrupted:
        print(
            f"  Continuing with best completed LightGBM trial at val RMSE SGD "
            f"{float(best_trial.value):,.0f}."
        )
    else:
        print(f"  Best val RMSE: SGD {float(best_trial.value):,.0f}")

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(-1),
    ]
    model = lgb.LGBMRegressor(**best)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
        sample_weight=sample_weight,
    )
    return model, best


# ---------------------------------------------------------------------------
# Model training: CatBoost with Optuna
# ---------------------------------------------------------------------------

def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_val_sgd: np.ndarray,
    y_val_pi: np.ndarray | None,
    run_dir: str,
    sample_weight: np.ndarray | None = None,
    current_meta: dict | None = None,
) -> tuple:
    """
    Train CatBoost with Optuna hyperparameter tuning.
    Early stopping on val set prevents overfitting.
    flat_type_ordinal and is_mature_estate are passed as cat_features.

    Parameters:
        X_train:       Full training features used for final refit.
        y_train:       Full training target on the model log scale.
        X_tune:        Sampled training features used during Optuna search.
        y_tune:        Sampled training target used during Optuna search.
        X_val:         Validation features.
        y_val:         Validation target on the model log scale.
        y_val_sgd:     Validation actual prices in SGD (for RMSE optimisation).
        y_val_pi:      Optional validation price index values for legacy runs.
        run_dir:       Directory to save the Optuna study.
        sample_weight: Optional per-row weights for the final refit (time-decay).
        current_meta:  Study invalidation metadata for auto-discard logic.

    Returns:
        Tuple of (trained model, best params dict).
    """
    print("\n  Tuning CatBoost with Optuna ...")
    study_path = os.path.join(OUTPUT_DIR, "optuna_study_catboost.pkl")

    cat_features = [
        i for i, col in enumerate(X_train.columns)
        if col in {"flat_type_ordinal", "is_mature_estate"}
    ]

    def objective(trial):
        params = {
            "iterations": CATBOOST_N_ESTIMATORS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "cat_features": cat_features,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "allow_writing_files": False,
            "random_seed": SEED,
            "verbose": 0,
        }
        model = cb.CatBoostRegressor(**params)
        model.fit(X_tune, y_tune, eval_set=(X_val, y_val))
        pred_log = model.predict(X_val)
        pred_sgd = _inv_transform(pred_log, y_val_pi)
        rmse = float(np.sqrt(mean_squared_error(y_val_sgd, pred_sgd)))
        return rmse

    study, _ = _load_or_create_study(study_path, "CatBoost", CATBOOST_TRIALS, current_meta=current_meta)
    best_trial, interrupted = _optimize_study(
        study,
        objective,
        total_trials=CATBOOST_TRIALS,
        study_path=study_path,
        label="CatBoost",
        meta=current_meta,
    )

    # Retrain best model
    best = dict(best_trial.params)
    best.update({
        "iterations": CATBOOST_N_ESTIMATORS,
        "cat_features": cat_features,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "allow_writing_files": False,
        "random_seed": SEED,
        "verbose": 0,
    })
    if interrupted:
        print(
            f"  Continuing with best completed CatBoost trial at val RMSE SGD "
            f"{float(best_trial.value):,.0f}."
        )
    else:
        print(f"  Best val RMSE: SGD {float(best_trial.value):,.0f}")

    model = cb.CatBoostRegressor(**best)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        sample_weight=sample_weight,
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
        n_jobs=MODEL_N_JOBS,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    models: dict,
    evaluation_splits: dict[str, dict[str, object]],
) -> dict:
    """
    Evaluate all models on val/test and any optional future holdout.
    Returns a results dict containing metrics for every model on each split,
    plus flat_type error breakdowns.

    Parameters:
        models:            Dict of {name: trained_model}.
        evaluation_splits: Dict keyed by split name with X, y_sgd, optional
            price_index, and flat_type_series values.

    Returns:
        Dict with structure {model_name: {split_name: metrics, ...}}.
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    all_results = {}

    for name in models:
        results = {}

        for split_name, split_data in evaluation_splits.items():
            X = split_data["X"]
            y_sgd = split_data["y_sgd"]
            pi = split_data["price_index"]
            ft_series = split_data["flat_type_series"]
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
        if "future_holdout" in results:
            print(f"    Future holdout: {_fmt_metrics(results['future_holdout'])}")

        # Temporal drift warning
        if test_m["rmse"] > val_m["rmse"] * 1.2:
            print(
                f"    *** DRIFT WARNING: test RMSE is {test_m['rmse']/val_m['rmse']:.2f}x "
                f"val RMSE — likely temporal distribution shift."
            )
        if "future_holdout" in results:
            future_m = results["future_holdout"]
            if future_m["rmse"] > test_m["rmse"] * 1.1:
                print(
                    f"    *** CURRENT-MARKET WARNING: future holdout RMSE is "
                    f"{future_m['rmse']/test_m['rmse']:.2f}x test RMSE."
                )

    return all_results


# ---------------------------------------------------------------------------
# Determine winner
# ---------------------------------------------------------------------------

def determine_winner(results: dict) -> dict:
    """
    Select the production-safe winner from the base models only.

    Parameters:
        results: Full evaluation results dict from evaluate_all().

    Returns:
        Dict with winner name, selection metrics, and test metrics.
    """
    eligible_models = [
        name for name in ("xgboost", "lgbm", "catboost", "ensemble")
        if name in results
    ]
    if not eligible_models:
        raise ValueError("No tuned model found in results for winner selection.")

    ranked = sorted(
        ((name, results[name]["val"]["mape"]) for name in eligible_models),
        key=lambda item: item[1],
    )
    best_name, best_val_mape = ranked[0]
    best_test = results[best_name]["test"]

    justification = (
        f"{best_name} achieved the lowest validation MAPE of {best_val_mape:.2f}%. "
        f"Its locked test metrics are RMSE={best_test['rmse']:,.0f}, "
        f"R2={best_test['r2']:.4f}, "
        f"MAPE={best_test['mape']:.2f}%. "
    )
    if len(ranked) > 1:
        runner_up_name, runner_up_mape = ranked[1]
        justification += (
            f"Runner-up on validation: {runner_up_name} at {runner_up_mape:.2f}% MAPE. "
        )
    if "future_holdout" in results[best_name]:
        future_metrics = results[best_name]["future_holdout"]
        justification += (
            f"Future holdout metrics: RMSE={future_metrics['rmse']:,.0f}, "
            f"R2={future_metrics['r2']:.4f}, "
            f"MAPE={future_metrics['mape']:.2f}%."
        )

    return {
        "winner": best_name,
        "selection_metric": "val_mape",
        "val_mape": round(best_val_mape, 4),
        "val_rmse": round(results[best_name]["val"]["rmse"], 2),
        "test_rmse": round(best_test["rmse"], 2),
        "test_r2": round(best_test["r2"], 6),
        "test_mape": round(best_test["mape"], 4),
        "justification": justification,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    run_dir: str,
    models: dict,
    results: dict,
    winner: dict,
    xgb_params: dict,
    lgbm_params: dict,
    catboost_params: dict | None = None,
    ensemble_weights: dict | None = None,
) -> None:
    """
    Save all training artefacts to the run directory.

    Parameters:
        run_dir:          Feature engineering run directory path.
        models:           Dict of trained base models.
        results:          Full evaluation results dict.
        winner:           Winner declaration dict.
        xgb_params:       Best XGBoost hyperparameters.
        lgbm_params:      Best LightGBM hyperparameters.
        catboost_params:  Best CatBoost hyperparameters (optional).
    """
    print(f"\n  Saving outputs to '{run_dir}/' ...")
    training_config = _training_config_snapshot()
    training_config_notes = _training_config_notes(training_config)

    # Base models (ensemble stores references to base models — pickle the whole object)
    for name, model in models.items():
        path = os.path.join(run_dir, f"{name}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"    {name}_model.pkl")

    # Metrics — update existing metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    metrics.pop("note", None)

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
    metrics["training_config"] = training_config
    metrics["training_config_notes"] = training_config_notes
    metrics["xgb_best_params"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in xgb_params.items()}
    metrics["lgbm_best_params"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in lgbm_params.items()}
    if catboost_params is not None:
        metrics["catboost_best_params"] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in catboost_params.items()
            if k != "cat_features"  # list of ints, not JSON-unsafe but noisy
        }

    if ensemble_weights:
        metrics["ensemble_weights"] = ensemble_weights

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("    metrics.json (updated)")

    # Model comparison
    comparison_path = os.path.join(run_dir, "model_comparison.json")
    comparison = {
        "winner": winner,
        "training_config": training_config,
        "training_config_notes": training_config_notes,
        "all_models": {
            name: {
                metric_name: metric_value
                for split_name, split_metrics in res.items()
                for metric_name, metric_value in {
                    f"{split_name}_rmse": round(split_metrics["rmse"], 2),
                    f"{split_name}_r2": round(split_metrics["r2"], 6),
                    f"{split_name}_mape": round(split_metrics["mape"], 4),
                }.items()
            }
            for name, res in results.items()
        },
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print("    model_comparison.json")

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["training_config"] = training_config
        manifest["training_config_notes"] = training_config_notes
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Training report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("HDB RESALE PRICE PREDICTION — TRAINING REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("TRAINING CONFIG")
    report_lines.append("=" * 70)
    report_lines.append(
        "  "
        + ", ".join(
            [
                f"XGB_TRIALS={training_config['xgb_trials']}",
                f"LGBM_TRIALS={training_config['lgbm_trials']}",
                f"CATBOOST_TRIALS={training_config['catboost_trials']}",
                f"XGB_N_ESTIMATORS={training_config['xgb_n_estimators']}",
                f"LGBM_N_ESTIMATORS={training_config['lgbm_n_estimators']}",
                f"CATBOOST_N_ESTIMATORS={training_config['catboost_n_estimators']}",
                f"TUNING_SAMPLE_SIZE={int(training_config['tuning_sample_size']):,}",
                f"RF_ESTIMATORS={training_config['rf_estimators']}",
                f"MODEL_N_JOBS={training_config['model_n_jobs']}",
                f"FRESH_TUNING={training_config['fresh_tuning']}",
            ]
        )
    )
    if training_config_notes:
        report_lines.append("  Notes:")
        for note in training_config_notes:
            report_lines.append(f"    - {note}")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("MODEL RESULTS")
    report_lines.append("=" * 70)

    for name in results:
        report_lines.append(f"\n--- {name.upper()} ---")
        for split in ("val", "test", "future_holdout"):
            if split not in results[name]:
                continue
            m = results[name][split]
            split_label = split.replace("_", " ").upper()
            report_lines.append(f"  {split_label}: {_fmt_metrics(m)}")
            report_lines.append(f"  {split_label} flat_type breakdown:")
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

    # Optional SHAP values for winner model on test set (set COMPUTE_SHAP=1)
    if COMPUTE_SHAP:
        winner_name = winner.get("winner", "")
        winner_model = models.get(winner_name)
        if winner_model is not None:
            try:
                import shap as shap_lib

                X_test_path = os.path.join(run_dir, "X_test.parquet")
                X_test_shap = pd.read_parquet(X_test_path)

                if isinstance(winner_model, EnsembleModel):
                    sv = winner_model.shap_values(X_test_shap)
                else:
                    explainer = shap_lib.TreeExplainer(winner_model)
                    sv = explainer.shap_values(X_test_shap)

                shap_path = os.path.join(run_dir, "shap_values_test.npy")
                np.save(shap_path, sv)
                print(f"    shap_values_test.npy  ({winner_name}, shape {sv.shape})")
            except ImportError:
                print("    SHAP skipped — run: pip install shap")
            except Exception as exc:
                print(f"    SHAP computation failed: {exc}")


# ---------------------------------------------------------------------------
# Flat type lookup for error breakdown
# ---------------------------------------------------------------------------

def _load_flat_type_series(
    run_dir: str,
    split_names: list[str],
) -> dict[str, pd.Series]:
    """
    Load flat_type labels for evaluation splits from the saved parquet files.

    We use flat_type_ordinal from X_val/X_test and reverse-map it to
    flat_type strings for the error breakdown.

    Parameters:
        run_dir: Path to the feature engineering run directory.
        split_names: Evaluation split names to load.

    Returns:
        Dict of split name to pd.Series.
    """
    ordinal_to_flat = {v: k for k, v in {
        "1 Room": 1, "2 Room": 2, "3 Room": 3, "4 Room": 4,
        "5 Room": 5, "Executive": 6, "Multi-Generation": 7,
    }.items()}

    results = {}
    for name in split_names:
        X = pd.read_parquet(os.path.join(run_dir, f"X_{name}.parquet"))
        ft_series = X["flat_type_ordinal"].map(ordinal_to_flat).fillna("Unknown")
        results[name] = ft_series

    return results


# ---------------------------------------------------------------------------
# Final refit on train+val
# ---------------------------------------------------------------------------

def _refit_on_trainval(
    name: str,
    model,
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> object:
    """
    Refit a trained gradient-boosted model on the combined train+val set.

    n_estimators is taken from the early-stopping result of the phase-1
    train-only fit, so the model does not overfit by running indefinitely.
    Val rows receive weight 1.0 (they are the most recent data).

    Parameters:
        name:          One of 'xgboost', 'lgbm', 'catboost'.
        model:         Phase-1 trained model (used only to read best_iteration).
        best_params:   Optuna best hyperparameters dict.
        X_train / y_train: Training split features / log-price target.
        X_val / y_val:     Validation split features / log-price target.
        sample_weight: Per-row weights for X_train rows (val rows get 1.0).

    Returns:
        New model instance fitted on train+val, or the original model if the
        name is not a supported gradient booster.
    """
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = np.concatenate([y_train, y_val])
    sw_tv = (
        np.concatenate([sample_weight, np.ones(len(X_val))])
        if sample_weight is not None
        else None
    )

    if name == "xgboost":
        raw = getattr(model, "best_iteration", None)
        n_trees = (raw + 1) if (raw is not None and raw > 0) else XGB_N_ESTIMATORS
        params = {k: v for k, v in best_params.items()}
        params["n_estimators"] = n_trees
        params.pop("early_stopping_rounds", None)
        refitted = xgb.XGBRegressor(**params)
        refitted.fit(X_tv, y_tv, verbose=False, sample_weight=sw_tv)

    elif name == "lgbm":
        raw = getattr(model, "best_iteration_", None)
        n_trees = raw if (raw is not None and raw > 0) else LGBM_N_ESTIMATORS
        params = {k: v for k, v in best_params.items()}
        params["n_estimators"] = n_trees
        refitted = lgb.LGBMRegressor(**params)
        refitted.fit(X_tv, y_tv, sample_weight=sw_tv)

    elif name == "catboost":
        n_trees = getattr(model, "tree_count_", None) or CATBOOST_N_ESTIMATORS
        params = {k: v for k, v in best_params.items()}
        params["iterations"] = n_trees
        params.pop("early_stopping_rounds", None)
        params["cat_features"] = [
            i for i, col in enumerate(X_tv.columns)
            if col in {"flat_type_ordinal", "is_mature_estate"}
        ]
        params["allow_writing_files"] = False
        refitted = cb.CatBoostRegressor(**params)
        refitted.fit(X_tv, y_tv, sample_weight=sw_tv)

    else:
        return model  # RF: skip refit

    print(f"  {name}: refit on train+val ({len(X_tv):,} rows, {n_trees} trees).")
    return refitted


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
    split_frames = _load_splits(run_dir)
    X_train, y_train_df = split_frames["train"]
    X_val, y_val_df = split_frames["val"]
    X_test, y_test_df = split_frames["test"]

    y_train_log = y_train_df["log_price"].values
    y_val_log   = y_val_df["log_price"].values

    y_val_sgd  = y_val_df["resale_price"].values

    y_val_pi   = y_val_df["price_index"].values if "price_index" in y_val_df else None

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    if "future_holdout" in split_frames:
        X_future_holdout, _ = split_frames["future_holdout"]
        print(f"  X_future_holdout: {X_future_holdout.shape}")
    print(
        f"  Tuning config: XGB_TRIALS={XGB_TRIALS}, LGBM_TRIALS={LGBM_TRIALS}, "
        f"CATBOOST_TRIALS={CATBOOST_TRIALS}, "
        f"TUNING_SAMPLE_SIZE={TUNING_SAMPLE_SIZE:,}, "
        f"XGB_N_ESTIMATORS={XGB_N_ESTIMATORS}, "
        f"LGBM_N_ESTIMATORS={LGBM_N_ESTIMATORS}, "
        f"CATBOOST_N_ESTIMATORS={CATBOOST_N_ESTIMATORS}, "
        f"RF_ESTIMATORS={N_RF_ESTIMATORS}, "
        f"FRESH_TUNING={int(FRESH_TUNING)}, "
        f"DECAY_RATE={DECAY_RATE}"
    )
    config_notes = _training_config_notes(_training_config_snapshot())
    for note in config_notes:
        print(f"  WARNING: {note}")

    # Build study invalidation metadata from current run.
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    split_strategy = "rolling_temporal_holdout"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            _manifest = json.load(f)
        split_strategy = (
            _manifest.get("split_metadata", {}).get("split_strategy", split_strategy)
        )
    current_meta = {
        "split_strategy": split_strategy,
        "feature_cols": sorted(X_train.columns),
        "n_train_rows": len(X_train),
        "optuna_objective": "rmse",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Compute time-decay sample weights (recent transactions weighted higher).
    sample_weight: np.ndarray | None = None
    if "year_month_raw" in y_train_df.columns:
        ym = y_train_df["year_month_raw"].values
        max_months = int((ym.max() // 100) * 12 + (ym.max() % 100))
        months_array = (ym // 100).astype(int) * 12 + (ym % 100).astype(int)
        months_from_latest = max_months - months_array
        sample_weight = np.exp(-DECAY_RATE * months_from_latest)
        print(
            f"  Sample weighting: DECAY_RATE={DECAY_RATE}, "
            f"weight range [{sample_weight.min():.4f}, {sample_weight.max():.4f}]."
        )

    X_tune, y_tune_log = _build_tuning_subset(X_train, y_train_log, y_train_df)

    # Load flat type labels for error breakdown
    eval_split_names = [name for name in ("val", "test", "future_holdout") if name in split_frames]
    flat_type_by_split = _load_flat_type_series(run_dir, eval_split_names)

    # Step 1: Train XGBoost
    print("\n--- STEP 1: XGBOOST ---")
    t0 = time.time()
    xgb_model, xgb_params = train_xgboost(
        X_train, y_train_log, X_tune, y_tune_log, X_val, y_val_log, y_val_sgd, y_val_pi, run_dir,
        sample_weight=sample_weight, current_meta=current_meta,
    )
    print(f"  XGBoost training time: {time.time() - t0:.1f}s")

    # Step 2: Train LightGBM
    print("\n--- STEP 2: LIGHTGBM ---")
    t0 = time.time()
    lgbm_model, lgbm_params = train_lightgbm(
        X_train, y_train_log, X_tune, y_tune_log, X_val, y_val_log, y_val_sgd, y_val_pi, run_dir,
        sample_weight=sample_weight, current_meta=current_meta,
    )
    print(f"  LightGBM training time: {time.time() - t0:.1f}s")

    # Step 3: Train CatBoost
    print("\n--- STEP 3: CATBOOST ---")
    t0 = time.time()
    catboost_model, catboost_params = train_catboost(
        X_train, y_train_log, X_tune, y_tune_log, X_val, y_val_log, y_val_sgd, y_val_pi, run_dir,
        sample_weight=sample_weight, current_meta=current_meta,
    )
    print(f"  CatBoost training time: {time.time() - t0:.1f}s")

    # Step 4: Train Random Forest
    print("\n--- STEP 4: RANDOM FOREST ---")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train_log)
    print(f"  Random Forest training time: {time.time() - t0:.1f}s")

    base_models = {"xgboost": xgb_model, "lgbm": lgbm_model, "catboost": catboost_model, "rf": rf_model}

    # Step 4b: Build ensemble (CatBoost + LGBM, val-MAPE optimised blend)
    print("\n--- STEP 4b: ENSEMBLE ---")
    ensemble_model, ensemble_weights = build_ensemble(
        base_models, X_val, y_val_sgd, y_val_pi
    )
    if ensemble_model is not None:
        base_models["ensemble"] = ensemble_model

    # Step 5: Evaluate all
    print("\n--- STEP 5: EVALUATION ---")
    evaluation_splits = {
        "val": {
            "X": X_val,
            "y_sgd": y_val_sgd,
            "price_index": y_val_pi,
            "flat_type_series": flat_type_by_split["val"],
        },
        "test": {
            "X": X_test,
            "y_sgd": y_test_df["resale_price"].values,
            "price_index": y_test_df["price_index"].values if "price_index" in y_test_df else None,
            "flat_type_series": flat_type_by_split["test"],
        },
    }
    if "future_holdout" in split_frames:
        X_future_holdout, y_future_holdout_df = split_frames["future_holdout"]
        evaluation_splits["future_holdout"] = {
            "X": X_future_holdout,
            "y_sgd": y_future_holdout_df["resale_price"].values,
            "price_index": (
                y_future_holdout_df["price_index"].values
                if "price_index" in y_future_holdout_df
                else None
            ),
            "flat_type_series": flat_type_by_split["future_holdout"],
        }
    results = evaluate_all(base_models, evaluation_splits)

    # Step 6: Determine winner
    winner = determine_winner(results)
    print(f"\n  WINNER: {winner['winner'].upper()}")
    print(f"  {winner['justification']}")

    # Step 6b: Final refit of winner on train+val
    # Val metrics above were computed on train-only models (unbiased).
    # We now refit the winner using those same hyperparameters but with the val
    # period included in training, giving the model 3 extra months of context
    # before the test window.  We re-evaluate only on test and update the winner.
    print("\n--- STEP 6b: FINAL REFIT (TRAIN+VAL) ---")
    winner_name = winner["winner"]
    params_map = {
        "xgboost": xgb_params,
        "lgbm": lgbm_params,
        "catboost": catboost_params,
    }
    model_map = {
        "xgboost": xgb_model,
        "lgbm": lgbm_model,
        "catboost": catboost_model,
    }

    if winner_name == "ensemble":
        # Refit both base components; keep the blend weights unchanged.
        for comp in ("catboost", "lgbm"):
            refitted_comp = _refit_on_trainval(
                comp, model_map[comp], params_map[comp],
                X_train, y_train_log, X_val, y_val_log, sample_weight,
            )
            base_models[comp] = refitted_comp
        old_weights = base_models["ensemble"].weights
        base_models["ensemble"] = EnsembleModel(
            [base_models["catboost"], base_models["lgbm"]], old_weights
        )
    elif winner_name in params_map:
        refitted = _refit_on_trainval(
            winner_name, model_map[winner_name], params_map[winner_name],
            X_train, y_train_log, X_val, y_val_log, sample_weight,
        )
        base_models[winner_name] = refitted

    # Re-evaluate the refitted winner on test
    refitted_winner = base_models[winner_name]
    pred_log_test = refitted_winner.predict(X_test)
    pred_sgd_test = _inv_transform(
        pred_log_test,
        y_test_df["price_index"].values if "price_index" in y_test_df else None,
    )
    refit_test = _compute_metrics(y_test_df["resale_price"].values, pred_sgd_test)
    print(
        f"\n  {winner_name} after train+val refit — "
        f"Test: {_fmt_metrics(refit_test)}"
    )
    print(
        f"  MAPE change: {winner['test_mape']:.2f}% → {refit_test['mape']:.2f}% "
        f"({'↓' if refit_test['mape'] < winner['test_mape'] else '↑'}"
        f"{abs(refit_test['mape'] - winner['test_mape']):.2f}pp)"
    )
    # Preserve pre-refit metrics for audit, then update the primary test fields
    # so downstream systems (Supabase sync, pipeline summary) see the correct
    # metrics for the actually deployed model.
    winner["pre_refit_test_mape"] = winner["test_mape"]
    winner["pre_refit_test_rmse"] = winner["test_rmse"]
    winner["pre_refit_test_r2"] = winner["test_r2"]
    winner["refit_test_mape"] = round(refit_test["mape"], 4)
    winner["refit_test_rmse"] = round(refit_test["rmse"], 2)
    winner["refit_test_r2"] = round(refit_test["r2"], 6)
    winner["test_mape"] = winner["refit_test_mape"]
    winner["test_rmse"] = winner["refit_test_rmse"]
    winner["test_r2"] = winner["refit_test_r2"]

    # Step 7: Save outputs
    print("\n--- STEP 7: SAVING OUTPUTS ---")
    save_outputs(
        run_dir, base_models, results, winner, xgb_params, lgbm_params, catboost_params,
        ensemble_weights=ensemble_weights,
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
    if any("future_holdout" in res for res in results.values()):
        print("\n  Future holdout snapshot")
        print(f"  {'Model':<12} {'Future RMSE':>12} {'Future R2':>10} {'Future MAPE':>13}")
        print("  " + "-" * 52)
        for name in results:
            if "future_holdout" not in results[name]:
                continue
            future = results[name]["future_holdout"]
            marker = " <-- WINNER" if name == winner["winner"] else ""
            print(
                f"  {name:<12} {future['rmse']:>12,.0f} {future['r2']:>10.4f} "
                f"{future['mape']:>12.2f}%{marker}"
            )
        print("=" * 70)

    print(f"\nDone. All outputs saved to: {run_dir}/")


if __name__ == "__main__":
    main()
