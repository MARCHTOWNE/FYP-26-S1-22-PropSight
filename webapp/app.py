"""
app.py — HDB Resale Price Analytics Platform
=============================================
Flask web application serving:
  - Property valuation predictions 
  - Interactive transaction heatmaps (Leaflet.js)
  - Market analytics dashboard (Chart.js)
  - Guest / General user views

Run:
    cd webapp && python app.py
"""

import json
import math
import os
import pickle
from datetime import datetime, timedelta, timezone
from functools import wraps
from socket import timeout as SocketTimeout
import time as _time_mod
from urllib import error, parse, request as urllib_request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np
import pandas as pd
from flask import (
    Flask, flash, g, jsonify, redirect, render_template, request,
    session, url_for,
)

# ---------------------------------------------------------------------------
def _ttl_cache(maxsize=128, ttl=3600):
    """lru_cache replacement that expires entries after *ttl* seconds."""
    def decorator(fn):
        _cache = {}
        _timestamps = {}

        @wraps(fn)
        def wrapper(*args):
            now = _time_mod.monotonic()
            if args in _cache and (now - _timestamps[args]) < ttl:
                return _cache[args]
            #evict the oldest if at capacity
            if len(_cache) >= maxsize and args not in _cache:
                oldest_key = min(_timestamps, key=_timestamps.get)
                _cache.pop(oldest_key, None)
                _timestamps.pop(oldest_key, None)
            result = fn(*args)
            _cache[args] = result
            _timestamps[args] = now
            return result

        wrapper.cache_clear = lambda: (_cache.clear(), _timestamps.clear())
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# App setup

app = Flask(__name__)
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    raise RuntimeError(
        "SECRET_KEY environment variable must be set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _secret


def _first_existing_path(paths):
    fallback = None
    for p in paths:
        if not p:
            continue
        if fallback is None:
            fallback = p
        if os.path.exists(p):
            return p
    return fallback or paths[0]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
REFERENCE_DATA_DIR = _first_existing_path([
    os.environ.get("HDB_REFERENCE_DATA_DIR", ""),
    os.path.join(PROJECT_DIR, "Data Preprocessing", "reference_data"),
])


@_ttl_cache(maxsize=8, ttl=3600)
def _load_reference_points(filename):
    path = os.path.join(REFERENCE_DATA_DIR, filename)
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(records, list):
        return []
    return [
        {
            "name": record.get("name"),
            "lat": record.get("lat"),
            "lng": record.get("lng"),
        }
        for record in records
        if record.get("lat") is not None and record.get("lng") is not None
    ]

ASSETS_DIR = _first_existing_path([
    os.environ.get("MODEL_ASSETS_DIR", ""),
    os.path.join(PROJECT_DIR, "ML", "model_assets"),
])


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GENERAL_DAILY_AI_ANSWER_LIMIT = 3

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_KEY", "")
).strip()
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)
SUPABASE_USERS_TABLE = os.environ.get("SUPABASE_USERS_TABLE", "users")
SUPABASE_PREDICTIONS_TABLE = os.environ.get(
    "SUPABASE_PREDICTIONS_TABLE", "saved_predictions"
)

if not SUPABASE_ENABLED:
    raise RuntimeError(
        "Supabase runtime is required. Set SUPABASE_URL and "
        "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY in .env before starting the app."
    )

# ---------------------------------------------------------------------------
# Constants (mirrored from feature_engineering.py)

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "HOUGANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES",
    "TOA PAYOH",
}

FLAT_TYPE_ORDINAL = {
    "1 Room": 1, "2 Room": 2, "3 Room": 3, "4 Room": 4,
    "5 Room": 5, "Executive": 6, "Multi-Generation": 7,
}

FLAT_MODELS_BY_TYPE = {
    "1 Room": ["Improved"],
    "2 Room": ["2-Room", "DBSS", "Improved", "Model A", "Premium Apartment", "Standard"],
    "3 Room": [
        "Adjoined Flat", "DBSS", "Improved", "Model A", "New Generation",
        "Premium Apartment", "Simplified", "Standard", "Terrace",
    ],
    "4 Room": [
        "Adjoined Flat", "DBSS", "Improved", "Model A", "Model A2",
        "New Generation", "Premium Apartment", "Premium Apartment Loft",
        "Simplified", "Standard", "Terrace", "Type S1",
    ],
    "5 Room": [
        "3Gen", "Adjoined Flat", "DBSS", "Improved", "Improved Maisonette",
        "Model A", "Model A-Maisonette", "Premium Apartment",
        "Premium Apartment Loft", "Standard", "Type S2",
    ],
    "Executive": [
        "Adjoined Flat", "Apartment", "Maisonette", "Premium Apartment",
        "Premium Maisonette",
    ],
    "Multi-Generation": ["Multi Generation"],
}

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "rf": "Random Forest",
}

SCALE_COLS = [
    "floor_area_sqm", "storey_midpoint", "flat_age", "remaining_lease",
    "lease_commence_date", "month_sin", "month_cos", "year",
    "dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall",
    "dist_hawker_centre", "hawker_count_1km",
    "dist_high_demand_primary_school", "high_demand_primary_count_1km",
    "town_yoy_appreciation_lag1", "town_5yr_cagr_lag1",
    "sora_3m",
]

FEATURE_COLS = [
    "flat_type_ordinal", "town_enc", "flat_model_enc",
    "floor_area_sqm", "storey_midpoint", "flat_age", "remaining_lease",
    "lease_commence_date", "month_sin", "month_cos", "year",
    "is_mature_estate", "dist_mrt", "dist_cbd",
    "dist_primary_school", "dist_major_mall",
    "dist_hawker_centre", "hawker_count_1km",
    "dist_high_demand_primary_school", "high_demand_primary_count_1km",
    "town_yoy_appreciation_lag1", "town_5yr_cagr_lag1",
    "sora_3m",
]

STOREY_RANGES = [str(i) for i in range(1, 52)]

HDB_FIRST_YEAR = 1960
HDB_DATASET_START_YEAR = 1990
DEFAULT_FLOOR_AREA = 90
MAP_TRANSACTION_START_YEAR = 2024
MAP_TRANSACTION_LIMIT = 10000


def _build_map_storey_range_options(storey_values):
    floors = sorted(
        int(value) for value in (storey_values or [])
        if str(value).isdigit()
    )
    if not floors:
        return []

    grouped_ranges = []
    for idx in range(0, len(floors), 3):
        chunk = floors[idx:idx + 3]
        if not chunk:
            continue
        if len(chunk) == 1:
            grouped_ranges.append(f"{chunk[0]:02d}")
        else:
            grouped_ranges.append(f"{chunk[0]:02d} TO {chunk[-1]:02d}")
    return grouped_ranges


MAP_STOREY_RANGE_OPTIONS = _build_map_storey_range_options(STOREY_RANGES)


def _storey_midpoint(storey_range):
    """Parse storey value: individual floor number or 'XX TO YY' range."""
    if " TO " in str(storey_range):
        parts = storey_range.split(" TO ")
        return (int(parts[0]) + int(parts[1])) / 2
    return float(storey_range)


def _current_year():
    return datetime.now().year


def _default_lease_year_range():
    max_year = _current_year()
    avg_year = max(HDB_FIRST_YEAR, min(max_year, max_year - 35))
    return {
        "min_year": HDB_FIRST_YEAR,
        "max_year": max_year,
        "avg_year": avg_year,
    }


def _format_model_label(model_key):
    return MODEL_LABELS.get(model_key, str(model_key).replace("_", " ").title())


def _safe_metric(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_mape(prediction_year=None):
    performance = ARTEFACTS.get("performance", {}) or {}
    test_mape = _safe_metric(performance.get("test_mape"))
    future_mape = _safe_metric(performance.get("future_holdout_mape"))

    try:
        if (
            prediction_year is not None
            and int(prediction_year) > _current_year()
            and future_mape is not None
        ):
            return future_mape
    except (TypeError, ValueError):
        pass

    return test_mape if test_mape is not None else future_mape


def _enrich_prediction_result(predicted_price, prediction_year=None, result=None):
    rounded_price = int(round(_coerce_float(predicted_price, 0.0) or 0.0))
    mape = _prediction_mape(prediction_year)

    if mape is None:
        price_low = rounded_price
        price_high = rounded_price
        mape_display = None
    else:
        margin = max(0.0, mape) / 100.0
        price_low = int(round(max(0.0, rounded_price * (1 - margin))))
        price_high = int(round(max(0.0, rounded_price * (1 + margin))))
        mape_display = round(mape, 2)

    enriched = dict(result or {})
    enriched["predicted_price"] = rounded_price
    enriched["price_low"] = price_low
    enriched["price_high"] = price_high
    enriched["mape"] = mape_display
    performance = (globals().get("ARTEFACTS") or {}).get("performance") or {}
    enriched.setdefault("model_trained_at", performance.get("model_trained_at"))
    return enriched


def _rpc_param_not_available(exc, function_name, param_name):
    details = str(exc)
    return (
        function_name in details
        and param_name in details
        and (
            "Could not find the function" in details
            or "no matches were found" in details
            or "does not exist" in details
            or "schema cache" in details
        )
    )


def _year_window_label(start_year, end_year):
    if start_year is None and end_year is None:
        return None
    if start_year == end_year:
        return str(start_year)
    if start_year is None:
        return f"<= {end_year}"
    if end_year is None:
        return f">= {start_year}"
    return f"{start_year}-{end_year}"


def _manifest_split_window(manifest, split_name):
    split_metadata = manifest.get("split_metadata", {}) or {}
    start_year = split_metadata.get(f"{split_name}_min_year")
    end_year = split_metadata.get(f"{split_name}_max_year")

    if start_year is None and end_year is None:
        split_years = manifest.get("split_years", {}) or {}
        start_year = split_years.get(f"{split_name}_start_year")
        end_year = split_years.get(f"{split_name}_end_year")

    return _year_window_label(start_year, end_year)


def _format_model_trained_at(run_at_str):
    """Return a human-readable SGT datetime string from manifest run_at ISO string."""
    if not run_at_str:
        return None
    try:
        from datetime import timezone, timedelta
        import datetime as _dt
        dt = _dt.datetime.fromisoformat(run_at_str)
        sgt = dt.astimezone(timezone(timedelta(hours=8)))
        return sgt.strftime("%-d %b %Y, %-I:%M %p SGT")
    except Exception:
        return None


def _build_model_performance(metrics, manifest, serving_model_key):
    winner = metrics.get("winner", {}) or {}
    model_results = metrics.get("model_results", {}) or {}
    serving_results = model_results.get(serving_model_key, {}) or {}
    test_metrics = serving_results.get("test", {}) or {}
    future_metrics = serving_results.get("future_holdout", {}) or {}

    test_mape = _safe_metric(test_metrics.get("mape"))
    if test_mape is None and winner.get("winner") == serving_model_key:
        test_mape = _safe_metric(winner.get("test_mape"))

    test_rmse = _safe_metric(test_metrics.get("rmse"))
    if test_rmse is None and winner.get("winner") == serving_model_key:
        test_rmse = _safe_metric(winner.get("test_rmse"))

    test_r2 = _safe_metric(test_metrics.get("r2"))
    if test_r2 is None and winner.get("winner") == serving_model_key:
        test_r2 = _safe_metric(winner.get("test_r2"))

    future_mape = _safe_metric(future_metrics.get("mape"))
    future_rmse = _safe_metric(future_metrics.get("rmse"))
    future_r2 = _safe_metric(future_metrics.get("r2"))

    return {
        "key": serving_model_key,
        "label": _format_model_label(serving_model_key),
        "is_winner": winner.get("winner") == serving_model_key,
        "selection_metric": winner.get("selection_metric"),
        "test_mape": test_mape,
        "test_mape_display": round(test_mape, 2) if test_mape is not None else None,
        "test_rmse": test_rmse,
        "test_rmse_display": f"{round(test_rmse):,}" if test_rmse is not None else None,
        "test_r2": test_r2,
        "test_r2_display": f"{test_r2:.3f}" if test_r2 is not None else None,
        "future_holdout_mape": future_mape,
        "future_holdout_mape_display": round(future_mape, 2) if future_mape is not None else None,
        "future_holdout_rmse": future_rmse,
        "future_holdout_rmse_display": (
            f"{round(future_rmse):,}" if future_rmse is not None else None
        ),
        "future_holdout_r2": future_r2,
        "future_holdout_r2_display": (
            f"{future_r2:.3f}" if future_r2 is not None else None
        ),
        "val_window": _manifest_split_window(manifest, "val"),
        "test_window": _manifest_split_window(manifest, "test"),
        "future_holdout_window": _manifest_split_window(manifest, "future_holdout"),
        "model_trained_at": _format_model_trained_at(manifest.get("run_at")),
    }


def _resolve_serving_model_key(run_dir, metrics):
    preferred = (metrics.get("winner", {}) or {}).get("winner")
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["xgboost", "lgbm", "catboost", "rf"])

    seen = set()
    for model_key in candidates:
        if model_key in seen:
            continue
        seen.add(model_key)
        model_path = os.path.join(run_dir, f"{model_key}_model.pkl")
        if not os.path.exists(model_path):
            continue
        return model_key, model_path

    raise FileNotFoundError(
        f"No supported serving model artefact found in {run_dir}"
    )


def _data_year_bounds():
    manifest = (globals().get("ARTEFACTS") or {}).get("manifest", {}) or {}
    split_metadata = manifest.get("split_metadata", {}) or {}

    years = []
    for key, value in split_metadata.items():
        if key.endswith("_min_year") or key.endswith("_max_year"):
            try:
                years.append(int(value))
            except (TypeError, ValueError):
                continue

    if not years:
        split_years = manifest.get("split_years", {}) or {}
        for key, value in split_years.items():
            if key.endswith("_start_year") or key.endswith("_end_year"):
                try:
                    years.append(int(value))
                except (TypeError, ValueError):
                    continue

    if years:
        return min(years), max(years)

    return HDB_DATASET_START_YEAR, _current_year()


@app.context_processor
def inject_runtime_template_globals():
    lease_year_range = _default_lease_year_range()
    data_year_start, data_year_end = _data_year_bounds()
    return {
        "current_year": _current_year(),
        "lease_year_min": lease_year_range["min_year"],
        "lease_year_max": lease_year_range["max_year"],
        "default_lease_year": lease_year_range["avg_year"],
        "data_year_start": data_year_start,
        "data_year_end": data_year_end,
        "active_model_performance": ARTEFACTS.get("performance"),
    }


# ---------------------------------------------------------------------------
# Load model artefacts at startup
# ---------------------------------------------------------------------------

REQUIRED_ARTEFACT_FILES = ("scaler.pkl", "target_encoders.pkl", "metrics.json")


def _is_valid_run_dir(run_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return False

    for filename in REQUIRED_ARTEFACT_FILES:
        if not os.path.exists(os.path.join(run_dir, filename)):
            return False

    try:
        with open(os.path.join(run_dir, "metrics.json")) as f:
            metrics = json.load(f)
        _resolve_serving_model_key(run_dir, metrics)
    except (FileNotFoundError, json.JSONDecodeError, OSError, pickle.PickleError):
        return False

    return True


def _resolve_run_dir():
    latest_file = os.path.join(ASSETS_DIR, "latest.txt")
    run_dir = None

    if os.path.exists(latest_file):
        with open(latest_file) as f:
            configured = f.read().strip()
        if configured:
            if os.path.isabs(configured):
                candidates = [configured]
            else:
                candidates = [
                    os.path.join(PROJECT_DIR, configured),
                    os.path.join(ASSETS_DIR, configured),
                    os.path.join(ASSETS_DIR, os.path.basename(configured)),
                ]
            run_dir = _first_existing_path(candidates)
            if not _is_valid_run_dir(run_dir):
                run_dir = None

    if run_dir is None:
        # Fallback: pick the newest valid artefact directory under ASSETS_DIR.
        dirs = []
        if os.path.isdir(ASSETS_DIR):
            for name in os.listdir(ASSETS_DIR):
                p = os.path.join(ASSETS_DIR, name)
                if _is_valid_run_dir(p):
                    dirs.append(p)
        if not dirs:
            raise FileNotFoundError(
                f"No valid model artefact run directory found under {ASSETS_DIR}"
            )
        run_dir = sorted(dirs)[-1]

    return run_dir


def _resolve_serving_feature_cols(artefacts, run_dir):
    """Feature order/names for the loaded model (may differ from app defaults)."""
    manifest = artefacts.get("manifest") or {}
    cols = manifest.get("feature_cols")
    if isinstance(cols, list) and cols:
        return list(cols)
    path = os.path.join(run_dir, "feature_cols.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            return lines
    return list(FEATURE_COLS)


def _resolve_serving_scale_cols(artefacts, run_dir):
    """Columns the fitted StandardScaler expects (may omit newer engineered fields)."""
    manifest = artefacts.get("manifest") or {}
    cols = manifest.get("scale_cols")
    if isinstance(cols, list) and cols:
        return list(cols)
    scaler = artefacts.get("scaler")
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return [str(x) for x in fn]
    return list(SCALE_COLS)


def _load_artefacts():
    run_dir = _resolve_run_dir()
    artefacts = {}

    with open(os.path.join(run_dir, "scaler.pkl"), "rb") as f:
        artefacts["scaler"] = pickle.load(f)

    with open(os.path.join(run_dir, "target_encoders.pkl"), "rb") as f:
        artefacts["encoders"] = pickle.load(f)

    with open(os.path.join(run_dir, "metrics.json")) as f:
        artefacts["metrics"] = json.load(f)

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            artefacts["manifest"] = json.load(f)
    else:
        artefacts["manifest"] = {}

    serving_model_key, serving_model_path = _resolve_serving_model_key(
        run_dir,
        artefacts["metrics"],
    )

    with open(serving_model_path, "rb") as f:
        artefacts["model"] = pickle.load(f)

    artefacts["model_key"] = serving_model_key
    artefacts["model_label"] = _format_model_label(serving_model_key)

    price_index_path = os.path.join(run_dir, "price_index.pkl")
    if os.path.exists(price_index_path):
        with open(price_index_path, "rb") as f:
            artefacts["price_index"] = pickle.load(f)
    else:
        artefacts["price_index"] = None

    artefacts["target_transform"] = artefacts["manifest"].get(
        "target_transform",
        "rpi_adjusted_log_price" if artefacts["price_index"] is not None else "log1p_resale_price",
    )
    artefacts["performance"] = _build_model_performance(
        artefacts["metrics"],
        artefacts["manifest"],
        serving_model_key,
    )

    artefacts["run_dir"] = run_dir
    artefacts["serving_feature_cols"] = _resolve_serving_feature_cols(artefacts, run_dir)
    artefacts["serving_scale_cols"] = _resolve_serving_scale_cols(artefacts, run_dir)

    rolling_snapshot_path = os.path.join(run_dir, "rolling_stats_snapshot.pkl")
    if os.path.exists(rolling_snapshot_path):
        with open(rolling_snapshot_path, "rb") as f:
            artefacts["rolling_stats"] = pickle.load(f)
    else:
        artefacts["rolling_stats"] = {}

    return artefacts


ARTEFACTS = _load_artefacts()


# ---------------------------------------------------------------------------
# SORA rate — fetched once at startup, used for all predictions
# ---------------------------------------------------------------------------

def _fetch_current_sora() -> float:
    """Fetch the most recent 3-month compounded SORA from MAS API."""
    _MAS_SORA_URL = (
        "https://eservices.mas.gov.sg/api/action/datastore/search.json"
        "?resource_id=9a0bf149-308c-4bd2-832d-76c8e6cb47ed&limit=30&sort=end_of_day+desc"
    )
    try:
        req = urllib_request.Request(
            _MAS_SORA_URL,
            headers={"User-Agent": "PropSight/1.0 (HDB Resale Price Prediction)"},
        )
        with urllib_request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        records = data.get("result", {}).get("records", [])
        for rec in records:
            val = rec.get("comp_sora_3m")
            if val not in (None, "", "-"):
                return float(val)
    except Exception:
        pass
    return 2.5


CURRENT_SORA_3M: float = _fetch_current_sora()


def _serving_feature_cols():
    cols = ARTEFACTS.get("serving_feature_cols")
    if isinstance(cols, list) and cols:
        return cols
    return FEATURE_COLS


def _serving_scale_cols():
    cols = ARTEFACTS.get("serving_scale_cols")
    if isinstance(cols, list) and cols:
        return cols
    return SCALE_COLS


SHAP_SUPPORTED_MODEL_KEYS = {"xgboost", "lgbm", "catboost", "rf"}
_SHAP_EXPLAINER = None
_SHAP_IMPORT_ERROR = None


class SupabaseError(RuntimeError):
    """Raised when the Supabase REST API returns an error."""


# Track the most recent RPC timeout for diagnostics, but do not permanently
# disable Supabase for the lifetime of the process.
_supabase_last_rpc_error = None


def _supabase_headers(prefer=None):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _supabase_request(table, method="GET", filters=None, payload=None, prefer=None):
    if not SUPABASE_ENABLED:
        raise SupabaseError("Supabase is not configured.")

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if filters:
        url = f"{url}?{parse.urlencode(filters)}"

    data = None
    headers = _supabase_headers(prefer=prefer)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8")
        raise SupabaseError(details or f"Supabase request failed with {exc.code}") from exc


def _supabase_count(table, filters=None):
    if not SUPABASE_ENABLED:
        raise SupabaseError("Supabase is not configured.")

    query_filters = dict(filters or {})
    query_filters.setdefault("select", "id")
    query_filters.setdefault("limit", "1")
    url = f"{SUPABASE_URL}/rest/v1/{table}?{parse.urlencode(query_filters)}"
    req = urllib_request.Request(
        url,
        headers={**_supabase_headers(prefer="count=exact"), "Range": "0-0"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(req) as resp:
            content_range = resp.headers.get("Content-Range", "")
            if "/" not in content_range:
                return 0
            total = content_range.rsplit("/", 1)[-1]
            return int(total) if total.isdigit() else 0
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8")
        raise SupabaseError(details or f"Supabase count failed with {exc.code}") from exc


def _supabase_rpc(function_name, params=None):
    """Call a Supabase PostgreSQL RPC function."""
    global _supabase_last_rpc_error
    if not SUPABASE_ENABLED:
        raise SupabaseError("Supabase is not configured.")

    url = f"{SUPABASE_URL}/rest/v1/rpc/{function_name}"
    payload = params or {}
    data = json.dumps(payload).encode("utf-8")
    headers = _supabase_headers()
    headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return []
            return json.loads(raw)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8")
        raise SupabaseError(details or f"Supabase RPC failed with {exc.code}") from exc
    except (error.URLError, SocketTimeout, OSError) as exc:
        _supabase_last_rpc_error = str(exc)
        raise SupabaseError(f"Supabase RPC timed out: {exc}") from exc


def _supabase_auth(path, method="POST", payload=None, access_token=None):
    """Call a Supabase Auth API endpoint."""
    if not SUPABASE_ENABLED:
        raise SupabaseError("Supabase is not configured.")
    url = f"{SUPABASE_URL}/auth/v1{path}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token or SUPABASE_KEY}",
    }
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(req) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else None
    except error.HTTPError as exc:
        details = exc.read().decode()
        raise SupabaseError(details or f"Auth API failed with {exc.code}") from exc


def _call_gemini(prompt, max_tokens=1024):
    """call Google Gemini API and return text response, or None on error."""
    if not GEMINI_API_KEY:
        return None
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
    }).encode("utf-8")
    req = urllib_request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except (error.HTTPError, error.URLError, SocketTimeout, KeyError, IndexError):
        return None


def _get_daily_ai_answer_count(user_id):
    """count AI answer expansions for today."""
    cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    rows = _supabase_request(
        "feature_view_log",
        filters={
            "user_id": f"eq.{user_id}",
            "feature": "eq.ai_answer",
            "created_at": f"gte.{cutoff}",
        },
    ) or []
    return len(rows)


def _check_ai_answer_limit():
    """returns (allowed, used, limit). Premium always allowed."""
    tier = session.get("subscription_tier", "general")
    if tier == "premium":
        return True, 0, 0
    user_id = _session_user_id()
    used = _get_daily_ai_answer_count(user_id)
    return used < GENERAL_DAILY_AI_ANSWER_LIMIT, used, GENERAL_DAILY_AI_ANSWER_LIMIT


def _session_user_id():
    try:
        user_id = int(session.get("user_id"))
    except (TypeError, ValueError):
        return None
    return user_id if user_id > 0 else None


def _get_supabase_user_by_email(email):
    rows = _supabase_request(
        SUPABASE_USERS_TABLE,
        filters={"email": f"eq.{email}", "limit": "1"},
    ) or []
    return rows[0] if rows else None


def _get_supabase_feature_view_rows(user_id, feature):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return _supabase_request(
        "feature_view_log",
        filters={
            "user_id": f"eq.{user_id}",
            "feature": f"eq.{feature}",
            "created_at": f"gte.{cutoff}",
        },
    ) or []


def _save_prediction_record(user_id, prediction):
    payload = {"user_id": user_id, **prediction}
    try:
        rows = _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            method="POST",
            payload=payload,
            prefer="return=representation",
        )
        return rows[0] if rows else None
    except SupabaseError:
        # Retry without street_name/block if columns don't exist yet
        payload.pop("street_name", None)
        payload.pop("block", None)
        rows = _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            method="POST",
            payload=payload,
            prefer="return=representation",
        )
        return rows[0] if rows else None


def _get_saved_predictions(user_id):
    return _supabase_request(
        SUPABASE_PREDICTIONS_TABLE,
        filters={"user_id": f"eq.{user_id}", "order": "created_at.desc"},
    ) or []


def _parse_saved_prediction_timestamp(value):
    if not value:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        parsed = None

        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue

    if parsed and parsed.tzinfo is not None:
        return parsed.astimezone()
    return parsed


def _normalize_saved_prediction(prediction):
    item = dict(prediction) if not isinstance(prediction, dict) else dict(prediction)

    try:
        item["id"] = int(item["id"])
    except (KeyError, TypeError, ValueError):
        pass

    created_at = _parse_saved_prediction_timestamp(item.get("created_at"))
    if created_at:
        item["created_at_display"] = created_at.strftime("%d %b %Y, %I:%M %p")
        item["created_at_date_display"] = created_at.strftime("%d %b %Y")
        item["created_at_time_display"] = created_at.strftime("%I:%M %p").lstrip("0")
    else:
        fallback = str(item.get("created_at") or "Unknown")
        item["created_at_display"] = fallback
        item["created_at_date_display"] = fallback
        item["created_at_time_display"] = ""

    try:
        price_label = f"${float(item.get('predicted_price', 0)):,.0f}"
    except (TypeError, ValueError):
        price_label = "N/A"

    predicted_price = _coerce_float(item.get("predicted_price"))
    if predicted_price is not None:
        if _coerce_float(item.get("price_low")) is None or _coerce_float(item.get("price_high")) is None:
            enriched = _enrich_prediction_result(predicted_price, result=item)
            item["price_low"] = enriched["price_low"]
            item["price_high"] = enriched["price_high"]

    item["comparison_option_label"] = " · ".join(
        part
        for part in (
            item.get("town"),
            item.get("flat_type"),
            price_label,
            item["created_at_display"],
        )
        if part
    )
    return item


def _prepare_saved_predictions(predictions):
    return [_normalize_saved_prediction(p) for p in predictions]


def _get_saved_prediction_by_id(predictions, prediction_id):
    try:
        target_id = int(prediction_id)
    except (TypeError, ValueError):
        return None
    return next((p for p in predictions if p.get("id") == target_id), None)


def _format_distance(value):
    """Format a distance value in meters to a readable string."""
    if value is None:
        return "N/A"
    try:
        m = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if m >= 1000:
        return f"{m / 1000:,.1f} km"
    return f"{m:,.0f}m"


def _build_comparison_analysis(payloads):
    """Analyze ALL properties together and return unified comparison data."""
    if len(payloads) < 2:
        return None

    labels = [chr(ord("A") + i) for i in range(len(payloads))]

    # Define feature rows: (key, label, format_type, best_direction)
    # best_direction: "min" = lower is better, "max" = higher is better, None = neutral
    feature_defs = [
        ("predicted_price", "Predicted Price", "currency", None),
        ("price_per_sqm", "Price / sqm", "currency", None),
        ("floor_area", "Floor Area", "sqm", "max"),
        ("storey_midpoint", "Storey (mid)", "floor", "max"),
        ("remaining_lease", "Remaining Lease", "yrs", "max"),
        ("flat_age", "Flat Age", "yrs", "min"),
        ("dist_mrt", "Nearest MRT", "dist", "min"),
        ("dist_school", "Nearest School", "dist", "min"),
        ("dist_high_demand_school", "Top Primary School", "dist", "min"),
        ("dist_mall", "Nearest Mall", "dist", "min"),
        ("hawker_count_1km", "Hawkers within 1km", "count", "max"),
        ("dist_cbd", "Distance to CBD", "dist", "min"),
        ("is_mature", "Mature Estate", "yesno", None),
    ]

    features = []
    for key, label, fmt, best_dir in feature_defs:
        raw_values = [p.get(key) for p in payloads]

        # Format display values
        display_values = []
        for v in raw_values:
            if v is None:
                display_values.append("N/A")
            elif fmt == "currency":
                display_values.append(f"${float(v):,.0f}")
            elif fmt == "sqm":
                display_values.append(f"{float(v):,.0f} sqm")
            elif fmt == "floor":
                display_values.append(f"{float(v):,.0f}")
            elif fmt == "yrs":
                display_values.append(f"{int(v)} yrs")
            elif fmt == "dist":
                display_values.append(_format_distance(v))
            elif fmt == "yesno":
                display_values.append("Yes" if v else "No")
            elif fmt == "count":
                display_values.append(f"{float(v):,.0f}")
            else:
                display_values.append(str(v))

        # Determine best/worst indices
        best_idx = None
        worst_idx = None
        if best_dir:
            numeric = []
            for i, v in enumerate(raw_values):
                try:
                    numeric.append((i, float(v)))
                except (TypeError, ValueError):
                    pass
            if len(numeric) >= 2:
                if best_dir == "min":
                    best_idx = min(numeric, key=lambda x: x[1])[0]
                    worst_idx = max(numeric, key=lambda x: x[1])[0]
                else:
                    best_idx = max(numeric, key=lambda x: x[1])[0]
                    worst_idx = min(numeric, key=lambda x: x[1])[0]
                # Don't highlight if all values are the same
                if all(n[1] == numeric[0][1] for n in numeric):
                    best_idx = None
                    worst_idx = None

        features.append({
            "label": label,
            "values": display_values,
            "best_idx": best_idx,
            "worst_idx": worst_idx,
        })

    # Generate dynamic insights from the actual data
    insights = _generate_comparison_insights(payloads, labels)

    return {
        "labels": labels,
        "features": features,
        "insights": insights,
    }


def _generate_comparison_insights(payloads, labels):
    """Generate dynamic insights by analyzing real feature data across all properties."""
    insights = []

    def _best_worst(key, direction="min"):
        """Find best and worst property for a numeric key."""
        vals = []
        for i, p in enumerate(payloads):
            v = p.get(key)
            if v is not None:
                try:
                    vals.append((i, float(v)))
                except (TypeError, ValueError):
                    pass
        if len(vals) < 2:
            return None, None, None, None
        if all(v[1] == vals[0][1] for v in vals):
            return None, None, None, None  # all same
        if direction == "min":
            best = min(vals, key=lambda x: x[1])
            worst = max(vals, key=lambda x: x[1])
        else:
            best = max(vals, key=lambda x: x[1])
            worst = min(vals, key=lambda x: x[1])
        return labels[best[0]], best[1], labels[worst[0]], worst[1]

    # MRT distance insight
    best_l, best_v, worst_l, worst_v = _best_worst("dist_mrt", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} is closest to an MRT station ({_format_distance(best_v)}), "
            f"while {worst_l} is farthest ({_format_distance(worst_v)}). "
            f"Proximity to MRT typically increases property value."
        )

    # School distance insight
    best_l, best_v, worst_l, worst_v = _best_worst("dist_school", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} is nearest to a primary school ({_format_distance(best_v)}), "
            f"while {worst_l} is farthest ({_format_distance(worst_v)})."
        )

    best_l, best_v, worst_l, worst_v = _best_worst("dist_high_demand_school", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} is closest to a high-demand primary school "
            f"({_format_distance(best_v)}), while {worst_l} is farthest "
            f"({_format_distance(worst_v)})."
        )

    # Mall distance insight
    best_l, best_v, worst_l, worst_v = _best_worst("dist_mall", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} is closest to a major mall ({_format_distance(best_v)}), "
            f"while {worst_l} is farthest ({_format_distance(worst_v)})."
        )

    best_l, best_v, worst_l, worst_v = _best_worst("hawker_count_1km", "max")
    if best_l:
        insights.append(
            f"Prediction {best_l} has the densest hawker access with {best_v:,.0f} hawker "
            f"centre(s) within 1 km, while {worst_l} has the least at {worst_v:,.0f}."
        )

    # CBD distance insight
    best_l, best_v, worst_l, worst_v = _best_worst("dist_cbd", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} is closest to the CBD ({_format_distance(best_v)}), "
            f"while {worst_l} is farthest ({_format_distance(worst_v)}). "
            f"Closer CBD proximity generally commands a premium."
        )

    # Remaining lease
    best_l, best_v, worst_l, worst_v = _best_worst("remaining_lease", "max")
    if best_l:
        insights.append(
            f"Prediction {best_l} has the most remaining lease ({int(best_v)} yrs) "
            f"vs {worst_l} ({int(worst_v)} yrs). "
            f"Longer remaining lease supports higher valuations."
        )

    # Price per sqm (value for money)
    best_l, best_v, worst_l, worst_v = _best_worst("price_per_sqm", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} offers the best value at ${best_v:,.0f}/sqm, "
            f"while {worst_l} is the most expensive at ${worst_v:,.0f}/sqm."
        )

    # Storey premium
    best_l, best_v, worst_l, worst_v = _best_worst("storey_midpoint", "max")
    if best_l:
        insights.append(
            f"Prediction {best_l} is on a higher floor (level {int(best_v)}) "
            f"vs {worst_l} (level {int(worst_v)}). "
            f"Higher floors generally command a premium for better views and ventilation."
        )

    # Mature estate
    mature_flags = [p.get("is_mature", False) for p in payloads]
    towns = [p.get("town", "") for p in payloads]
    unique_towns = list(dict.fromkeys(towns))  # preserve order, deduplicate
    if all(mature_flags):
        if len(unique_towns) == 1:
            insights.append(
                f"All properties are in {unique_towns[0]} (mature estate), "
                f"so estate maturity is not a differentiating factor."
            )
        else:
            insights.append(
                f"All properties are in mature estates ({', '.join(unique_towns)}), "
                f"which tend to have established amenities and higher demand."
            )
    elif not any(mature_flags):
        insights.append(
            f"All properties are in non-mature estates ({', '.join(unique_towns)}), "
            f"which may offer newer developments but typically lower prices."
        )
    else:
        mature_labels = [labels[i] for i, m in enumerate(mature_flags) if m]
        non_mature_labels = [labels[i] for i, m in enumerate(mature_flags) if not m]
        insights.append(
            f"Predictions {', '.join(mature_labels)} are in mature estates, "
            f"while {', '.join(non_mature_labels)} are in non-mature estates. "
            f"Mature estates typically command higher prices due to established amenities."
        )

    # Different flat types
    flat_types = [p.get("flat_type", "") for p in payloads]
    unique_ft = set(flat_types)
    if len(unique_ft) > 1:
        insights.append(
            f"Properties have different flat types ({', '.join(unique_ft)}), "
            f"which significantly affects pricing."
        )

    if not insights:
        insights.append(
            "All properties have very similar attributes and location features, "
            "resulting in close valuations."
        )

    return insights


def _comparison_max_panels():
    """Return maximum number of comparison panels for the current user."""
    tier = session.get("subscription_tier", "general")
    return 5 if tier == "premium" else 2


def _get_comparison_saved_prediction_ids():
    max_panels = _comparison_max_panels()
    raw_ids = session.get("comparison_saved_prediction_ids", [])
    ids = []
    for value in raw_ids[:max_panels]:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _set_comparison_saved_prediction_ids(prediction_ids):
    max_panels = _comparison_max_panels()
    session["comparison_saved_prediction_ids"] = [int(pid) for pid in prediction_ids[:max_panels]]


def _push_comparison_saved_prediction_id(prediction_id):
    max_panels = _comparison_max_panels()
    updated = [pid for pid in _get_comparison_saved_prediction_ids() if pid != prediction_id]
    updated.append(int(prediction_id))
    updated = updated[-max_panels:]
    _set_comparison_saved_prediction_ids(updated)
    return updated


def _default_prediction_form_data():
    default_lease_year = _default_lease_year_range()["avg_year"]
    return {
        "town": "",
        "flat_type": next(iter(FLAT_TYPE_ORDINAL.keys()), ""),
        "flat_model": FLAT_MODELS[0] if FLAT_MODELS else "",
        "floor_area": DEFAULT_FLOOR_AREA,
        "storey_range": STOREY_RANGES[0] if STOREY_RANGES else "",
        "lease_commence": default_lease_year,
        "street_name": "",
        "block": "",
    }


def _prediction_form_from_saved(saved_prediction):
    form_data = _default_prediction_form_data()
    if not saved_prediction:
        return form_data

    form_data.update({
        "town": saved_prediction.get("town", ""),
        "flat_type": saved_prediction.get("flat_type", form_data["flat_type"]),
        "flat_model": saved_prediction.get("flat_model", form_data["flat_model"]),
        "floor_area": saved_prediction.get("floor_area", form_data["floor_area"]),
        "storey_range": saved_prediction.get("storey_range", form_data["storey_range"]),
        "lease_commence": saved_prediction.get("lease_commence", form_data["lease_commence"]),
        "street_name": saved_prediction.get("street_name", ""),
        "block": saved_prediction.get("block", ""),
    })
    return form_data


def _extract_prediction_form_data(source, prefix, seed=None):
    form_data = _default_prediction_form_data()
    if seed:
        form_data.update(seed)

    for field in ("town", "flat_type", "flat_model", "storey_range", "street_name", "block"):
        value = source.get(f"{prefix}_{field}", "")
        if value:
            form_data[field] = value.strip()

    for field in ("floor_area", "lease_commence"):
        value = source.get(f"{prefix}_{field}", "")
        if value != "":
            form_data[field] = value.strip() if isinstance(value, str) else value

    return form_data


def _run_prediction_form(form_data, infer_flat_type=False):
    resolved_form, assumptions = _complete_prediction_form_data(
        form_data,
        infer_flat_type=infer_flat_type,
    )

    # Block-level distances if available
    block_distances = None
    if resolved_form.get("block") and resolved_form.get("street_name"):
        block_distances = _get_block_distances(
            resolved_form["town"], resolved_form["street_name"], resolved_form["block"]
        )

    result = predict_price(
        resolved_form["town"],
        resolved_form["flat_type"],
        resolved_form["flat_model"],
        resolved_form["floor_area"],
        resolved_form["storey_range"],
        resolved_form["lease_commence"],
        override_distances=block_distances,
    )
    result["assumptions"] = assumptions

    # Enrich payload with distances and derived features for comparison
    town = resolved_form["town"]
    if block_distances:
        dists = block_distances
    else:
        town_dists = TOWN_DISTANCES.get(town, {})
        dists = {
            "dist_mrt": town_dists.get("avg_dist_mrt"),
            "dist_cbd": town_dists.get("avg_dist_cbd"),
            "dist_school": town_dists.get("avg_dist_school"),
            "dist_mall": town_dists.get("avg_dist_mall"),
            "dist_hawker": town_dists.get("avg_dist_hawker"),
            "hawker_count_1km": town_dists.get("avg_hawker_count_1km"),
            "dist_high_demand_school": town_dists.get("avg_dist_high_demand_school"),
            "high_demand_primary_count_1km": town_dists.get("avg_high_demand_primary_count_1km"),
        }

    flat_age = datetime.now().year - resolved_form["lease_commence"]
    remaining_lease = max(0, 99 - flat_age)
    storey_mid = _storey_midpoint(resolved_form["storey_range"])
    price_per_sqm = round(result["predicted_price"] / resolved_form["floor_area"], 2) if resolved_form["floor_area"] else 0

    payload = {
        **resolved_form, **result,
        "dist_mrt": dists.get("dist_mrt"),
        "dist_cbd": dists.get("dist_cbd"),
        "dist_school": dists.get("dist_school"),
        "dist_high_demand_school": dists.get("dist_high_demand_school"),
        "dist_mall": dists.get("dist_mall"),
        "dist_hawker": dists.get("dist_hawker"),
        "hawker_count_1km": dists.get("hawker_count_1km"),
        "high_demand_primary_count_1km": dists.get("high_demand_primary_count_1km"),
        "is_mature": town in MATURE_ESTATES,
        "flat_age": flat_age,
        "remaining_lease": remaining_lease,
        "storey_midpoint": storey_mid,
        "price_per_sqm": price_per_sqm,
    }

    return resolved_form, result, payload


def _delete_saved_prediction(pred_id, user_id):
    _supabase_request(
        SUPABASE_PREDICTIONS_TABLE,
        method="DELETE",
        filters={"id": f"eq.{int(pred_id)}", "user_id": f"eq.{user_id}"},
    )


def _get_towns():
    try:
        rows = _supabase_rpc("rpc_get_towns") or []
        return [r["town"] for r in rows]
    except SupabaseError:
        return []


def _get_flat_models():
    try:
        rows = _supabase_rpc("rpc_get_flat_models") or []
        return [r["flat_model"] for r in rows]
    except SupabaseError:
        return []


def _get_town_avg_distances():
    """Pre-compute average distances per town for prediction defaults."""
    try:
        rows = _supabase_rpc("rpc_get_town_avg_distances") or []
        return {
            r["town"]: {
                "avg_dist_mrt": r["avg_dist_mrt"],
                "avg_dist_cbd": r["avg_dist_cbd"],
                "avg_dist_school": r["avg_dist_school"],
                "avg_dist_mall": r["avg_dist_mall"],
                "avg_dist_hawker": r.get("avg_dist_hawker"),
                "avg_hawker_count_1km": r.get("avg_hawker_count_1km"),
                "avg_dist_high_demand_school": r.get("avg_dist_high_demand_school"),
                "avg_high_demand_primary_count_1km": r.get("avg_high_demand_primary_count_1km"),
                "avg_lat": r["avg_lat"],
                "avg_lng": r["avg_lng"],
            }
            for r in rows
        }
    except SupabaseError:
        return {}


@_ttl_cache(maxsize=1, ttl=3600)
def _get_district_summary_data():
    try:
        return _supabase_rpc("rpc_api_district_summary") or []
    except SupabaseError:
        return []


@_ttl_cache(maxsize=1, ttl=3600)
def _get_district_comparison_data():
    try:
        return _supabase_rpc("rpc_api_district_comparison") or []
    except SupabaseError:
        return []


def _get_prediction_map_seed_data():
    """Return town-level rows with coordinates for prediction and summary maps."""
    district_rows = [dict(row) for row in _get_district_summary_data()]
    fallback_distances = _get_town_avg_distances()

    if not district_rows:
        district_rows = [
            {
                "town": town,
                "avg_price": 0,
                "recent_avg": 0,
                "total_txns": 0,
                "recent_txns": 0,
                "lat": meta.get("avg_lat"),
                "lng": meta.get("avg_lng"),
            }
            for town, meta in sorted(fallback_distances.items())
        ]

    for row in district_rows:
        fallback = fallback_distances.get(row.get("town"), {})
        if not row.get("lat"):
            row["lat"] = fallback.get("avg_lat")
        if not row.get("lng"):
            row["lng"] = fallback.get("avg_lng")

    return [
        row for row in district_rows
        if row.get("town") and row.get("lat") is not None and row.get("lng") is not None
    ]


@_ttl_cache(maxsize=256, ttl=3600)
def _get_flat_type_breakdown_data(town, street_name="", block=""):
    town = town or ""
    street_name = street_name or ""
    block = block or ""

    try:
        return _supabase_rpc(
            "rpc_api_flat_type_breakdown",
            {
                "p_town": town or None,
                "p_street_name": street_name or None,
                "p_block": block or None,
            },
        ) or []
    except SupabaseError:
        return []


@_ttl_cache(maxsize=256, ttl=3600)
def _get_town_flat_type_appreciation_history(town, flat_type):
    town = (town or "").strip()
    flat_type = (flat_type or "").strip()
    if not town or not flat_type:
        return []
    try:
        rows = _supabase_rpc(
            "rpc_api_price_trend_simple",
            {
                "p_town": town,
                "p_flat_type": flat_type,
                "p_street_name": None,
                "p_block": None,
            },
        ) or []
    except SupabaseError:
        return []

    history = []
    for row in rows:
        year = _coerce_int(row.get("year"))
        avg_price = _coerce_float(row.get("avg_price"))
        if year is None or avg_price is None or avg_price <= 0:
            continue
        history.append({"year": year, "avg_price": avg_price})
    history.sort(key=lambda row: row["year"])
    return history


def _resolve_town_flat_type_appreciation_features(town, flat_type, prediction_year):
    history = _get_town_flat_type_appreciation_history(town, flat_type)
    if not history:
        return {
            "town_yoy_appreciation_lag1": 0.0,
            "town_5yr_cagr_lag1": 0.0,
        }

    eligible = [row for row in history if row["year"] < int(prediction_year)]
    if not eligible:
        eligible = history

    by_year = {row["year"]: row["avg_price"] for row in eligible}
    anchor_year = max(by_year)
    anchor_avg = by_year.get(anchor_year)
    prev_avg = by_year.get(anchor_year - 1)
    prev_5y_avg = by_year.get(anchor_year - 5)

    yoy = 0.0
    if anchor_avg and prev_avg and prev_avg > 0:
        yoy = (anchor_avg - prev_avg) / prev_avg

    cagr = 0.0
    if anchor_avg and prev_5y_avg and prev_5y_avg > 0:
        cagr = (anchor_avg / prev_5y_avg) ** (1 / 5) - 1

    return {
        "town_yoy_appreciation_lag1": float(yoy),
        "town_5yr_cagr_lag1": float(cagr),
    }


def _get_available_models_data(town, flat_type, street_name="", block=""):
    town = town or ""
    flat_type = flat_type or ""
    street_name = street_name or ""
    block = block or ""

    if not flat_type:
        return list(FLAT_MODELS)
    if not town:
        return list(FLAT_MODELS_BY_TYPE.get(flat_type, FLAT_MODELS))

    try:
        params = {"p_town": town, "p_flat_type": flat_type}
        if street_name or block:
            params["p_street_name"] = street_name or None
            params["p_block"] = block or None
            try:
                rows = _supabase_rpc("rpc_api_available_models", params) or []
            except SupabaseError as exc:
                if not _rpc_param_not_available(exc, "rpc_api_available_models", "p_street_name"):
                    raise
                rows = _supabase_rpc("rpc_api_available_models", {
                    "p_town": town, "p_flat_type": flat_type,
                }) or []
        else:
            rows = _supabase_rpc("rpc_api_available_models", params) or []
        return [r["flat_model"] for r in rows]
    except SupabaseError:
        return []


def _get_available_storey_ranges_data(town, flat_type, street_name="", block=""):
    """Returns individual floor numbers derived from DB storey ranges."""
    town = town or ""
    flat_type = flat_type or ""
    street_name = street_name or ""
    block = block or ""

    try:
        params = {"p_town": town or None, "p_flat_type": flat_type or None}
        if street_name or block:
            params["p_street_name"] = street_name or None
            params["p_block"] = block or None
            try:
                rows = _supabase_rpc("rpc_api_available_storey_ranges", params) or []
            except SupabaseError as exc:
                if not _rpc_param_not_available(exc, "rpc_api_available_storey_ranges", "p_street_name"):
                    raise
                rows = _supabase_rpc("rpc_api_available_storey_ranges", {
                    "p_town": town or None, "p_flat_type": flat_type or None,
                }) or []
        else:
            rows = _supabase_rpc("rpc_api_available_storey_ranges", params) or []
        floors = set()
        for r in rows:
            sr = r["storey_range"]
            if " TO " in sr:
                parts = sr.split(" TO ")
                for f in range(int(parts[0]), int(parts[1]) + 1):
                    floors.add(f)
            else:
                floors.add(int(sr))
        return [str(f) for f in sorted(floors)]
    except SupabaseError:
        return []


def _get_floor_area_stats_data(town, flat_type, street_name="", block=""):
    town = town or ""
    flat_type = flat_type or ""
    street_name = street_name or ""
    block = block or ""

    try:
        params = {"p_town": town or None, "p_flat_type": flat_type or None}
        if street_name or block:
            params["p_street_name"] = street_name or None
            params["p_block"] = block or None
            try:
                rows = _supabase_rpc("rpc_api_floor_area_stats", params) or []
            except SupabaseError as exc:
                if not _rpc_param_not_available(exc, "rpc_api_floor_area_stats", "p_street_name"):
                    raise
                rows = _supabase_rpc("rpc_api_floor_area_stats", {
                    "p_town": town or None, "p_flat_type": flat_type or None,
                }) or []
        else:
            rows = _supabase_rpc("rpc_api_floor_area_stats", params) or []
        if rows and isinstance(rows, list):
            return rows[0]
        if rows and isinstance(rows, dict):
            return rows
    except SupabaseError:
        pass

    return {"min_area": 30, "max_area": 300, "avg_area": DEFAULT_FLOOR_AREA}


def _get_lease_year_range_data(town, street_name="", block=""):
    town = town or ""
    street_name = street_name or ""
    block = block or ""

    try:
        params = {"p_town": town or None}
        if street_name or block:
            params["p_street_name"] = street_name or None
            params["p_block"] = block or None
            try:
                rows = _supabase_rpc("rpc_api_lease_year_range", params) or []
            except SupabaseError as exc:
                if not _rpc_param_not_available(exc, "rpc_api_lease_year_range", "p_street_name"):
                    raise
                rows = _supabase_rpc(
                    "rpc_api_lease_year_range", {"p_town": town or None}
                ) or []
        else:
            rows = _supabase_rpc("rpc_api_lease_year_range", params) or []
        if rows and isinstance(rows, list):
            return rows[0]
        if rows and isinstance(rows, dict):
            return rows
    except SupabaseError:
        pass

    return _default_lease_year_range()


TOWNS = _get_towns()
FLAT_MODELS = _get_flat_models()
TOWN_DISTANCES = _get_town_avg_distances()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if _session_user_id() is None:
            session.clear()
            flash("Please log in to access this feature.", "warning")
            next_url = request.full_path.rstrip("?") if request.query_string else request.path
            return redirect(url_for("login", next=next_url))
        return f(*args, **kwargs)
    return decorated


def _safe_next_url(target):
    if not target:
        return ""
    parsed = parse.urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return ""
    if not target.startswith("/"):
        return ""
    return target


def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if _session_user_id() is None:
            session.clear()
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


def premium_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if _session_user_id() is None:
            session.clear()
            flash("Please log in to access this feature.", "warning")
            return redirect(url_for("login"))
        if session.get("subscription_tier", "general") != "premium":
            flash("This feature requires a Premium subscription.", "info")
            return redirect(url_for("pricing"))
        return f(*args, **kwargs)
    return decorated


def api_premium_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if _session_user_id() is None:
            session.clear()
            return jsonify({"error": "Authentication required"}), 401
        if session.get("subscription_tier", "general") != "premium":
            return jsonify({"error": "Premium subscription required"}), 403
        return f(*args, **kwargs)
    return decorated


# Weekly view limits for general users per feature
GENERAL_WEEKLY_VIEW_LIMITS = {"map": 3, "analytics": 3, "comparison": 3}


def _get_weekly_view_count(user_id, feature):
    """Count views of a feature by user in the current week."""
    return len(_get_supabase_feature_view_rows(user_id, feature))


def _log_feature_view(user_id, feature):
    """Record a feature view."""
    _supabase_request(
        "feature_view_log",
        method="POST",
        payload={"user_id": user_id, "feature": feature},
    )


def _check_feature_limit(feature):
    """Check if general user has exceeded weekly view limit for a feature.
    Returns (allowed, views_used, views_limit)."""
    tier = session.get("subscription_tier", "general")
    if tier == "premium":
        return True, 0, 0
    limit = GENERAL_WEEKLY_VIEW_LIMITS.get(feature, 3)
    count = _get_weekly_view_count(_session_user_id(), feature)
    return count < limit, count, limit


def _log_feature_view_once(user_id, feature):
    """Log a feature view only once per session to avoid burning views on reloads."""
    session_key = f"_viewed_{feature}"
    if session.get(session_key):
        return
    _log_feature_view(user_id, feature)
    session[session_key] = True


@app.before_request
def load_user():
    g.user = None
    user_id = _session_user_id()
    if user_id is not None:
        # Reconstruct from session — no extra DB round-trip needed
        g.user = {
            "id": user_id,
            "username": session.get("username", ""),
            "email": session.get("email", ""),
            "subscription_tier": session.get("subscription_tier", "general"),
        }
    elif "user_id" in session:
        session.clear()


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

def _load_shap_module():
    global _SHAP_IMPORT_ERROR
    if _SHAP_IMPORT_ERROR is not None:
        return None
    try:
        import shap
    except Exception as exc:
        _SHAP_IMPORT_ERROR = exc
        return None
    return shap


def _get_shap_explainer():
    global _SHAP_EXPLAINER
    if ARTEFACTS.get("model_key") not in SHAP_SUPPORTED_MODEL_KEYS:
        return None
    if ARTEFACTS.get("model") is None:
        return None
    if _SHAP_EXPLAINER is not None:
        return _SHAP_EXPLAINER

    shap = _load_shap_module()
    if shap is None:
        return None

    _SHAP_EXPLAINER = shap.TreeExplainer(ARTEFACTS["model"])
    return _SHAP_EXPLAINER


def _compute_feature_contributions(feature_frame):
    model_key = ARTEFACTS.get("model_key")
    model = ARTEFACTS.get("model")

    if model_key == "xgboost" and model is not None:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        contrib_source = feature_frame
        if hasattr(booster, "predict"):
            try:
                import xgboost as xgb
                cols = list(feature_frame.columns)
                contrib_source = xgb.DMatrix(feature_frame, feature_names=cols)
            except Exception:
                contrib_source = feature_frame
        contribs = booster.predict(contrib_source, pred_contribs=True)
        contribs = np.asarray(contribs, dtype=float)
        if contribs.ndim > 2:
            contribs = contribs.reshape(contribs.shape[0], -1)
        contrib_row = contribs[0]
        return contrib_row[:-1], float(contrib_row[-1])

    explainer = _get_shap_explainer()
    if explainer is None:
        return None, None

    shap_values = explainer.shap_values(feature_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
    expected_value = _coerce_scalar(getattr(explainer, "expected_value", 0.0))
    return shap_values, expected_value


def _resolve_price_index_multiplier(year, month_num):
    target_transform = ARTEFACTS.get("target_transform", "log1p_resale_price")
    price_index = ARTEFACTS.get("price_index")
    if target_transform != "rpi_adjusted_log_price" or price_index is None:
        return 1.0

    quarter_key = int(year) * 10 + ((int(month_num) - 1) // 3 + 1)
    pi = None
    try:
        pi = price_index.get(quarter_key)
    except AttributeError:
        pi = None

    if pi is None:
        try:
            pi = price_index[quarter_key]
        except Exception:
            pi = None

    if pi is None:
        if hasattr(price_index, "loc") and hasattr(price_index, "index") and len(price_index.index):
            pi = price_index.loc[price_index.index.max()]
        elif isinstance(price_index, dict) and price_index:
            pi = price_index[max(price_index)]

    try:
        return float(pi)
    except (TypeError, ValueError):
        return 1.0


def _inverse_target_prediction(pred_log, year, month_num):
    return float(np.expm1(pred_log)) * _resolve_price_index_multiplier(year, month_num)


def _predict_log_price_from_scaled_df(df):
    fc = _serving_feature_cols()
    return float(ARTEFACTS["model"].predict(df[fc])[0])


def _build_scaled_feature_df(town, flat_type, flat_model, floor_area, storey_range,
                             lease_commence, override_year=None, override_distances=None):
    """Build a single-row scaled feature frame for prediction and explainability."""
    scaler = ARTEFACTS["scaler"]
    encoders = ARTEFACTS["encoders"]

    now = datetime.now()
    year = override_year if override_year is not None else now.year
    month_num = now.month

    flat_age = year - lease_commence
    remaining_lease = max(0, 99 - flat_age)
    month_sin = math.sin(2 * math.pi * month_num / 12)
    month_cos = math.cos(2 * math.pi * month_num / 12)
    is_mature = 1 if town in MATURE_ESTATES else 0
    flat_type_ord = FLAT_TYPE_ORDINAL.get(flat_type, 4)

    town_enc_map = encoders["town"]["means"]
    town_enc = town_enc_map.get(town, encoders["town"]["global_mean"])

    flat_model_enc_map = encoders["flat_model"]["means"]
    flat_model_enc = flat_model_enc_map.get(
        flat_model, encoders["flat_model"]["global_mean"]
    )

    storey_mid = _storey_midpoint(storey_range)

    if override_distances:
        dist_mrt = override_distances.get("dist_mrt", 500)
        dist_cbd = override_distances.get("dist_cbd", 10000)
        dist_school = override_distances.get("dist_school", 500)
        dist_mall = override_distances.get("dist_mall", 1000)
        dist_hawker = override_distances.get("dist_hawker", 1.0)
        hawker_count_1km = override_distances.get("hawker_count_1km", 0)
        dist_high_demand_school = override_distances.get("dist_high_demand_school", 500)
        high_demand_primary_count_1km = override_distances.get(
            "high_demand_primary_count_1km", 0
        )
    else:
        dists = TOWN_DISTANCES.get(town, {})
        dist_mrt = dists.get("avg_dist_mrt", 500)
        dist_cbd = dists.get("avg_dist_cbd", 10000)
        dist_school = dists.get("avg_dist_school", 500)
        dist_mall = dists.get("avg_dist_mall", 1000)
        dist_hawker = dists.get("avg_dist_hawker", 1.0)
        hawker_count_1km = dists.get("avg_hawker_count_1km", 0)
        dist_high_demand_school = dists.get("avg_dist_high_demand_school", 500)
        high_demand_primary_count_1km = dists.get(
            "avg_high_demand_primary_count_1km", 0
        )

    scale_cols = _serving_scale_cols()
    feat_cols = _serving_feature_cols()
    need_appreciation = (
        "town_yoy_appreciation_lag1" in scale_cols
        or "town_yoy_appreciation_lag1" in feat_cols
        or "town_5yr_cagr_lag1" in scale_cols
        or "town_5yr_cagr_lag1" in feat_cols
    )
    if need_appreciation:
        appreciation = _resolve_town_flat_type_appreciation_features(town, flat_type, year)
    else:
        appreciation = {
            "town_yoy_appreciation_lag1": 0.0,
            "town_5yr_cagr_lag1": 0.0,
        }

    rolling_snap = ARTEFACTS.get("rolling_stats", {})
    rolling = (
        rolling_snap.get((town, flat_type))
        or rolling_snap.get("_global_defaults")
        or {}
    )

    raw = {
        "flat_type_ordinal": flat_type_ord,
        "town_enc": town_enc,
        "flat_model_enc": flat_model_enc,
        "floor_area_sqm": floor_area,
        "storey_midpoint": storey_mid,
        "flat_age": flat_age,
        "remaining_lease": remaining_lease,
        "lease_commence_date": lease_commence,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "year": year,
        "is_mature_estate": is_mature,
        "dist_mrt": dist_mrt,
        "dist_cbd": dist_cbd,
        "dist_primary_school": dist_school,
        "dist_major_mall": dist_mall,
        "dist_hawker_centre": dist_hawker,
        "hawker_count_1km": hawker_count_1km,
        "dist_high_demand_primary_school": dist_high_demand_school,
        "high_demand_primary_count_1km": high_demand_primary_count_1km,
        "town_yoy_appreciation_lag1": appreciation["town_yoy_appreciation_lag1"],
        "town_5yr_cagr_lag1": appreciation["town_5yr_cagr_lag1"],
        "town_flattype_median_3m": rolling.get("town_flattype_median_3m", 0.0),
        "town_flattype_median_6m": rolling.get("town_flattype_median_6m", 0.0),
        "town_flattype_psf_3m": rolling.get("town_flattype_psf_3m", 0.0),
        "town_median_3m": rolling.get("town_median_3m", 0.0),
        "town_txn_volume_3m": rolling.get("town_txn_volume_3m", 0.0),
        "price_momentum_3m": rolling.get("price_momentum_3m", 0.0),
        "national_median_psf_3m": rolling.get("national_median_psf_3m", 0.0),
        "sora_3m": CURRENT_SORA_3M,
    }

    df = pd.DataFrame([raw])
    if ARTEFACTS["manifest"].get("scaling_enabled", True):
        df[scale_cols] = ARTEFACTS["scaler"].transform(df[scale_cols])

    raw_values = dict(raw)
    raw_values.update({
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "storey_range": storey_range,
        "prediction_year": year,
        "prediction_month": month_num,
    })
    return df, raw_values


def _coerce_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(arr[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_metric_number(value, suffix=""):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return suffix.strip() or "N/A"

    if number.is_integer():
        return f"{int(number):,}{suffix}"
    return f"{number:,.1f}{suffix}"


def _format_rate_label(value):
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        return "0.0%"
    return f"{pct:+.1f}%"


def _describe_storey_label(storey_range, midpoint):
    try:
        level = int(round(float(midpoint)))
    except (TypeError, ValueError):
        level = None

    if level is None:
        return f"Storey ({storey_range})" if storey_range else "Storey"

    if level <= 3:
        band = "Low floor"
    elif level <= 9:
        band = "Mid floor"
    elif level <= 18:
        band = "High floor"
    else:
        band = "Very high floor"

    detail = storey_range or str(level)
    return f"{band} ({detail})"


def _feature_label(feature_name, raw_values):
    if feature_name == "floor_area_sqm":
        return f"Floor area ({_format_metric_number(raw_values.get(feature_name), ' sqm')})"
    if feature_name == "storey_midpoint":
        return _describe_storey_label(
            raw_values.get("storey_range"),
            raw_values.get(feature_name),
        )
    if feature_name == "flat_age":
        return f"Flat age ({_format_metric_number(raw_values.get(feature_name), ' yrs')})"
    if feature_name == "remaining_lease":
        return f"Remaining lease ({_format_metric_number(raw_values.get(feature_name), ' yrs')})"
    if feature_name == "lease_commence_date":
        return f"Lease start year ({int(raw_values.get(feature_name) or 0)})"
    if feature_name == "year":
        return f"Market year ({int(raw_values.get(feature_name) or 0)})"
    if feature_name == "is_mature_estate":
        return (
            "Estate maturity (Mature estate)"
            if raw_values.get(feature_name)
            else "Estate maturity (Non-mature estate)"
        )
    if feature_name == "dist_mrt":
        return f"Distance to MRT ({_format_distance(raw_values.get(feature_name))})"
    if feature_name == "dist_cbd":
        return f"Distance to CBD ({_format_distance(raw_values.get(feature_name))})"
    if feature_name == "dist_primary_school":
        return f"Distance to school ({_format_distance(raw_values.get(feature_name))})"
    if feature_name == "dist_major_mall":
        return f"Distance to mall ({_format_distance(raw_values.get(feature_name))})"
    if feature_name == "flat_type_ordinal":
        return f"Flat type ({raw_values.get('flat_type', 'Unknown')})"
    if feature_name == "town_enc":
        return f"Town profile ({str(raw_values.get('town', 'Unknown')).title()})"
    if feature_name == "flat_model_enc":
        return f"Flat model ({raw_values.get('flat_model', 'Unknown')})"
    if feature_name == "town_yoy_appreciation_lag1":
        return f"Town 1Y appreciation ({_format_rate_label(raw_values.get(feature_name))})"
    if feature_name == "town_5yr_cagr_lag1":
        return f"Town 5Y CAGR ({_format_rate_label(raw_values.get(feature_name))})"
    if feature_name in {"month_sin", "month_cos"}:
        return "Seasonal timing"
    return feature_name.replace("_", " ").title()


def _feature_phrase(label):
    return label.split(" (", 1)[0].lower()


def _join_readable(items):
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _generate_narrative_template(features, predicted_price, town, town_avg_price=None,
                                 baseline_price=None):
    positives = [item for item in features if item["dollar_impact"] > 0]
    negatives = [item for item in features if item["dollar_impact"] < 0]
    sentences = []
    town_title = str(town or "").title()

    if town_avg_price:
        diff = int(round(predicted_price - float(town_avg_price)))
        if abs(diff) < 1000:
            sentences.append(
                f"This estimate is broadly in line with the average {town_title} resale value for this flat type."
            )
        else:
            direction = "above" if diff > 0 else "below"
            sentences.append(
                f"This estimate sits about ${abs(diff):,} {direction} the average {town_title} resale value for this flat type."
            )
    elif baseline_price is not None:
        diff = int(round(predicted_price - float(baseline_price)))
        if abs(diff) >= 1000:
            direction = "above" if diff > 0 else "below"
            sentences.append(
                f"This flat is about ${abs(diff):,} {direction} the model baseline for similar homes."
            )

    top_positives = positives[:2]
    if top_positives:
        pos_total = sum(item["dollar_impact"] for item in top_positives)
        pos_labels = [_feature_phrase(item["label"]) for item in top_positives]
        sentences.append(
            f"The strongest upward pushes come from {_join_readable(pos_labels)}, adding about ${abs(int(round(pos_total))):,} combined."
        )

    top_negatives = negatives[:2]
    if top_negatives:
        neg_total = sum(abs(item["dollar_impact"]) for item in top_negatives)
        neg_labels = [_feature_phrase(item["label"]) for item in top_negatives]
        sentences.append(
            f"The main downward pressure comes from {_join_readable(neg_labels)}, trimming roughly ${abs(int(round(neg_total))):,}."
        )

    if not sentences:
        sentences.append(
            "The feature contributions are tightly balanced, so no single factor dominates this estimate."
        )

    return " ".join(sentences)


def compute_shap_explanation(town, flat_type, flat_model, floor_area, storey_range,
                             lease_commence, predicted_price=None, override_year=None,
                             override_distances=None, town_avg_price=None):
    if ARTEFACTS.get("model_key") not in SHAP_SUPPORTED_MODEL_KEYS:
        return None

    df, raw_values = _build_scaled_feature_df(
        town,
        flat_type,
        flat_model,
        floor_area,
        storey_range,
        lease_commence,
        override_year=override_year,
        override_distances=override_distances,
    )
    fc = _serving_feature_cols()
    feature_frame = df[fc]
    shap_values, baseline_log = _compute_feature_contributions(feature_frame)
    if shap_values is None:
        return None

    pred_log = _predict_log_price_from_scaled_df(df)
    prediction_year = raw_values["prediction_year"]
    prediction_month = raw_values["prediction_month"]
    predicted_price_raw = _inverse_target_prediction(
        pred_log,
        prediction_year,
        prediction_month,
    )
    baseline_price = _inverse_target_prediction(
        baseline_log,
        prediction_year,
        prediction_month,
    )

    raw_impacts = []
    for feature_name, shap_value in zip(fc, shap_values):
        counterfactual_price = _inverse_target_prediction(
            pred_log - float(shap_value),
            prediction_year,
            prediction_month,
        )
        raw_impacts.append({
            "key": feature_name,
            "label": _feature_label(feature_name, raw_values),
            "dollar_impact": predicted_price_raw - counterfactual_price,
        })

    delta_target = predicted_price_raw - baseline_price
    raw_total = sum(item["dollar_impact"] for item in raw_impacts)
    scale = (delta_target / raw_total) if abs(raw_total) > 1e-9 else 1.0

    for item in raw_impacts:
        item["dollar_impact"] *= scale

    grouped_items = []
    grouped_map = {}
    for item in raw_impacts:
        group_key = (
            "seasonal_timing"
            if item["key"] in {"month_sin", "month_cos"}
            else item["key"]
        )
        if group_key not in grouped_map:
            grouped_map[group_key] = {
                "key": group_key,
                "label": "Seasonal timing" if group_key == "seasonal_timing" else item["label"],
                "dollar_impact": 0.0,
            }
            grouped_items.append(grouped_map[group_key])
        grouped_map[group_key]["dollar_impact"] += item["dollar_impact"]

    rounded_items = []
    for item in grouped_items:
        rounded_impact = int(round(item["dollar_impact"]))
        rounded_items.append({
            "key": item["key"],
            "label": item["label"],
            "dollar_impact": rounded_impact,
            "is_positive": rounded_impact >= 0,
        })

    rounded_items.sort(key=lambda item: abs(item["dollar_impact"]), reverse=True)

    return {
        "features": rounded_items,
        "baseline_price": int(round(baseline_price)),
        "predicted_price": int(round(
            predicted_price_raw if predicted_price is None else float(predicted_price)
        )),
        "delta_from_baseline": int(round(predicted_price_raw - baseline_price)),
        "narrative": _generate_narrative_template(
            rounded_items,
            predicted_price_raw,
            town,
            town_avg_price=town_avg_price,
            baseline_price=baseline_price,
        ),
        "model_note": None,
        "model_label": ARTEFACTS.get("model_label", "Model"),
        "feature_count": len(rounded_items),
    }


def predict_price(town, flat_type, flat_model, floor_area, storey_range,
                  lease_commence, override_year=None, override_distances=None):
    """
    Run the full feature engineering + prediction pipeline for a single property.
    Returns dict with predicted_price and model_label.
    """
    df, raw_values = _build_scaled_feature_df(
        town,
        flat_type,
        flat_model,
        floor_area,
        storey_range,
        lease_commence,
        override_year=override_year,
        override_distances=override_distances,
    )
    pred_log = _predict_log_price_from_scaled_df(df)
    predicted_price = _inverse_target_prediction(
        pred_log,
        raw_values["prediction_year"],
        raw_values["prediction_month"],
    )

    performance = ARTEFACTS.get("performance", {})

    return _enrich_prediction_result(
        predicted_price,
        prediction_year=raw_values.get("prediction_year"),
        result={
            "model_label": performance.get("label", ARTEFACTS.get("model_label", "Model")),
        },
    )


def _get_recent_similar_transactions(
    town,
    flat_type,
    limit=5,
    street_name="",
    block="",
):
    """Return recent transactions for same town + flat_type, optionally scoped."""
    try:
        rows = _supabase_rpc("rpc_recent_similar_transactions", {
            "p_town": town,
            "p_flat_type": flat_type,
            "p_limit": limit,
            "p_street_name": street_name or None,
            "p_block": block or None,
        }) or []
        return rows
    except SupabaseError:
        return []


def _get_block_distances(town, street_name, block):
    """Look up actual distances for a specific block."""
    try:
        rows = _supabase_rpc("rpc_block_distances", {
            "p_town": town,
            "p_street": street_name,
            "p_block": block,
        }) or []
        if rows:
            r = rows[0]
            return {
                "dist_mrt": r.get("dist_mrt"),
                "dist_cbd": r.get("dist_cbd"),
                "dist_school": r.get("dist_school"),
                "dist_mall": r.get("dist_mall"),
                "dist_hawker": r.get("dist_hawker"),
                "hawker_count_1km": r.get("hawker_count_1km"),
                "dist_high_demand_school": r.get("dist_high_demand_school"),
                "high_demand_primary_count_1km": r.get("high_demand_primary_count_1km"),
            }
    except SupabaseError:
        return None
    return None


def _resolve_prediction_inputs(
    town,
    flat_type,
    floor_area_raw,
    lease_commence_raw,
    street_name="",
    block="",
):
    """
    Resolve optional prediction inputs.
    If floor_area or lease_commence is missing, infer from historical averages.
    """
    assumptions = []
    street_name = street_name or ""
    block = block or ""

    floor_area = None
    if floor_area_raw:
        floor_area = float(floor_area_raw)
    else:
        try:
            v = _supabase_rpc(
                "rpc_resolve_floor_area",
                {
                    "p_town": town,
                    "p_flat_type": flat_type,
                    "p_street_name": street_name or None,
                    "p_block": block or None,
                },
            )
            floor_area = float(v) if v else float(DEFAULT_FLOOR_AREA)
        except SupabaseError:
            floor_area = float(DEFAULT_FLOOR_AREA)
        assumptions.append(f"Used inferred floor area: {floor_area} sqm")

    lease_commence = None
    if lease_commence_raw:
        lease_commence = int(lease_commence_raw)
    else:
        try:
            v = _supabase_rpc(
                "rpc_resolve_lease_commence",
                {
                    "p_town": town,
                    "p_flat_type": flat_type,
                    "p_street_name": street_name or None,
                    "p_block": block or None,
                },
            )
            lease_commence = int(v) if v else _default_lease_year_range()["avg_year"]
        except SupabaseError:
            lease_commence = _default_lease_year_range()["avg_year"]
        assumptions.append(f"Used inferred lease start year: {lease_commence}")

    return floor_area, lease_commence, assumptions


def _default_flat_model_for_type(flat_type):
    candidates = list(FLAT_MODELS_BY_TYPE.get(flat_type, []))
    if not candidates:
        return FLAT_MODELS[0] if FLAT_MODELS else "Model A"

    preferred_order = [
        "Model A",
        "Improved",
        "New Generation",
        "Apartment",
        "Standard",
        "Premium Apartment",
    ]
    candidate_lookup = {str(model).strip().upper(): model for model in candidates}
    for preferred in preferred_order:
        match = candidate_lookup.get(preferred.upper())
        if match:
            return match
    return candidates[0]


def _resolve_prediction_flat_model(town, flat_type, flat_model=""):
    requested_model = str(flat_model or "").strip()
    available_models = _get_available_models_data(town, flat_type)

    if requested_model:
        matched_model = next(
            (
                model for model in available_models
                if str(model).strip().upper() == requested_model.upper()
            ),
            None,
        )
        if matched_model:
            return matched_model, []
        if not available_models:
            return requested_model, []

    if available_models:
        resolved_model = available_models[0]
        if requested_model:
            return resolved_model, [
                f"Adjusted flat model to {resolved_model} for the selected town and flat type."
            ]
        return resolved_model, [f"Used representative flat model: {resolved_model}"]

    fallback_model = requested_model or _default_flat_model_for_type(flat_type)
    if requested_model:
        return fallback_model, []
    return fallback_model, [f"Used fallback flat model: {fallback_model}"]


def _resolve_prediction_storey(town, flat_type, storey_range=""):
    requested_storey = str(storey_range or "").strip()
    available_storeys = (
        _get_available_storey_ranges_data(town, flat_type)
        or _get_available_storey_ranges_data(town, "")
    )

    if requested_storey and requested_storey in available_storeys:
        return requested_storey, []

    requested_floor = None
    if requested_storey:
        try:
            requested_floor = int(round(_storey_midpoint(requested_storey)))
        except (TypeError, ValueError):
            requested_floor = _coerce_int(requested_storey)

    if available_storeys:
        available_floors = [
            int(value) for value in available_storeys if str(value).isdigit()
        ]
        if available_floors:
            if requested_floor is not None:
                resolved_floor = min(
                    available_floors,
                    key=lambda floor: (abs(floor - requested_floor), floor),
                )
                if requested_storey and str(resolved_floor) == requested_storey:
                    return str(resolved_floor), []
                return str(resolved_floor), [
                    f"Used nearest available floor: {resolved_floor}"
                ]

            resolved_floor = available_floors[len(available_floors) // 2]
            return str(resolved_floor), [f"Used representative floor: {resolved_floor}"]

        resolved_storey = available_storeys[len(available_storeys) // 2]
        if requested_storey and requested_storey == resolved_storey:
            return resolved_storey, []
        note = (
            f"Used nearest available floor: {resolved_storey}"
            if requested_storey
            else f"Used representative floor: {resolved_storey}"
        )
        return resolved_storey, [note]

    if requested_floor is not None:
        if requested_storey and str(requested_floor) == requested_storey:
            return str(requested_floor), []
        return str(requested_floor), [
            f"Converted storey range to representative floor: {requested_floor}"
        ]

    fallback_storey = STOREY_RANGES[len(STOREY_RANGES) // 2] if STOREY_RANGES else "8"
    return fallback_storey, [f"Used representative floor: {fallback_storey}"]


def _complete_prediction_form_data(form_data, infer_flat_type=False):
    raw_form = form_data or {}
    resolved_form = {
        "town": str(raw_form.get("town", "") or "").strip(),
        "flat_type": str(raw_form.get("flat_type", "") or "").strip(),
        "flat_model": str(raw_form.get("flat_model", "") or "").strip(),
        "floor_area": raw_form.get("floor_area", ""),
        "storey_range": str(raw_form.get("storey_range", "") or "").strip(),
        "lease_commence": raw_form.get("lease_commence", ""),
        "street_name": str(raw_form.get("street_name", "") or "").strip(),
        "block": str(raw_form.get("block", "") or "").strip(),
    }
    assumptions = []

    town = resolved_form["town"]
    if not town:
        return resolved_form, assumptions

    flat_type = resolved_form["flat_type"]
    if infer_flat_type or not flat_type:
        flat_type, flat_type_assumptions = _resolve_forecast_flat_type(
            town,
            flat_type,
            street_name=resolved_form["street_name"],
            block=resolved_form["block"],
        )
        resolved_form["flat_type"] = flat_type
        assumptions.extend(flat_type_assumptions)

    if not resolved_form["flat_type"]:
        return resolved_form, assumptions

    floor_area, lease_commence, input_assumptions = _resolve_prediction_inputs(
        town,
        resolved_form["flat_type"],
        str(resolved_form.get("floor_area", "")).strip(),
        str(resolved_form.get("lease_commence", "")).strip(),
        resolved_form["street_name"],
        resolved_form["block"],
    )
    resolved_form["floor_area"] = floor_area
    resolved_form["lease_commence"] = lease_commence
    assumptions.extend(input_assumptions)

    flat_model, model_assumptions = _resolve_prediction_flat_model(
        town,
        resolved_form["flat_type"],
        resolved_form.get("flat_model", ""),
    )
    resolved_form["flat_model"] = flat_model
    assumptions.extend(model_assumptions)

    storey_range, storey_assumptions = _resolve_prediction_storey(
        town,
        resolved_form["flat_type"],
        resolved_form.get("storey_range", ""),
    )
    resolved_form["storey_range"] = storey_range
    assumptions.extend(storey_assumptions)

    return resolved_form, assumptions


def _resolve_forecast_flat_type(town, flat_type, street_name="", block=""):
    """Choose a flat type for analytics forecasts when the filter is broad."""
    flat_type = (flat_type or "").strip()
    if flat_type:
        return flat_type, []

    breakdown = _get_flat_type_breakdown_data(town, street_name, block)
    ranked = sorted(
        (row for row in breakdown if row.get("flat_type")),
        key=lambda row: (-int(row.get("txn_count") or 0), row.get("flat_type")),
    )
    if ranked:
        resolved_flat_type = ranked[0]["flat_type"]
        return resolved_flat_type, [f"Used representative flat type: {resolved_flat_type}"]

    return "4 Room", ["Used representative flat type: 4 Room"]


def _pick_representative_storey(storey_ranges):
    floors = sorted(
        int(value) for value in (storey_ranges or [])
        if str(value).isdigit()
    )
    if not floors:
        return ""
    return str(floors[len(floors) // 2])


@_ttl_cache(maxsize=64, ttl=3600)
def _infer_map_prediction_profile(town):
    """Infer a representative model input profile from a town's own data."""
    resolved_form, _ = _complete_prediction_form_data(
        {"town": town},
        infer_flat_type=True,
    )
    return {
        "flat_type": resolved_form["flat_type"],
        "flat_model": resolved_form["flat_model"],
        "floor_area": _coerce_float(resolved_form["floor_area"], float(DEFAULT_FLOOR_AREA)),
        "storey_range": resolved_form["storey_range"],
        "lease_commence": _coerce_int(
            resolved_form["lease_commence"],
            _default_lease_year_range()["avg_year"],
        ),
    }


# ---------------------------------------------------------------------------
# Routes: Auth
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        try:
            result = _supabase_auth("/signup", payload={
                "email": email,
                "password": password,
                "data": {"username": username},
            })
        except SupabaseError as exc:
            msg = str(exc)
            if "already registered" in msg or "already exists" in msg:
                flash("An account with that email already exists.", "danger")
            else:
                flash(f"Registration failed: {exc}", "danger")
            return render_template("register.html")

        # Also write to public.users so saved_predictions integer FK keeps working
        try:
            rows = _supabase_request(
                SUPABASE_USERS_TABLE,
                method="POST",
                payload={"username": username, "email": email, "password_hash": "supabase-auth"},
                prefer="return=representation",
            )
            db_user = rows[0] if rows else {}
        except SupabaseError:
            db_user = _get_supabase_user_by_email(email) or {}

        if result.get("access_token"):
            if not db_user.get("id"):
                flash("Account created, but the app profile could not be provisioned in Supabase. Please contact support before logging in.", "danger")
                return redirect(url_for("login"))
            session["user_id"] = db_user.get("id")
            session["username"] = username
            session["email"] = email
            session["access_token"] = result["access_token"]
            session["subscription_tier"] = "general"
            flash("Account created! Welcome.", "success")
            return redirect(url_for("home"))

        flash("Account created! Check your email to confirm before logging in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = _safe_next_url(
        request.form.get("next", "") if request.method == "POST" else request.args.get("next", "")
    )

    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        try:
            result = _supabase_auth(
                "/token?grant_type=password",
                payload={"email": email, "password": password},
            )
        except SupabaseError:
            flash("Invalid email or password.", "danger")
            return render_template("login.html", next_url=next_url)

        # Fetch public.users record for integer ID (used by saved_predictions FK)
        try:
            db_user = _get_supabase_user_by_email(email) or {}
        except SupabaseError:
            db_user = {}

        auth_user = result.get("user") or {}
        if not db_user.get("id"):
            session.clear()
            flash("Your account authenticated with Supabase, but the app user profile is missing. Please contact support.", "danger")
            return render_template("login.html", next_url=next_url)
        session["user_id"] = db_user.get("id")
        session["username"] = db_user.get("username") or auth_user.get("user_metadata", {}).get("username", email.split("@")[0])
        session["email"] = email
        session["access_token"] = result.get("access_token", "")
        session["subscription_tier"] = db_user.get("subscription_tier", "general")
        flash(f"Welcome back, {session['username']}!", "success")
        return redirect(next_url or url_for("home"))

    return render_template("login.html", next_url=next_url)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        try:
            _supabase_auth("/recover", payload={"email": email})
        except SupabaseError:
            pass  # Don't reveal whether the email exists
        flash("If that email is registered, you'll receive a password reset link.", "info")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")


@app.route("/logout")
def logout():
    token = session.get("access_token")
    if token:
        try:
            _supabase_auth("/logout", access_token=token)
        except SupabaseError:
            pass
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Routes: Subscription
# ---------------------------------------------------------------------------

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")


@app.route("/upgrade", methods=["POST"])
@login_required
def upgrade():
    user_id = _session_user_id()
    try:
        _supabase_request(
            SUPABASE_USERS_TABLE,
            method="PATCH",
            filters={"id": f"eq.{user_id}"},
            payload={"subscription_tier": "premium"},
        )
    except SupabaseError:
        flash("Could not upgrade via Supabase.", "danger")
        return redirect(url_for("pricing"))
    session["subscription_tier"] = "premium"
    flash("You've been upgraded to Premium! Enjoy unlimited access.", "success")
    return redirect(url_for("pricing"))


# ---------------------------------------------------------------------------
# Routes: Pages
# ---------------------------------------------------------------------------

def _get_popular_predictions(limit=3):
    """Return the most common town+flat_type prediction combos across all users."""
    try:
        # Cap the fetch to the 500 most recent predictions to avoid full table scans
        rows = _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            filters={
                "select": "town,flat_type,predicted_price",
                "order": "created_at.desc",
                "limit": "500",
            },
        ) or []
        aggregates = {}
        for row in rows:
            key = (row.get("town"), row.get("flat_type"))
            if not all(key):
                continue
            bucket = aggregates.setdefault(key, {"sum": 0.0, "count": 0})
            bucket["sum"] += float(row.get("predicted_price") or 0)
            bucket["count"] += 1

        ranked = sorted(
            (
                {
                    "town": town,
                    "flat_type": flat_type,
                    "avg_price": round(data["sum"] / data["count"]),
                    "count": data["count"],
                }
                for (town, flat_type), data in aggregates.items()
                if data["count"] > 0
            ),
            key=lambda item: item["count"],
            reverse=True,
        )
        return ranked[:limit]
    except SupabaseError:
        return []


@app.route("/")
def landing():
    """Public marketing landing page."""
    return render_template("landing.html", landing_stats=_build_landing_stats())


@app.route("/home")
def home():
    # Total transaction count
    try:
        total_txns = _supabase_count("transactions")
    except Exception:
        total_txns = None

    total_txns_display = f"{total_txns:,}" if total_txns is not None else "N/A"

    performance = ARTEFACTS.get("performance", {})
    artefact_mape = performance.get("test_mape_display")

    # Popular / personalized predictions for homepage cards
    popular_predictions = []
    is_personalized = False
    if g.user:
        try:
            user_preds = _prepare_saved_predictions(
                _get_saved_predictions(session["user_id"])
            )
            if user_preds:
                popular_predictions = user_preds[:3]
                is_personalized = True
            else:
                popular_predictions = _get_popular_predictions()
        except Exception:
            popular_predictions = _get_popular_predictions()
    else:
        popular_predictions = _get_popular_predictions()

    # Town coordinates for map thumbnails
    town_coords = {}
    try:
        for d in _get_district_summary_data():
            if d.get("lat") and d.get("lng"):
                town_coords[d["town"]] = {"lat": d["lat"], "lng": d["lng"]}
    except Exception:
        pass

    return render_template(
        "home.html",
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        storey_ranges=STOREY_RANGES,
        total_txns=total_txns,
        total_txns_display=total_txns_display,
        artefact_mape=artefact_mape,
        active_model_performance=performance,
        popular_predictions=popular_predictions,
        is_personalized=is_personalized,
        town_coords=town_coords,
    )


@app.route("/comparison", methods=["GET", "POST"])
@login_required
def comparison():
    allowed, _, limit = _check_feature_limit("comparison")
    if not allowed:
        flash(f"You've used all {limit} free Comparison views this week. Upgrade to Premium for unlimited access.", "warning")
        return redirect(url_for("pricing"))
    _log_feature_view_once(session["user_id"], "comparison")
    saved_predictions = []
    if g.user:
        try:
            saved_predictions = _prepare_saved_predictions(_get_saved_predictions(session["user_id"]))
        except SupabaseError:
            flash("Could not load saved predictions from Supabase.", "danger")

    is_premium = session.get("subscription_tier", "general") == "premium"
    max_panels = _comparison_max_panels()

    selected_saved_ids = _get_comparison_saved_prediction_ids() if g.user else []

    # Determine how many panels to show
    is_add_or_remove = False
    if request.method == "POST":
        panel_count = int(request.form.get("panel_count", 2))
        if request.form.get("add_panel"):
            is_add_or_remove = True
        if request.form.get("remove_panel") is not None:
            is_add_or_remove = True
    else:
        # Check query param panel_count first (from saved predictions page)
        panel_count = int(request.args.get("panel_count", 0)) or max(2, len(selected_saved_ids))
    panel_count = max(2, min(panel_count, max_panels))

    # Build panels data
    panels = []
    for i in range(panel_count):
        prefix = f"p{i}"
        label = chr(ord("A") + i)  # A, B, C, D, E

        saved_id = request.values.get(f"{prefix}_id", "").strip() or (
            str(selected_saved_ids[i]) if i < len(selected_saved_ids) else ""
        )
        saved = _get_saved_prediction_by_id(saved_predictions, saved_id)
        form_data = _prediction_form_from_saved(saved)
        result = None

        if request.method == "POST":
            form_data = _extract_prediction_form_data(request.form, prefix, seed=form_data)
        else:
            form_data = _extract_prediction_form_data(request.args, prefix, seed=form_data)

        panels.append({
            "index": i,
            "prefix": prefix,
            "label": label,
            "prefilled": bool(saved),
            "form_data": form_data,
            "result": result,
        })

    # Run predictions (skip if just adding/removing a panel)
    should_compare = not is_add_or_remove and (
        request.method == "POST" or all(
            _get_saved_prediction_by_id(saved_predictions,
                request.values.get(f"p{i}_id", "").strip() or
                (str(selected_saved_ids[i]) if i < len(selected_saved_ids) else ""))
            for i in range(panel_count)
        )
    )
    all_have_town = all(p["form_data"].get("town") for p in panels)

    payloads = []
    if should_compare and all_have_town:
        for p in panels:
            resolved_form, result, payload = _run_prediction_form(p["form_data"])
            p["form_data"] = resolved_form
            p["result"] = result
            payloads.append(payload)

    # Build unified comparison analysis across all properties
    comparison_analysis = _build_comparison_analysis(payloads) if len(payloads) >= 2 else None

    return render_template(
        "comparison.html",
        saved_predictions=saved_predictions,
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        storey_ranges=STOREY_RANGES,
        panels=panels,
        panel_count=panel_count,
        max_panels=max_panels,
        is_premium=is_premium,
        comparison_analysis=comparison_analysis,
    )


@app.route("/comparison/select/<int:pred_id>")
@login_required
def comparison_select_saved(pred_id):
    try:
        predictions = _prepare_saved_predictions(_get_saved_predictions(session["user_id"]))
    except SupabaseError:
        flash("Could not load saved predictions from Supabase.", "danger")
        return redirect(url_for("my_predictions"))

    selected_prediction = _get_saved_prediction_by_id(predictions, pred_id)
    if not selected_prediction:
        flash("That saved prediction could not be found.", "warning")
        return redirect(url_for("my_predictions"))

    updated_ids = _push_comparison_saved_prediction_id(pred_id)
    max_panels = _comparison_max_panels()
    if len(updated_ids) < max_panels:
        remaining = max_panels - len(updated_ids)
        flash(f"Saved prediction added to comparison. You can add {remaining} more.", "info")
    else:
        flash(f"Saved prediction added. All {max_panels} comparison slots are now filled.", "success")

    return redirect(url_for("comparison"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    is_premium = session.get("subscription_tier", "general") == "premium"
    prefill_source = request.args.get("source", "").strip()
    should_auto_predict = (
        request.method == "GET"
        and request.args.get("auto_predict", "").strip() == "1"
    )
    form_data = {
        "town": request.args.get("town", ""),
        "flat_type": request.args.get("flat_type", ""),
        "flat_model": request.args.get("flat_model", ""),
        "floor_area": request.args.get("floor_area", ""),
        "storey_range": request.args.get("storey_range", ""),
        "lease_commence": request.args.get("lease_commence", ""),
        "street_name": request.args.get("street_name", ""),
        "block": request.args.get("block", ""),
    }
    timeline = None
    flat_age = None
    remaining_lease = None
    town_avg_price = None
    recent_transactions = None
    prediction_input = None
    explanation = None

    if request.method == "POST":
        form_data = {
            "town": request.form.get("town", "").strip(),
            "flat_type": request.form.get("flat_type", "").strip(),
            "flat_model": request.form.get("flat_model", "").strip(),
            "floor_area": request.form.get("floor_area", "").strip(),
            "storey_range": request.form.get("storey_range", "").strip(),
            "lease_commence": request.form.get("lease_commence", "").strip(),
            "street_name": request.form.get("street_name", "").strip(),
            "block": request.form.get("block", "").strip(),
        }

        if not form_data["town"] or not form_data["flat_type"]:
            flash("Cannot get estimate. Please select a town and flat type.", "warning")
            return render_template(
                "predict.html",
                result=None,
                form_data=form_data,
                towns=TOWNS,
                flat_types=list(FLAT_TYPE_ORDINAL.keys()),
                flat_models=FLAT_MODELS,
                storey_ranges=STOREY_RANGES,
                timeline=None,
                flat_age=None,
                remaining_lease=None,
                town_avg_price=None,
                recent_transactions=None,
                prefill_source=prefill_source,
                explanation=None,
                is_premium=is_premium,
            )

        if form_data["flat_type"] not in FLAT_TYPE_ORDINAL:
            flash("Cannot get estimate for this flat type.", "warning")
            return render_template(
                "predict.html",
                result=None,
                form_data=form_data,
                towns=TOWNS,
                flat_types=list(FLAT_TYPE_ORDINAL.keys()),
                flat_models=FLAT_MODELS,
                storey_ranges=STOREY_RANGES,
                timeline=None,
                flat_age=None,
                remaining_lease=None,
                town_avg_price=None,
                recent_transactions=None,
                prefill_source=prefill_source,
                explanation=None,
                is_premium=is_premium,
            )
        prediction_input = dict(form_data)
    elif form_data["town"]:
        if should_auto_predict and form_data["flat_type"] in FLAT_TYPE_ORDINAL:
            prediction_input = dict(form_data)
        else:
            form_data, _ = _complete_prediction_form_data(form_data)

    if prediction_input is not None:
        form_data, result, _ = _run_prediction_form(prediction_input)

        block_distances = None
        if form_data["block"] and form_data["street_name"]:
            block_distances = _get_block_distances(
                form_data["town"], form_data["street_name"], form_data["block"]
            )

        # Timeline: predict for 1-5 years ahead
        current_year = datetime.now().year
        timeline = [{
            **result,
            "year": current_year,
            "remaining_lease": max(0, 99 - (current_year - form_data["lease_commence"])),
        }]
        for y_offset in range(1, 6):
            future_year = current_year + y_offset
            fp = predict_price(
                form_data["town"], form_data["flat_type"], form_data["flat_model"],
                form_data["floor_area"], form_data["storey_range"],
                form_data["lease_commence"], override_year=future_year,
                override_distances=block_distances,
            )
            fp["year"] = future_year
            fp["remaining_lease"] = max(0, 99 - (future_year - form_data["lease_commence"]))
            timeline.append(fp)

        # Extra context
        flat_age = current_year - form_data["lease_commence"]
        remaining_lease = max(0, 99 - flat_age)

        # Town average for this flat type
        town_avg_price = None
        breakdown = _get_flat_type_breakdown_data(form_data["town"])
        for entry in breakdown:
            if entry.get("flat_type") == form_data["flat_type"]:
                town_avg_price = entry.get("avg_price")
                break

        recent_transactions = _get_recent_similar_transactions(
            form_data["town"],
            form_data["flat_type"],
            street_name=form_data.get("street_name", ""),
            block=form_data.get("block", ""),
        )

        if is_premium:
            try:
                explanation = compute_shap_explanation(
                    town=form_data["town"],
                    flat_type=form_data["flat_type"],
                    flat_model=form_data["flat_model"],
                    floor_area=form_data["floor_area"],
                    storey_range=form_data["storey_range"],
                    lease_commence=form_data["lease_commence"],
                    predicted_price=result["predicted_price"],
                    override_distances=block_distances,
                    town_avg_price=town_avg_price,
                )
            except Exception:
                app.logger.warning("SHAP explanation failed", exc_info=True)

    return render_template(
        "predict.html",
        result=result,
        form_data=form_data,
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        storey_ranges=STOREY_RANGES,
        timeline=timeline,
        flat_age=flat_age,
        remaining_lease=remaining_lease,
        town_avg_price=town_avg_price,
        recent_transactions=recent_transactions,
        prefill_source=prefill_source,
        explanation=explanation,
        is_premium=is_premium,
    )


@app.route("/save_prediction", methods=["POST"])
@login_required
def save_prediction():
    # Enforce save limit for general users
    tier = session.get("subscription_tier", "general")
    if tier != "premium":
        existing = _get_saved_predictions(session["user_id"])
        if len(existing) >= 3:
            flash("Free users can save up to 3 predictions. Upgrade to Premium for unlimited saves.", "warning")
            return redirect(url_for("my_predictions"))

    prediction = {
        "town": request.form["town"],
        "flat_type": request.form["flat_type"],
        "flat_model": request.form["flat_model"],
        "floor_area": float(request.form["floor_area"]),
        "storey_range": request.form["storey_range"],
        "lease_commence": int(request.form["lease_commence"]),
        "predicted_price": float(request.form["predicted_price"]),
        "street_name": request.form.get("street_name", "").strip(),
        "block": request.form.get("block", "").strip(),
    }
    enriched_prediction = _enrich_prediction_result(prediction["predicted_price"])
    prediction["price_low"] = enriched_prediction["price_low"]
    prediction["price_high"] = enriched_prediction["price_high"]
    try:
        _save_prediction_record(session["user_id"], prediction)
        flash("Prediction saved!", "success")
    except SupabaseError:
        flash("Could not save prediction to Supabase.", "danger")
    return redirect(url_for("my_predictions"))


@app.route("/my_predictions")
@login_required
def my_predictions():
    try:
        preds = _prepare_saved_predictions(_get_saved_predictions(session["user_id"]))
    except SupabaseError:
        flash("Could not load saved predictions from Supabase.", "danger")
        preds = []
    return render_template("my_predictions.html", predictions=preds)


@app.route("/delete_prediction/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    try:
        _delete_saved_prediction(pred_id, session["user_id"])
        flash("Prediction deleted.", "info")
    except SupabaseError:
        flash("Could not delete prediction from Supabase.", "danger")
    return redirect(url_for("my_predictions"))


@app.route("/my_predictions/bulk_delete", methods=["POST"])
@login_required
def bulk_delete_predictions():
    ids = request.form.getlist("ids")
    deleted = 0
    for pred_id in ids:
        try:
            _delete_saved_prediction(int(pred_id), session["user_id"])
            deleted += 1
        except (SupabaseError, ValueError):
            continue
    if deleted:
        flash(f"Deleted {deleted} prediction(s).", "info")
    return redirect(url_for("my_predictions"))


@app.route("/map")
@login_required
def map_view():
    allowed, _, limit = _check_feature_limit("map")
    if not allowed:
        flash(f"You've used all {limit} free Map views this week. Upgrade to Premium for unlimited access.", "warning")
        return redirect(url_for("pricing"))
    _log_feature_view_once(session["user_id"], "map")
    return render_template(
        "map.html",
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        flat_models_by_type=FLAT_MODELS_BY_TYPE,
        storey_ranges=STOREY_RANGES,
        map_storey_ranges=MAP_STOREY_RANGE_OPTIONS,
        map_transaction_start_year=MAP_TRANSACTION_START_YEAR,
        hawker_centres=_load_reference_points("hawker_centres.json"),
        high_demand_primary_schools=_load_reference_points("high_demand_primary_schools.json"),
    )


@app.route("/analytics")
@login_required
def analytics():
    allowed, _, limit = _check_feature_limit("analytics")
    if not allowed:
        flash(f"You've used all {limit} free Analytics views this week. Upgrade to Premium for unlimited access.", "warning")
        return redirect(url_for("pricing"))
    _log_feature_view_once(session["user_id"], "analytics")
    is_premium = session.get("subscription_tier", "general") == "premium"
    return render_template("analytics.html", towns=TOWNS, is_premium=is_premium,
                           ai_daily_limit=GENERAL_DAILY_AI_ANSWER_LIMIT)


# ---------------------------------------------------------------------------
# API endpoints (JSON) for AJAX calls from frontend
# ---------------------------------------------------------------------------

@app.route("/api/transactions")
@api_login_required
def api_transactions():
    """Return recent transactions with lat/lng for map pins."""
    town = request.args.get("town", "")
    limit = max(1, min(_coerce_int(request.args.get("limit", 500), 500), MAP_TRANSACTION_LIMIT))
    min_year = _coerce_int(request.args.get("min_year"))

    try:
        rpc_params = {
            "p_town": town or None,
            "p_limit": limit,
        }

        if min_year is not None:
            try:
                rows = _supabase_rpc("rpc_api_transactions", {
                    **rpc_params,
                    "p_min_year": min_year,
                }) or []
            except SupabaseError as exc:
                if not _rpc_param_not_available(exc, "rpc_api_transactions", "p_min_year"):
                    raise
                rows = _supabase_rpc("rpc_api_transactions", rpc_params) or []
                rows = [
                    row for row in rows
                    if _coerce_int((row or {}).get("year")) is not None
                    and _coerce_int((row or {}).get("year")) >= min_year
                ]
        else:
            rows = _supabase_rpc("rpc_api_transactions", rpc_params) or []
        return jsonify(rows)
    except SupabaseError:
        return jsonify([])


@app.route("/api/district_summary")
@api_login_required
def api_district_summary():
    """Return per-town summary stats for district heatmap."""
    return jsonify(_get_prediction_map_seed_data())


@app.route("/api/predicted_heatmap")
@api_login_required
def api_predicted_heatmap():
    """Return model-based town estimates using inputs inferred from each town."""
    district_data = _get_prediction_map_seed_data()
    if not district_data:
        return jsonify({"error": "Town map data is currently unavailable."}), 503

    scenario_input = {
        "flat_type": request.args.get("flat_type", "").strip(),
        "flat_model": request.args.get("flat_model", "").strip(),
        "floor_area": request.args.get("floor_area", "").strip(),
        "storey_range": request.args.get("storey_range", "").strip(),
        "lease_commence": request.args.get("lease_commence", "").strip(),
    }
    use_scenario = any(scenario_input.values())

    comparison_by_town = {
        row["town"]: dict(row)
        for row in _get_district_comparison_data()
        if row.get("town")
    }
    results = []

    for d in district_data:
        town = d["town"]
        if not d.get("lat") or not d.get("lng"):
            continue

        if use_scenario:
            profile, _ = _complete_prediction_form_data(
                {"town": town, **scenario_input},
                infer_flat_type=True,
            )
        else:
            profile = _infer_map_prediction_profile(town)
        latest_row = comparison_by_town.get(town, {})
        latest_avg = _safe_metric(latest_row.get("avg_price"))
        recent_avg = _safe_metric(d.get("recent_avg"))
        historical_avg = _safe_metric(d.get("avg_price"))

        try:
            pred = predict_price(
                town,
                profile["flat_type"],
                profile["flat_model"],
                profile["floor_area"],
                profile["storey_range"],
                profile["lease_commence"],
            )
        except Exception:
            fallback_estimate = next(
                (
                    value for value in (latest_avg, recent_avg, historical_avg)
                    if value is not None and value > 0
                ),
                0.0,
            )
            pred = {
                "predicted_price": round(fallback_estimate),
            }

        comparison_values = [
            value for value in (latest_avg, recent_avg, historical_avg)
            if value is not None and value > 0
        ]
        market_low = min(comparison_values) if comparison_values else pred["predicted_price"]
        market_high = max(comparison_values) if comparison_values else pred["predicted_price"]

        results.append({
            "town": town,
            "lat": d["lat"],
            "lng": d["lng"],
            "predicted_price": pred["predicted_price"],
            "avg_price": round(historical_avg) if historical_avg is not None else 0,
            "recent_avg": round(recent_avg) if recent_avg is not None else 0,
            "latest_avg": round(latest_avg) if latest_avg is not None else 0,
            "latest_txns": _coerce_int(latest_row.get("txn_count"), 0) or 0,
            "total_txns": _coerce_int(d.get("total_txns"), 0) or 0,
            "market_low": round(market_low),
            "market_high": round(market_high),
            "flat_type": profile["flat_type"],
            "flat_model": profile["flat_model"],
            "storey_range": profile["storey_range"],
            "floor_area": round(_coerce_float(profile["floor_area"], DEFAULT_FLOOR_AREA), 1),
            "lease_commence": _coerce_int(profile["lease_commence"], _default_lease_year_range()["avg_year"]),
        })

    return jsonify(results)


@app.route("/api/price_trend")
@api_login_required
def api_price_trend():
    """Return yearly price trend data."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")

    try:
        rows = _supabase_rpc("rpc_api_price_trend_simple", {
            "p_town": town or None,
            "p_flat_type": flat_type or None,
        }) or []
        normalized = []
        for row in rows:
            item = dict(row)
            item["q1"] = item.get("min_price")
            item["q3"] = item.get("max_price")
            normalized.append(item)
        return jsonify(normalized)
    except SupabaseError:
        return jsonify([])


@app.route("/api/price_trend_simple")
@api_login_required
def api_price_trend_simple():
    """Yearly average price trend."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")

    try:
        return jsonify(_supabase_rpc("rpc_api_price_trend_simple", {
            "p_town": town or None,
            "p_flat_type": flat_type or None,
            "p_street_name": street_name or None,
            "p_block": block or None,
        }) or [])
    except SupabaseError:
        return jsonify([])


@app.route("/api/street_price_trend")
@api_login_required
def api_street_price_trend():
    """Return yearly price trends grouped by street within a town."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")

    if not town:
        return jsonify({"error": "town is required"}), 400

    try:
        return jsonify(_supabase_rpc("rpc_api_street_price_trend", {
            "p_town": town,
            "p_flat_type": flat_type or None,
            "p_street_name": street_name or None,
            "p_block": block or None,
        }) or [])
    except SupabaseError:
        return jsonify([])


@app.route("/api/district_comparison")
@api_login_required
def api_district_comparison():
    """Return per-town avg prices for the most recent year."""
    try:
        return jsonify(_get_district_comparison_data())
    except SupabaseError:
        return jsonify([])


@app.route("/api/flat_type_breakdown")
@api_login_required
def api_flat_type_breakdown():
    """Return flat type breakdown for a town."""
    town = request.args.get("town", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")
    return jsonify(_get_flat_type_breakdown_data(town, street_name, block))


@app.route("/api/monthly_volume")
@api_login_required
def api_monthly_volume():
    """Return monthly transaction volume."""
    town = request.args.get("town", "")

    try:
        return jsonify(_supabase_rpc("rpc_api_monthly_volume", {"p_town": town or None}) or [])
    except SupabaseError:
        return jsonify([])


# ---------------------------------------------------------------------------
# API endpoints: Public (no auth required)
# ---------------------------------------------------------------------------

def _build_landing_stats():
    """Stable stats payload used by landing template and public API."""
    try:
        total_txns = _supabase_count("transactions")
    except Exception:
        total_txns = None

    try:
        town_count = len(_get_district_summary_data())
    except Exception:
        town_count = len(TOWNS)

    performance = ARTEFACTS.get("performance", {})
    mape = performance.get("test_mape_display")
    run_dir = ARTEFACTS.get("run_dir")
    last_updated = os.path.basename(run_dir) if run_dir else None

    return {
        "total_txns": total_txns,
        "mape": mape,
        "town_count": town_count,
        "model_label": ARTEFACTS.get("model_label", "Model"),
        "last_updated": last_updated,
        "data_sources": "HDB resale + Supabase",
    }


@app.route("/api/public/landing-stats")
def api_public_landing_stats():
    """Public KPI payload for landing hero counters."""
    return jsonify(_build_landing_stats())


@app.route("/api/public/location_summary")
def api_public_location_summary():
    """Per-town centroids with blurred price bucket (1-5) for guest teaser map."""
    town_list = [dict(r) for r in _get_district_summary_data()]
    town_list = [t for t in town_list if t.get("lat") and t.get("lng")]
    town_list.sort(key=lambda x: x["avg_price"] or 0)
    n = len(town_list)
    for i, t in enumerate(town_list):
        t["price_bucket"] = min(5, int(i / max(n, 1) * 5) + 1)
        t["total_txns"] = t.get("total_txns", 0)
        del t["avg_price"]
        t.pop("recent_avg", None)
        t.pop("recent_txns", None)
    town_list.sort(key=lambda x: x["town"])
    return jsonify(town_list)


@app.route("/api/public/recent_ticker")
def api_public_recent_ticker():
    """20 most recent transactions for homepage ticker. No auth required."""
    try:
        rows = _supabase_rpc("rpc_api_transactions", {"p_limit": 20}) or []
        return jsonify([
            {
                "town": row.get("town"),
                "flat_type": row.get("flat_type"),
                "resale_price": row.get("resale_price"),
                "year": row.get("year"),
            }
            for row in rows
        ])
    except SupabaseError:
        return jsonify([])


# ---------------------------------------------------------------------------
# API endpoints: Authenticated helpers
# ---------------------------------------------------------------------------

@app.route("/api/available_models")
@api_login_required
def api_available_models():
    """Returns flat models available for a given town and flat_type."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")
    return jsonify({"models": _get_available_models_data(town, flat_type, street_name, block)})


@app.route("/api/available_storey_ranges")
@api_login_required
def api_available_storey_ranges():
    """Returns storey ranges available for a given town and flat_type."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")
    return jsonify({"storey_ranges": _get_available_storey_ranges_data(town, flat_type, street_name, block)})


@app.route("/api/floor_area_stats")
@api_login_required
def api_floor_area_stats():
    """Min, max, avg floor area for a town + flat_type combination."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")
    return jsonify(_get_floor_area_stats_data(town, flat_type, street_name, block))


@app.route("/api/lease_year_range")
@api_login_required
def api_lease_year_range():
    """Min and max lease_commence_date for a town."""
    town = request.args.get("town", "")
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")
    return jsonify(_get_lease_year_range_data(town, street_name, block))


@app.route("/api/available_streets")
@api_login_required
def api_available_streets():
    """Returns street names for a given town."""
    town = request.args.get("town", "")
    if not town:
        return jsonify({"streets": []})

    try:
        rows = _supabase_rpc("rpc_available_streets", {"p_town": town}) or []
        return jsonify({"streets": [r["street_name"] for r in rows]})
    except SupabaseError:
        return jsonify({"streets": []})


@app.route("/api/available_blocks")
@api_login_required
def api_available_blocks():
    """Returns blocks for a given town + street."""
    town = request.args.get("town", "")
    street = request.args.get("street_name", "")
    if not town or not street:
        return jsonify({"blocks": []})

    try:
        rows = _supabase_rpc("rpc_available_blocks", {"p_town": town, "p_street": street}) or []
        return jsonify({"blocks": [r["block"] for r in rows]})
    except SupabaseError:
        return jsonify({"blocks": []})


@app.route("/api/prediction_context")
@api_login_required
def api_prediction_context():
    """Returns lease decay + recent transactions for prediction analytics."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    predicted_price = request.args.get("predicted_price", type=float, default=0)
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")

    try:
        lease_decay = _supabase_rpc("rpc_lease_decay", {
            "p_town": town,
            "p_flat_type": flat_type or None,
            "p_street_name": street_name or None,
            "p_block": block or None,
        }) or []
    except SupabaseError:
        lease_decay = []

    # Recent transactions
    recent = _get_recent_similar_transactions(
        town,
        flat_type,
        limit=20,
        street_name=street_name,
        block=block,
    )

    return jsonify({
        "lease_decay": lease_decay,
        "recent_transactions": recent,
        "predicted_price": predicted_price,
    })


# ---------------------------------------------------------------------------
# API: Future Prediction
# ---------------------------------------------------------------------------

@app.route("/api/future_prediction")
@api_login_required
def api_future_prediction():
    """Return a 5-year price forecast as JSON."""
    form_data = {
        "town": request.args.get("town", "").strip(),
        "flat_type": request.args.get("flat_type", "").strip(),
        "flat_model": request.args.get("flat_model", "").strip(),
        "floor_area": request.args.get("floor_area", "").strip(),
        "storey_range": request.args.get("storey_range", "").strip(),
        "lease_commence": request.args.get("lease_commence", "").strip(),
        "street_name": request.args.get("street_name", "").strip(),
        "block": request.args.get("block", "").strip(),
    }
    town = form_data["town"]

    if not town:
        return jsonify({"error": "town is required"}), 400

    resolved_form, assumptions = _complete_prediction_form_data(
        form_data,
        infer_flat_type=True,
    )
    flat_type = resolved_form["flat_type"]
    flat_model = resolved_form["flat_model"]
    floor_area = resolved_form["floor_area"]
    storey_range = resolved_form["storey_range"]
    lease_commence = resolved_form["lease_commence"]
    street_name = resolved_form["street_name"]
    block = resolved_form["block"]

    # Resolve block distances if street_name and block provided
    block_distances = None
    if street_name and block:
        block_distances = _get_block_distances(town, street_name, block)

    current_year = datetime.now().year
    try:
        result = predict_price(
            town, flat_type, flat_model, floor_area, storey_range,
            lease_commence, override_distances=block_distances,
        )
    except Exception:
        return jsonify({"error": "Prediction failed"}), 500

    timeline = [{
        **result,
        "year": current_year,
        "remaining_lease": max(0, 99 - (current_year - lease_commence)),
    }]
    for y_offset in range(1, 6):
        future_year = current_year + y_offset
        try:
            fp = predict_price(
                town, flat_type, flat_model, floor_area, storey_range,
                lease_commence, override_year=future_year,
                override_distances=block_distances,
            )
        except Exception:
            fp = _enrich_prediction_result(0, prediction_year=future_year)
        fp["year"] = future_year
        fp["remaining_lease"] = max(0, 99 - (future_year - lease_commence))
        timeline.append(fp)

    return jsonify({
        "timeline": timeline,
        "resolved_inputs": {
            "flat_type": flat_type,
            "flat_model": flat_model,
            "floor_area": floor_area,
            "storey_range": storey_range,
            "lease_commence": lease_commence,
            "assumptions": assumptions,
        },
    })


# ---------------------------------------------------------------------------
# AI Insights (Gemini-powered FAQ)
# ---------------------------------------------------------------------------

_AI_QUESTIONS_PROMPT = """You are a Singapore HDB (public housing) market analyst.
The user is viewing analytics for: {filter_desc}

Chart data summary:
- Price Trend: {trend_summary}
- Transaction Volume: {volume_summary}
- Flat Type Mix: {flat_type_summary}
- Benchmark Comparison: {benchmark_summary}
- Price per sqm: {psf_summary}

Generate 3-6 insightful questions a home buyer or investor would ask about this data.
Group the questions by chart topic. Each group should have 1-2 questions.
Questions must be SPECIFIC to the numbers and patterns shown — not generic.

Return ONLY valid JSON in this exact format, with no other text:
{{"groups": {{"trend": ["question1", ...], "volume": ["question1", ...], "benchmark": ["question1", ...]}}}}

Use only these group keys: trend, volume, flat_type, benchmark, psf. Omit a group if no interesting question exists for it."""

_AI_ANSWER_PROMPT = """You are a Singapore HDB (public housing) market analyst.
The user is viewing analytics for: {filter_desc}

Relevant data:
{context_data}

Answer this question concisely in 2-3 sentences, using specific numbers from the data:
{question}"""


@app.route("/api/ai_questions", methods=["POST"])
@api_login_required
def api_ai_questions():
    """Generate AI-powered contextual questions grouped by chart topic."""
    if not GEMINI_API_KEY:
        return jsonify({"groups": {}, "tier": session.get("subscription_tier", "general"), "remaining": 0})

    body = request.get_json(silent=True) or {}
    filters = body.get("filters", {})
    chart_data = body.get("chart_data", {})

    town = filters.get("town") or "All towns"
    flat_type = filters.get("flat_type") or "All flat types"
    street = filters.get("street_name", "")
    block = filters.get("block", "")
    filter_desc = town
    if street:
        filter_desc += f" > {street}"
    if block:
        filter_desc += f" > Blk {block}"
    filter_desc += f", {flat_type}"

    prompt = _AI_QUESTIONS_PROMPT.format(
        filter_desc=filter_desc,
        trend_summary=json.dumps(chart_data.get("trend", []), default=str)[:500],
        volume_summary=json.dumps(chart_data.get("volume", []), default=str)[:300],
        flat_type_summary=json.dumps(chart_data.get("flat_type", []), default=str)[:300],
        benchmark_summary=json.dumps(chart_data.get("benchmark", []), default=str)[:500],
        psf_summary=json.dumps(chart_data.get("psf", []), default=str)[:300],
    )

    text = _call_gemini(prompt)
    if not text:
        return jsonify({"groups": {}, "tier": session.get("subscription_tier", "general"), "remaining": 0})

    #parse JSON: strip markdown fences if Gemini wraps the output
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        groups = json.loads(cleaned).get("groups", {})
    except (json.JSONDecodeError, AttributeError):
        groups = {}

    tier = session.get("subscription_tier", "general")
    if tier == "premium":
        remaining = -1  # unlimited
    else:
        used = _get_daily_ai_answer_count(_session_user_id())
        remaining = max(0, GENERAL_DAILY_AI_ANSWER_LIMIT - used)

    return jsonify({"groups": groups, "tier": tier, "remaining": remaining})


@app.route("/api/ai_answer", methods=["POST"])
@api_login_required
def api_ai_answer():
    """Generate an AI answer for a specific question. Enforces daily limit for general users."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "AI not configured"}), 503

    allowed, used, limit = _check_ai_answer_limit()
    if not allowed:
        return jsonify({"error": "limit_reached", "used": used, "limit": limit}), 429

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    context = body.get("context", {})
    if not question:
        return jsonify({"error": "No question provided"}), 400

    filters = context.get("filters", {})
    town = filters.get("town") or "All towns"
    flat_type = filters.get("flat_type") or "All flat types"
    street = filters.get("street_name", "")
    block = filters.get("block", "")
    filter_desc = town
    if street:
        filter_desc += f" > {street}"
    if block:
        filter_desc += f" > Blk {block}"
    filter_desc += f", {flat_type}"

    context_data = json.dumps(context.get("chart_data", {}), default=str)[:1500]

    prompt = _AI_ANSWER_PROMPT.format(
        filter_desc=filter_desc,
        context_data=context_data,
        question=question,
    )

    text = _call_gemini(prompt, max_tokens=512)
    if not text:
        return jsonify({"error": "AI temporarily unavailable"}), 503

    #log usage for general users
    tier = session.get("subscription_tier", "general")
    if tier != "premium":
        _log_feature_view(_session_user_id(), "ai_answer")
        remaining = max(0, GENERAL_DAILY_AI_ANSWER_LIMIT - used - 1)
    else:
        remaining = -1

    return jsonify({"answer": text.strip(), "remaining": remaining})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5001")),
    )
