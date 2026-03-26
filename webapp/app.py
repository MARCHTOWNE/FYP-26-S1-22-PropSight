"""
app.py — HDB Resale Price Analytics Platform
=============================================
Flask web application serving:
  - Property valuation predictions (XGBoost model)
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
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from socket import timeout as SocketTimeout
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
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


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

ASSETS_DIR = _first_existing_path([
    os.environ.get("MODEL_ASSETS_DIR", ""),
    os.path.join(PROJECT_DIR, "model_assets"),
    os.path.join(PROJECT_DIR, "ML", "model_assets"),
])


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
# ---------------------------------------------------------------------------

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

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "ensemble": "Ensemble",
}

SCALE_COLS = [
    "floor_area_sqm", "storey_midpoint", "flat_age", "remaining_lease",
    "lease_commence_date", "month_sin", "month_cos", "year",
    "dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall",
]

FEATURE_COLS = [
    "flat_type_ordinal", "town_enc", "flat_model_enc",
    "floor_area_sqm", "storey_midpoint", "flat_age", "remaining_lease",
    "lease_commence_date", "month_sin", "month_cos", "year",
    "is_mature_estate", "dist_mrt", "dist_cbd",
    "dist_primary_school", "dist_major_mall",
]

STOREY_RANGES = [str(i) for i in range(1, 52)]

HDB_FIRST_YEAR = 1960
HDB_DATASET_START_YEAR = 1990
DEFAULT_FLOOR_AREA = 90


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
        "test_mape_display": round(test_mape, 1) if test_mape is not None else None,
        "test_rmse": test_rmse,
        "test_rmse_display": f"{round(test_rmse):,}" if test_rmse is not None else None,
        "test_r2": test_r2,
        "test_r2_display": f"{test_r2:.3f}" if test_r2 is not None else None,
        "future_holdout_mape": future_mape,
        "future_holdout_mape_display": round(future_mape, 1) if future_mape is not None else None,
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
    }


def _resolve_serving_model_key(run_dir, metrics):
    preferred = (metrics.get("winner", {}) or {}).get("winner")
    candidates = []
    if preferred:
        candidates.append(preferred)
    # Prefer the declared winner, then fall back to ensemble or base models.
    candidates.extend(["ensemble", "xgboost", "lgbm", "rf"])

    seen = set()
    for model_key in candidates:
        if model_key in seen:
            continue
        seen.add(model_key)
        model_path = os.path.join(run_dir, f"{model_key}_model.pkl")
        if os.path.exists(model_path):
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
            if not os.path.exists(run_dir):
                run_dir = None

    if run_dir is None:
        # Fallback: pick the newest artefact directory under ASSETS_DIR.
        dirs = []
        if os.path.isdir(ASSETS_DIR):
            for name in os.listdir(ASSETS_DIR):
                p = os.path.join(ASSETS_DIR, name)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "xgboost_model.pkl")):
                    dirs.append(p)
        if not dirs:
            raise FileNotFoundError(
                f"No model artefact run directory found under {ASSETS_DIR}"
            )
        run_dir = sorted(dirs)[-1]

    return run_dir


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

    if serving_model_key == "ensemble":
        # Load ensemble: meta-learner + all base models
        with open(serving_model_path, "rb") as f:
            ensemble_data = pickle.load(f)
        artefacts["meta_model"] = ensemble_data["meta_model"]
        base_models = {}
        for base_name in artefacts["meta_model"].base_learner_order_:
            base_path = os.path.join(run_dir, f"{base_name}_model.pkl")
            with open(base_path, "rb") as f:
                base_models[base_name] = pickle.load(f)
        artefacts["base_models"] = base_models
        artefacts["model"] = None  # ensemble uses base_models + meta_model
    else:
        with open(serving_model_path, "rb") as f:
            artefacts["model"] = pickle.load(f)
        artefacts["base_models"] = None
        artefacts["meta_model"] = None

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

    return artefacts


ARTEFACTS = _load_artefacts()


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
    cutoff = (datetime.utcnow() - timedelta(days=7)).replace(microsecond=0).isoformat() + "Z"
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
        ("dist_mall", "Nearest Mall", "dist", "min"),
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

    # Mall distance insight
    best_l, best_v, worst_l, worst_v = _best_worst("dist_mall", "min")
    if best_l:
        insights.append(
            f"Prediction {best_l} is closest to a major mall ({_format_distance(best_v)}), "
            f"while {worst_l} is farthest ({_format_distance(worst_v)})."
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


def _run_prediction_form(form_data):
    floor_area_raw = str(form_data.get("floor_area", "")).strip()
    lease_commence_raw = str(form_data.get("lease_commence", "")).strip()
    floor_area, lease_commence, assumptions = _resolve_prediction_inputs(
        form_data["town"],
        form_data["flat_type"],
        floor_area_raw,
        lease_commence_raw,
        form_data.get("street_name", ""),
        form_data.get("block", ""),
    )

    resolved_form = dict(form_data)
    resolved_form["floor_area"] = floor_area
    resolved_form["lease_commence"] = lease_commence

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
        "dist_mall": dists.get("dist_mall"),
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
                "avg_lat": r["avg_lat"],
                "avg_lng": r["avg_lng"],
            }
            for r in rows
        }
    except SupabaseError:
        return {}


@lru_cache(maxsize=1)
def _get_district_summary_data():
    try:
        return _supabase_rpc("rpc_api_district_summary") or []
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


@lru_cache(maxsize=256)
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


@lru_cache(maxsize=128)
def _get_available_models_data(town, flat_type):
    town = town or ""
    flat_type = flat_type or ""

    try:
        rows = _supabase_rpc("rpc_api_available_models", {
            "p_town": town,
            "p_flat_type": flat_type,
        }) or []
        return [r["flat_model"] for r in rows]
    except SupabaseError:
        return []


@lru_cache(maxsize=128)
def _get_available_storey_ranges_data(town, flat_type):
    """Returns individual floor numbers derived from DB storey ranges."""
    town = town or ""
    flat_type = flat_type or ""

    try:
        rows = _supabase_rpc("rpc_api_available_storey_ranges", {
            "p_town": town or None,
            "p_flat_type": flat_type or None,
        }) or []
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


@lru_cache(maxsize=128)
def _get_floor_area_stats_data(town, flat_type):
    town = town or ""
    flat_type = flat_type or ""

    try:
        rows = _supabase_rpc("rpc_api_floor_area_stats", {
            "p_town": town or None,
            "p_flat_type": flat_type or None,
        }) or []
        if rows and isinstance(rows, list):
            return rows[0]
        if rows and isinstance(rows, dict):
            return rows
    except SupabaseError:
        pass

    return {"min_area": 30, "max_area": 300, "avg_area": DEFAULT_FLOOR_AREA}


@lru_cache(maxsize=64)
def _get_lease_year_range_data(town):
    town = town or ""

    try:
        rows = _supabase_rpc(
            "rpc_api_lease_year_range", {"p_town": town or None}
        ) or []
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

def predict_price(town, flat_type, flat_model, floor_area, storey_range,
                  lease_commence, override_year=None, override_distances=None):
    """
    Run the full feature engineering + prediction pipeline for a single property.
    Returns dict with predicted_price, price_low, price_high.
    """
    model = ARTEFACTS["model"]
    scaler = ARTEFACTS["scaler"]
    encoders = ARTEFACTS["encoders"]
    price_index = ARTEFACTS.get("price_index")
    target_transform = ARTEFACTS.get("target_transform", "log1p_resale_price")

    now = datetime.now()
    year = override_year if override_year is not None else now.year
    month_num = now.month

    # Derived features
    flat_age = year - lease_commence
    remaining_lease = max(0, 99 - flat_age)
    month_sin = math.sin(2 * math.pi * month_num / 12)
    month_cos = math.cos(2 * math.pi * month_num / 12)
    is_mature = 1 if town in MATURE_ESTATES else 0
    flat_type_ord = FLAT_TYPE_ORDINAL.get(flat_type, 4)

    # Target encoding
    town_enc_map = encoders["town"]["means"]
    town_enc = town_enc_map.get(town, encoders["town"]["global_mean"])

    flat_model_enc_map = encoders["flat_model"]["means"]
    flat_model_enc = flat_model_enc_map.get(
        flat_model, encoders["flat_model"]["global_mean"]
    )

    # Storey midpoint
    storey_mid = _storey_midpoint(storey_range)

    # Distances (use block-level if provided, else town averages)
    if override_distances:
        dist_mrt = override_distances.get("dist_mrt", 500)
        dist_cbd = override_distances.get("dist_cbd", 10000)
        dist_school = override_distances.get("dist_school", 500)
        dist_mall = override_distances.get("dist_mall", 1000)
    else:
        dists = TOWN_DISTANCES.get(town, {})
        dist_mrt = dists.get("avg_dist_mrt", 500)
        dist_cbd = dists.get("avg_dist_cbd", 10000)
        dist_school = dists.get("avg_dist_school", 500)
        dist_mall = dists.get("avg_dist_mall", 1000)

    # Build feature row (pre-scaling)
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
    }

    df = pd.DataFrame([raw])

    # Scale
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

    # Predict (log1p scale)
    if ARTEFACTS["model_key"] == "ensemble":
        # Ensemble: stack base learner predictions, pass through meta-learner
        base_models = ARTEFACTS["base_models"]
        meta_model = ARTEFACTS["meta_model"]
        meta_features = np.column_stack([
            base_models[name].predict(df[FEATURE_COLS])
            for name in meta_model.base_learner_order_
        ])
        pred_log = meta_model.predict(meta_features)[0]
    else:
        pred_log = model.predict(df[FEATURE_COLS])[0]

    if target_transform == "rpi_adjusted_log_price" and price_index is not None:
        quarter_key = year * 10 + ((month_num - 1) // 3 + 1)
        pi = price_index.get(quarter_key)
        if pi is None:
            # Fallback to chronologically latest quarter
            pi = price_index.loc[price_index.index.max()]
        predicted_price = float(np.expm1(pred_log) * pi)
    else:
        predicted_price = float(np.expm1(pred_log))

    # Confidence range based on the serving model's recorded test MAPE.
    performance = ARTEFACTS.get("performance", {})
    mape_pct = performance.get("test_mape")
    if mape_pct is None:
        winner = ARTEFACTS.get("metrics", {}).get("winner", {})
        mape_pct = _safe_metric(winner.get("test_mape")) or 10.0
    mape = mape_pct / 100
    price_low = predicted_price * (1 - mape)
    price_high = predicted_price * (1 + mape)

    return {
        "predicted_price": round(predicted_price),
        "price_low": round(price_low),
        "price_high": round(price_high),
        "mape": round(mape * 100, 1),
        "model_label": performance.get("label", ARTEFACTS.get("model_label", "Model")),
    }


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
        rows = _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            filters={"select": "town,flat_type,predicted_price"},
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
    _log_feature_view(session["user_id"], "comparison")
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

    if request.method == "POST":
        town = request.form.get("town", "").strip()
        flat_type = request.form.get("flat_type", "").strip()
        flat_model = request.form.get("flat_model", "").strip()
        storey_range = request.form.get("storey_range", "").strip()
        floor_area_raw = request.form.get("floor_area", "").strip()
        lease_commence_raw = request.form.get("lease_commence", "").strip()
        street_name = request.form.get("street_name", "").strip()
        block = request.form.get("block", "").strip()

        # Prevent crashes when the frontend disables options (disabled controls are not submitted).
        if not town or not flat_type or not flat_model or not storey_range:
            flash("Cannot get estimate. Please select a valid flat type/model.", "warning")
            form_data = {
                "town": town,
                "flat_type": flat_type,
                "flat_model": flat_model,
                "floor_area": floor_area_raw,
                "storey_range": storey_range,
                "lease_commence": lease_commence_raw,
                "street_name": street_name,
                "block": block,
            }
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
            )

        if flat_type not in FLAT_TYPE_ORDINAL:
            flash("Cannot get estimate for this flat type.", "warning")
            form_data = {
                "town": town,
                "flat_type": flat_type,
                "flat_model": flat_model,
                "floor_area": floor_area_raw,
                "storey_range": storey_range,
                "lease_commence": lease_commence_raw,
                "street_name": street_name,
                "block": block,
            }
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
            )

        floor_area, lease_commence, assumptions = _resolve_prediction_inputs(
            town,
            flat_type,
            floor_area_raw,
            lease_commence_raw,
            street_name,
            block,
        )

        form_data = {
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "floor_area": floor_area,
            "storey_range": storey_range,
            "lease_commence": lease_commence,
            "street_name": street_name,
            "block": block,
        }

        # Look up block-level distances if block is specified
        block_distances = None
        if form_data["block"] and form_data["street_name"]:
            block_distances = _get_block_distances(
                form_data["town"], form_data["street_name"], form_data["block"]
            )

        result = predict_price(
            form_data["town"],
            form_data["flat_type"],
            form_data["flat_model"],
            form_data["floor_area"],
            form_data["storey_range"],
            form_data["lease_commence"],
            override_distances=block_distances,
        )
        result["assumptions"] = assumptions

        # Timeline: predict for 1-5 years ahead
        current_year = datetime.now().year
        timeline = [{"year": current_year, "predicted_price": result["predicted_price"],
                      "price_low": result["price_low"], "price_high": result["price_high"],
                      "remaining_lease": max(0, 99 - (current_year - form_data["lease_commence"]))}]
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
        "price_low": float(request.form["price_low"]),
        "price_high": float(request.form["price_high"]),
        "street_name": request.form.get("street_name", "").strip(),
        "block": request.form.get("block", "").strip(),
    }
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
    _log_feature_view(session["user_id"], "map")
    return render_template("map.html", towns=TOWNS)


@app.route("/analytics")
@login_required
def analytics():
    allowed, _, limit = _check_feature_limit("analytics")
    if not allowed:
        flash(f"You've used all {limit} free Analytics views this week. Upgrade to Premium for unlimited access.", "warning")
        return redirect(url_for("pricing"))
    _log_feature_view(session["user_id"], "analytics")
    return render_template("analytics.html", towns=TOWNS)


# ---------------------------------------------------------------------------
# API endpoints (JSON) for AJAX calls from frontend
# ---------------------------------------------------------------------------

@app.route("/api/transactions")
@api_login_required
def api_transactions():
    """Return recent transactions with lat/lng for map pins."""
    town = request.args.get("town", "")
    limit = min(int(request.args.get("limit", 500)), 2000)

    try:
        rows = _supabase_rpc("rpc_api_transactions", {
            "p_town": town or None,
            "p_limit": limit,
        }) or []
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
    """Run the prediction model for a representative flat in each town.

    Query params (all optional — defaults to a typical 4 Room flat):
      flat_type       e.g. "4 Room"
      flat_model      e.g. "Model A"
      floor_area      e.g. 93
      storey_range    e.g. "8"
      lease_commence  e.g. 1995
    """
    flat_type = request.args.get("flat_type", "4 Room")
    flat_model = request.args.get("flat_model", "Model A")
    floor_area_raw = request.args.get("floor_area", "")
    storey_range = request.args.get("storey_range", "8")
    lease_commence_raw = request.args.get("lease_commence", "")

    district_data = _get_prediction_map_seed_data()
    if not district_data:
        return jsonify({"error": "Prediction map data is currently unavailable."}), 503
    results = []

    for d in district_data:
        town = d["town"]
        if not d.get("lat") or not d.get("lng"):
            continue

        try:
            floor_area, lease_commence, _ = _resolve_prediction_inputs(
                town, flat_type,
                floor_area_raw or "",
                lease_commence_raw or "",
            )
        except Exception:
            floor_area = DEFAULT_FLOOR_AREA
            lease_commence = _default_lease_year_range()["avg_year"]

        available_models = _get_available_models_data(town, flat_type)
        model_to_use = flat_model if flat_model in available_models else (
            available_models[0] if available_models else flat_model
        )

        try:
            pred = predict_price(
                town, flat_type, model_to_use,
                floor_area, storey_range, lease_commence,
            )
        except Exception:
            pred = {"predicted_price": 0, "price_low": 0, "price_high": 0}

        results.append({
            "town": town,
            "lat": d["lat"],
            "lng": d["lng"],
            "predicted_price": pred["predicted_price"],
            "price_low": pred["price_low"],
            "price_high": pred["price_high"],
            "avg_price": d.get("avg_price", 0),
            "recent_avg": d.get("recent_avg", 0),
            "total_txns": d.get("total_txns", 0),
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
        return jsonify(_supabase_rpc("rpc_api_district_comparison") or [])
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
    return jsonify({"models": _get_available_models_data(town, flat_type)})


@app.route("/api/available_storey_ranges")
@api_login_required
def api_available_storey_ranges():
    """Returns storey ranges available for a given town and flat_type."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    return jsonify({"storey_ranges": _get_available_storey_ranges_data(town, flat_type)})


@app.route("/api/floor_area_stats")
@api_login_required
def api_floor_area_stats():
    """Min, max, avg floor area for a town + flat_type combination."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    return jsonify(_get_floor_area_stats_data(town, flat_type))


@app.route("/api/lease_year_range")
@api_login_required
def api_lease_year_range():
    """Min and max lease_commence_date for a town."""
    town = request.args.get("town", "")
    return jsonify(_get_lease_year_range_data(town))


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
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    flat_model = request.args.get("flat_model", "")
    floor_area_raw = request.args.get("floor_area", "").strip()
    storey_range = request.args.get("storey_range", "")
    lease_commence_raw = request.args.get("lease_commence", "").strip()
    street_name = request.args.get("street_name", "")
    block = request.args.get("block", "")

    if not town:
        return jsonify({"error": "town is required"}), 400

    flat_type, forecast_assumptions = _resolve_forecast_flat_type(
        town,
        flat_type,
        street_name=street_name,
        block=block,
    )

    floor_area, lease_commence, assumptions = _resolve_prediction_inputs(
        town,
        flat_type,
        floor_area_raw,
        lease_commence_raw,
        street_name=street_name,
        block=block,
    )

    available_models = _get_available_models_data(town, flat_type)
    if flat_model and available_models and flat_model not in available_models:
        flat_model = ""
    if not flat_model:
        flat_model = available_models[0] if available_models else "Model A"

    available_storey_ranges = _get_available_storey_ranges_data(town, flat_type)
    if storey_range and available_storey_ranges and storey_range not in available_storey_ranges:
        storey_range = ""
    if not storey_range:
        if available_storey_ranges:
            storey_range = available_storey_ranges[len(available_storey_ranges) // 2]
        else:
            storey_range = "8"

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

    timeline = [{"year": current_year, "predicted_price": result["predicted_price"],
                 "price_low": result["price_low"], "price_high": result["price_high"],
                 "remaining_lease": max(0, 99 - (current_year - lease_commence))}]
    for y_offset in range(1, 6):
        future_year = current_year + y_offset
        try:
            fp = predict_price(
                town, flat_type, flat_model, floor_area, storey_range,
                lease_commence, override_year=future_year,
                override_distances=block_distances,
            )
        except Exception:
            fp = {"predicted_price": 0, "price_low": 0, "price_high": 0}
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
            "assumptions": forecast_assumptions + assumptions,
        },
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5001")),
    )
