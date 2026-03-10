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
import sqlite3
from datetime import datetime
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
from werkzeug.security import check_password_hash, generate_password_hash

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hdb-resale-dev-key-change-in-prod")

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


def _first_existing_sqlite_path(paths, required_table):
    """Pick the first existing SQLite DB that contains the required table."""
    fallback = None
    for p in paths:
        if not p:
            continue
        if fallback is None:
            fallback = p
        if not os.path.exists(p) or os.path.isdir(p):
            continue
        try:
            conn = sqlite3.connect(p)
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                (required_table,),
            ).fetchone()
            conn.close()
            if row:
                return p
        except sqlite3.Error:
            continue
    return fallback or paths[0]


DB_PATH = _first_existing_sqlite_path([
    os.environ.get("DB_PATH", ""),
    os.path.join(PROJECT_DIR, "hdb_resale.db"),
    os.path.join(PROJECT_DIR, "Database", "hdb_resale.db"),
], required_table="resale_prices")

USER_DB_PATH = _first_existing_path([
    os.environ.get("USER_DB_PATH", ""),
    os.path.join(PROJECT_DIR, "users.db"),
    os.path.join(BASE_DIR, "users.db"),
])

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

STOREY_RANGES = [
    "01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12",
    "13 TO 15", "16 TO 18", "19 TO 21", "22 TO 24",
    "25 TO 27", "28 TO 30", "31 TO 33", "34 TO 36",
    "37 TO 39", "40 TO 42", "43 TO 45", "46 TO 48", "49 TO 51",
]


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

    with open(os.path.join(run_dir, "xgboost_model.pkl"), "rb") as f:
        artefacts["model"] = pickle.load(f)

    with open(os.path.join(run_dir, "scaler.pkl"), "rb") as f:
        artefacts["scaler"] = pickle.load(f)

    with open(os.path.join(run_dir, "target_encoders.pkl"), "rb") as f:
        artefacts["encoders"] = pickle.load(f)

    with open(os.path.join(run_dir, "price_index.pkl"), "rb") as f:
        artefacts["price_index"] = pickle.load(f)

    with open(os.path.join(run_dir, "metrics.json")) as f:
        artefacts["metrics"] = json.load(f)

    return artefacts


ARTEFACTS = _load_artefacts()


# ---------------------------------------------------------------------------
# DB lookups (cached)
# ---------------------------------------------------------------------------

def _get_db():
    """Get a connection to the main HDB database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_user_db():
    """Get a connection to the user database."""
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _has_local_resale_db():
    return os.path.exists(DB_PATH) and not os.path.isdir(DB_PATH)


def _ensure_sqlite_prediction_indexes():
    """Create the composite indexes used by prediction page lookups."""
    if not os.path.exists(DB_PATH):
        return

    try:
        conn = _get_db()
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_resale_town_flat_type
                ON resale_prices(town, flat_type);
            CREATE INDEX IF NOT EXISTS idx_resale_town_flat_type_model
                ON resale_prices(town, flat_type, flat_model);
            CREATE INDEX IF NOT EXISTS idx_resale_town_flat_type_storey
                ON resale_prices(town, flat_type, storey_range);
            CREATE INDEX IF NOT EXISTS idx_resale_town_lease
                ON resale_prices(town, lease_commence_date);
            CREATE INDEX IF NOT EXISTS idx_resale_town_year_flat_type
                ON resale_prices(town, year, flat_type);
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error:
        # Keep startup resilient if the local DB is unavailable.
        pass


def _init_user_db():
    """Create user tables if they don't exist."""
    conn = _get_user_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS saved_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            town TEXT NOT NULL,
            flat_type TEXT NOT NULL,
            flat_model TEXT NOT NULL,
            floor_area REAL NOT NULL,
            storey_range TEXT NOT NULL,
            lease_commence INTEGER NOT NULL,
            predicted_price REAL NOT NULL,
            price_low REAL NOT NULL,
            price_high REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS pending_registrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            code TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feature_view_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            feature TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Migrate existing DBs: add subscription_tier column if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN subscription_tier TEXT NOT NULL DEFAULT 'general' CHECK(subscription_tier IN ('general', 'premium'))")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()


# Always init local DB — needed for pending_registrations even when Supabase is the main store
_init_user_db()
_ensure_sqlite_prediction_indexes()


class SupabaseError(RuntimeError):
    """Raised when the Supabase REST API returns an error."""


# Circuit breaker: after first timeout, skip Supabase for rest of process
_supabase_circuit_open = False


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


def _supabase_rpc(function_name, params=None):
    """Call a Supabase PostgreSQL RPC function."""
    global _supabase_circuit_open
    if not SUPABASE_ENABLED or _supabase_circuit_open:
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
        _supabase_circuit_open = True
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


# SQLite-only user helpers (used when SUPABASE_ENABLED is False)

def _sqlite_get_user_by_id(user_id):
    conn = _get_user_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def _sqlite_get_user_by_email(email):
    conn = _get_user_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return user


def _sqlite_create_user(username, email, password_hash):
    conn = _get_user_db()
    conn.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email, password_hash),
    )
    conn.commit()
    conn.close()


def _save_prediction_record(user_id, prediction):
    if SUPABASE_ENABLED:
        rows = _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            method="POST",
            payload={"user_id": user_id, **prediction},
            prefer="return=representation",
        )
        return rows[0] if rows else None

    conn = _get_user_db()
    conn.execute(
        """INSERT INTO saved_predictions
           (user_id, town, flat_type, flat_model, floor_area,
            storey_range, lease_commence, predicted_price, price_low, price_high)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            prediction["town"],
            prediction["flat_type"],
            prediction["flat_model"],
            prediction["floor_area"],
            prediction["storey_range"],
            prediction["lease_commence"],
            prediction["predicted_price"],
            prediction["price_low"],
            prediction["price_high"],
        ),
    )
    conn.commit()
    conn.close()
    return None


def _get_saved_predictions(user_id):
    if SUPABASE_ENABLED:
        return _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            filters={"user_id": f"eq.{user_id}", "order": "created_at.desc"},
        ) or []

    conn = _get_user_db()
    preds = conn.execute(
        "SELECT * FROM saved_predictions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return preds


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


def _format_comparison_delta(value, suffix="", currency=False):
    if value is None:
        return "N/A"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"

    sign = "+" if numeric > 0 else "-" if numeric < 0 else ""
    magnitude = abs(numeric)

    if currency:
        return f"{sign}${magnitude:,.0f}"

    if magnitude.is_integer():
        formatted = f"{int(magnitude):,}"
    else:
        formatted = f"{magnitude:,.1f}".rstrip("0").rstrip(".")
    return f"{sign}{formatted}{suffix}"


def _build_prediction_comparison(left, right):
    if not left or not right:
        return None

    predicted_delta = float(right["predicted_price"]) - float(left["predicted_price"])
    area_delta = float(right["floor_area"]) - float(left["floor_area"])
    lease_delta = int(right["lease_commence"]) - int(left["lease_commence"])
    overlap_low = max(float(left["price_low"]), float(right["price_low"]))
    overlap_high = min(float(left["price_high"]), float(right["price_high"]))
    overlap_exists = overlap_low <= overlap_high

    return {
        "predicted_delta_label": _format_comparison_delta(predicted_delta, currency=True),
        "area_delta_label": _format_comparison_delta(area_delta, suffix=" sqm"),
        "lease_delta_label": _format_comparison_delta(lease_delta, suffix=" yrs"),
        "range_overlap_exists": overlap_exists,
        "range_overlap_label": (
            f"${overlap_low:,.0f} - ${overlap_high:,.0f}"
            if overlap_exists
            else "No overlap"
        ),
    }


def _get_comparison_saved_prediction_ids():
    raw_ids = session.get("comparison_saved_prediction_ids", [])
    ids = []
    for value in raw_ids[:2]:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _set_comparison_saved_prediction_ids(prediction_ids):
    session["comparison_saved_prediction_ids"] = [int(pid) for pid in prediction_ids[:2]]


def _push_comparison_saved_prediction_id(prediction_id):
    updated = [pid for pid in _get_comparison_saved_prediction_ids() if pid != prediction_id]
    updated.append(int(prediction_id))
    updated = updated[-2:]
    _set_comparison_saved_prediction_ids(updated)
    return updated


def _default_prediction_form_data():
    return {
        "town": "",
        "flat_type": next(iter(FLAT_TYPE_ORDINAL.keys()), ""),
        "flat_model": FLAT_MODELS[0] if FLAT_MODELS else "",
        "floor_area": 90,
        "storey_range": STOREY_RANGES[0] if STOREY_RANGES else "",
        "lease_commence": 1990,
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

    return resolved_form, result, {**resolved_form, **result}


def _delete_saved_prediction(pred_id, user_id):
    if SUPABASE_ENABLED:
        _supabase_request(
            SUPABASE_PREDICTIONS_TABLE,
            method="DELETE",
            filters={"id": f"eq.{int(pred_id)}", "user_id": f"eq.{user_id}"},
        )
        return

    conn = _get_user_db()
    conn.execute(
        "DELETE FROM saved_predictions WHERE id = ? AND user_id = ?",
        (pred_id, user_id),
    )
    conn.commit()
    conn.close()


def _get_towns():
    if _has_local_resale_db():
        try:
            conn = _get_db()
            rows = conn.execute(
                "SELECT DISTINCT town FROM resale_prices ORDER BY town"
            ).fetchall()
            conn.close()
            return [r["town"] for r in rows]
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_get_towns") or []
            return [r["town"] for r in rows]
        except SupabaseError:
            pass

    return []


def _get_flat_models():
    if _has_local_resale_db():
        try:
            conn = _get_db()
            rows = conn.execute(
                "SELECT DISTINCT flat_model FROM resale_prices ORDER BY flat_model"
            ).fetchall()
            conn.close()
            return [r["flat_model"] for r in rows]
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_get_flat_models") or []
            return [r["flat_model"] for r in rows]
        except SupabaseError:
            pass

    return []


def _get_town_avg_distances():
    """Pre-compute average distances per town for prediction fallback."""
    if _has_local_resale_db():
        try:
            conn = _get_db()
            rows = conn.execute("""
                SELECT town,
                       AVG(dist_mrt) as avg_dist_mrt,
                       AVG(dist_cbd) as avg_dist_cbd,
                       AVG(dist_primary_school) as avg_dist_school,
                       AVG(dist_major_mall) as avg_dist_mall,
                       AVG(latitude) as avg_lat,
                       AVG(longitude) as avg_lng
                FROM resale_prices
                WHERE dist_mrt IS NOT NULL
                GROUP BY town
            """).fetchall()
            conn.close()
            return {r["town"]: dict(r) for r in rows}
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_get_town_avg_distances") or []
            return {
                r["town"]: {
                    "avg_dist_mrt":    r["avg_dist_mrt"],
                    "avg_dist_cbd":    r["avg_dist_cbd"],
                    "avg_dist_school": r["avg_dist_school"],
                    "avg_dist_mall":   r["avg_dist_mall"],
                    "avg_lat":         r["avg_lat"],
                    "avg_lng":         r["avg_lng"],
                }
                for r in rows
            }
        except SupabaseError:
            pass

    return {}


@lru_cache(maxsize=1)
def _get_district_summary_data():
    if _has_local_resale_db():
        try:
            conn = _get_db()
            rows = conn.execute("""
                SELECT town,
                       ROUND(AVG(resale_price)) as avg_price,
                       ROUND(AVG(CASE WHEN year >= 2023 THEN resale_price END)) as recent_avg,
                       COUNT(*) as total_txns,
                       SUM(CASE WHEN year >= 2023 THEN 1 ELSE 0 END) as recent_txns,
                       AVG(latitude) as lat,
                       AVG(longitude) as lng
                FROM resale_prices
                WHERE latitude IS NOT NULL
                GROUP BY town
                ORDER BY town
            """).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            return _supabase_rpc("rpc_api_district_summary") or []
        except SupabaseError:
            pass

    return []


@lru_cache(maxsize=64)
def _get_flat_type_breakdown_data(town):
    town = town or ""

    if _has_local_resale_db():
        try:
            conn = _get_db()
            query = """
                SELECT flat_type,
                       ROUND(AVG(resale_price)) as avg_price,
                       COUNT(*) as txn_count,
                       ROUND(AVG(floor_area_sqm)) as avg_area
                FROM resale_prices
                WHERE year >= 2023
            """
            params = []
            if town:
                query += " AND town = ?"
                params.append(town)
            query += " GROUP BY flat_type ORDER BY flat_type"
            rows = conn.execute(query, params).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            return _supabase_rpc(
                "rpc_api_flat_type_breakdown", {"p_town": town or None}
            ) or []
        except SupabaseError:
            pass

    return []


@lru_cache(maxsize=128)
def _get_available_models_data(town, flat_type):
    town = town or ""
    flat_type = flat_type or ""

    if _has_local_resale_db():
        try:
            conn = _get_db()
            rows = conn.execute(
                "SELECT DISTINCT flat_model FROM resale_prices WHERE town = ? AND flat_type = ? ORDER BY flat_model",
                (town, flat_type),
            ).fetchall()
            conn.close()
            return [r["flat_model"] for r in rows]
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_api_available_models", {
                "p_town": town, "p_flat_type": flat_type,
            }) or []
            return [r["flat_model"] for r in rows]
        except SupabaseError:
            pass

    return []


@lru_cache(maxsize=128)
def _get_available_storey_ranges_data(town, flat_type):
    town = town or ""
    flat_type = flat_type or ""

    if not _has_local_resale_db():
        return []

    conn = _get_db()
    query = "SELECT DISTINCT storey_range FROM resale_prices WHERE 1=1"
    params = []
    if town:
        query += " AND town = ?"
        params.append(town)
    if flat_type:
        query += " AND flat_type = ?"
        params.append(flat_type)
    query += " ORDER BY storey_range"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [r["storey_range"] for r in rows]


@lru_cache(maxsize=128)
def _get_floor_area_stats_data(town, flat_type):
    town = town or ""
    flat_type = flat_type or ""

    if _has_local_resale_db():
        try:
            conn = _get_db()
            query = """
                SELECT ROUND(MIN(floor_area_sqm)) as min_area,
                       ROUND(MAX(floor_area_sqm)) as max_area,
                       ROUND(AVG(floor_area_sqm)) as avg_area
                FROM resale_prices
                WHERE 1=1
            """
            params = []
            if town:
                query += " AND town = ?"
                params.append(town)
            if flat_type:
                query += " AND flat_type = ?"
                params.append(flat_type)
            row = conn.execute(query, params).fetchone()
            conn.close()

            if row and row["min_area"]:
                return {
                    "min_area": int(row["min_area"]),
                    "max_area": int(row["max_area"]),
                    "avg_area": int(row["avg_area"]),
                }
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_api_floor_area_stats", {
                "p_town": town or None, "p_flat_type": flat_type or None,
            }) or []
            if rows and isinstance(rows, list):
                return rows[0]
            if rows and isinstance(rows, dict):
                return rows
        except SupabaseError:
            pass

    return {"min_area": 30, "max_area": 300, "avg_area": 90}


@lru_cache(maxsize=64)
def _get_lease_year_range_data(town):
    town = town or ""

    if _has_local_resale_db():
        try:
            conn = _get_db()
            query = """
                SELECT MIN(lease_commence_date) as min_year,
                       MAX(lease_commence_date) as max_year,
                       ROUND(AVG(lease_commence_date)) as avg_year
                FROM resale_prices
            """
            params = []
            if town:
                query += " WHERE town = ?"
                params.append(town)
            row = conn.execute(query, params).fetchone()
            conn.close()

            if row and row["min_year"]:
                return {
                    "min_year": int(row["min_year"]),
                    "max_year": int(row["max_year"]),
                    "avg_year": int(row["avg_year"]),
                }
        except sqlite3.Error:
            pass

    if SUPABASE_ENABLED:
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

    return {"min_year": 1960, "max_year": 2024, "avg_year": 1990}


TOWNS = _get_towns()
FLAT_MODELS = _get_flat_models()
TOWN_DISTANCES = _get_town_avg_distances()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this feature.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


def premium_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
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
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        if session.get("subscription_tier", "general") != "premium":
            return jsonify({"error": "Premium subscription required"}), 403
        return f(*args, **kwargs)
    return decorated


# Weekly view limits for general users per feature
GENERAL_WEEKLY_VIEW_LIMITS = {"map": 3, "analytics": 3, "comparison": 3}


def _get_weekly_view_count(user_id, feature):
    """Count views of a feature by user in the current week."""
    conn = _get_user_db()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM feature_view_log WHERE user_id = ? AND feature = ? AND created_at >= datetime('now', '-7 days')",
        (user_id, feature),
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def _log_feature_view(user_id, feature):
    """Record a feature view."""
    conn = _get_user_db()
    conn.execute("INSERT INTO feature_view_log (user_id, feature) VALUES (?, ?)", (user_id, feature))
    conn.commit()
    conn.close()


def _check_feature_limit(feature):
    """Check if general user has exceeded weekly view limit for a feature.
    Returns (allowed, views_used, views_limit)."""
    tier = session.get("subscription_tier", "general")
    if tier == "premium":
        return True, 0, 0
    limit = GENERAL_WEEKLY_VIEW_LIMITS.get(feature, 3)
    count = _get_weekly_view_count(session["user_id"], feature)
    return count < limit, count, limit


@app.before_request
def load_user():
    g.user = None
    if "user_id" in session:
        if SUPABASE_ENABLED:
            # Reconstruct from session — no extra DB round-trip needed
            g.user = {
                "id": session["user_id"],
                "username": session.get("username", ""),
                "email": session.get("email", ""),
                "subscription_tier": session.get("subscription_tier", "general"),
            }
        else:
            g.user = _sqlite_get_user_by_id(session["user_id"])


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
    price_index = ARTEFACTS["price_index"]

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
    parts = storey_range.split(" TO ")
    storey_mid = (int(parts[0]) + int(parts[1])) / 2

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
    pred_log = model.predict(df[FEATURE_COLS])[0]

    # Get latest price index
    quarter_key = year * 10 + ((month_num - 1) // 3 + 1)
    pi = price_index.get(quarter_key)
    if pi is None:
        # Fallback to most recent available
        pi = price_index.iloc[-1]

    predicted_price = float(np.expm1(pred_log) * pi)

    # Confidence range based on model MAPE (~6.5%)
    mape = ARTEFACTS["metrics"]["winner"]["test_mape"] / 100
    price_low = predicted_price * (1 - mape)
    price_high = predicted_price * (1 + mape)

    return {
        "predicted_price": round(predicted_price),
        "price_low": round(price_low),
        "price_high": round(price_high),
        "mape": round(mape * 100, 1),
    }


def _get_recent_similar_transactions(town, flat_type, limit=5):
    """Return recent transactions for same town + flat_type."""
    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_recent_similar_transactions", {
                "p_town": town, "p_flat_type": flat_type, "p_limit": limit,
            }) or []
            return rows
        except SupabaseError:
            pass

    if not _has_local_resale_db():
        return []
    try:
        conn = _get_db()
        rows = conn.execute(
            """SELECT block, street_name, storey_range, floor_area_sqm,
                      resale_price, month
               FROM resale_prices
               WHERE town = ? AND flat_type = ?
               ORDER BY year DESC, month DESC
               LIMIT ?""",
            (town, flat_type, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def _get_block_distances(town, street_name, block):
    """Look up actual distances for a specific block."""
    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_block_distances", {
                "p_town": town, "p_street": street_name, "p_block": block,
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
            pass

    if _has_local_resale_db():
        try:
            conn = _get_db()
            row = conn.execute(
                """SELECT AVG(dist_mrt) as dist_mrt, AVG(dist_cbd) as dist_cbd,
                          AVG(dist_primary_school) as dist_school,
                          AVG(dist_major_mall) as dist_mall
                   FROM resale_prices
                   WHERE town = ? AND street_name = ? AND block = ?
                     AND dist_mrt IS NOT NULL""",
                (town, street_name, block),
            ).fetchone()
            conn.close()
            if row and row["dist_mrt"] is not None:
                return dict(row)
        except sqlite3.Error:
            pass
    return None


def _resolve_prediction_inputs(town, flat_type, floor_area_raw, lease_commence_raw):
    """
    Resolve optional prediction inputs.
    If floor_area or lease_commence is missing, infer from historical averages.
    """
    assumptions = []

    floor_area = None
    if floor_area_raw:
        floor_area = float(floor_area_raw)
    else:
        resolved = False
        if SUPABASE_ENABLED:
            try:
                v = _supabase_rpc("rpc_resolve_floor_area", {"p_town": town, "p_flat_type": flat_type})
                floor_area = float(v) if v else 90.0
                resolved = True
            except SupabaseError:
                pass
        if not resolved:
            conn = _get_db()
            row = conn.execute(
                "SELECT ROUND(AVG(floor_area_sqm), 1) AS v FROM resale_prices WHERE town = ? AND flat_type = ?",
                (town, flat_type),
            ).fetchone()
            if row and row["v"]:
                floor_area = float(row["v"])
            else:
                row = conn.execute(
                    "SELECT ROUND(AVG(floor_area_sqm), 1) AS v FROM resale_prices WHERE town = ?",
                    (town,),
                ).fetchone()
                floor_area = float(row["v"]) if row and row["v"] else 90.0
            conn.close()
        assumptions.append(f"Used inferred floor area: {floor_area} sqm")

    lease_commence = None
    if lease_commence_raw:
        lease_commence = int(lease_commence_raw)
    else:
        resolved = False
        if SUPABASE_ENABLED:
            try:
                v = _supabase_rpc("rpc_resolve_lease_commence", {"p_town": town, "p_flat_type": flat_type})
                lease_commence = int(v) if v else 1990
                resolved = True
            except SupabaseError:
                pass
        if not resolved:
            conn = _get_db()
            row = conn.execute(
                "SELECT ROUND(AVG(lease_commence_date), 0) AS v FROM resale_prices WHERE town = ? AND flat_type = ?",
                (town, flat_type),
            ).fetchone()
            if row and row["v"]:
                lease_commence = int(row["v"])
            else:
                row = conn.execute(
                    "SELECT ROUND(AVG(lease_commence_date), 0) AS v FROM resale_prices WHERE town = ?",
                    (town,),
                ).fetchone()
                lease_commence = int(row["v"]) if row and row["v"] else 1990
            conn.close()
        assumptions.append(f"Used inferred lease start year: {lease_commence}")

    return floor_area, lease_commence, assumptions


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

        if SUPABASE_ENABLED:
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
                db_user = {}

            if result.get("access_token"):
                session["user_id"] = db_user.get("id")
                session["username"] = username
                session["email"] = email
                session["access_token"] = result["access_token"]
                session["subscription_tier"] = "general"
                flash("Account created! Welcome.", "success")
                return redirect(url_for("home"))
            else:
                flash("Account created! Check your email to confirm before logging in.", "success")
                return redirect(url_for("login"))
        else:
            try:
                _sqlite_create_user(username, email, generate_password_hash(password))
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username or email already exists.", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if SUPABASE_ENABLED:
            try:
                result = _supabase_auth(
                    "/token?grant_type=password",
                    payload={"email": email, "password": password},
                )
            except SupabaseError:
                flash("Invalid email or password.", "danger")
                return render_template("login.html")

            # Fetch public.users record for integer ID (used by saved_predictions FK)
            try:
                rows = _supabase_request(
                    SUPABASE_USERS_TABLE,
                    filters={"email": f"eq.{email}", "limit": "1"},
                )
                db_user = rows[0] if rows else {}
            except SupabaseError:
                db_user = {}

            auth_user = result.get("user") or {}
            session["user_id"] = db_user.get("id")
            session["username"] = db_user.get("username") or auth_user.get("user_metadata", {}).get("username", email.split("@")[0])
            session["email"] = email
            session["access_token"] = result.get("access_token", "")
            session["subscription_tier"] = db_user.get("subscription_tier", "general")
            flash(f"Welcome back, {session['username']}!", "success")
            return redirect(url_for("home"))
        else:
            user = _sqlite_get_user_by_email(email)
            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["email"] = user["email"]
                session["subscription_tier"] = user["subscription_tier"] if "subscription_tier" in user.keys() else "general"
                flash(f"Welcome back, {user['username']}!", "success")
                return redirect(url_for("home"))
            flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()

        if SUPABASE_ENABLED:
            try:
                _supabase_auth("/recover", payload={"email": email})
            except SupabaseError:
                pass  # Don't reveal whether the email exists
            flash("If that email is registered, you'll receive a password reset link.", "info")
            return redirect(url_for("login"))
        else:
            flash("Password reset is not available in offline mode.", "warning")

    return render_template("forgot_password.html")


@app.route("/logout")
def logout():
    if SUPABASE_ENABLED:
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
    user_id = session["user_id"]
    if SUPABASE_ENABLED:
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
    else:
        conn = _get_user_db()
        conn.execute("UPDATE users SET subscription_tier = 'premium' WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
    session["subscription_tier"] = "premium"
    flash("You've been upgraded to Premium! Enjoy unlimited access.", "success")
    return redirect(url_for("pricing"))


# ---------------------------------------------------------------------------
# Routes: Pages
# ---------------------------------------------------------------------------

def _get_popular_predictions(limit=3):
    """Return the most common town+flat_type prediction combos across all users.
    Falls back to top towns by recent transaction volume from resale data."""
    # Try saved predictions first
    try:
        conn = _get_user_db()
        rows = conn.execute(
            """SELECT town, flat_type,
                      ROUND(AVG(predicted_price)) as avg_price,
                      COUNT(*) as count
               FROM saved_predictions
               GROUP BY town, flat_type
               ORDER BY count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        results = [dict(r) for r in rows]
        if results:
            return results
    except Exception:
        pass

    # Fallback: top towns by recent transactions from resale data
    try:
        conn = _get_db()
        rows = conn.execute(
            """SELECT town, flat_type,
                      ROUND(AVG(resale_price)) as avg_price,
                      COUNT(*) as count
               FROM resale_prices
               WHERE year >= 2023
               GROUP BY town, flat_type
               ORDER BY count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


@app.route("/")
def home():
    # Total transaction count
    try:
        if SUPABASE_ENABLED:
            try:
                result = _supabase_rpc("rpc_count_transactions")
                total_txns = int(result) if result else 0
            except SupabaseError:
                conn = _get_db()
                total_txns = conn.execute("SELECT COUNT(*) as c FROM resale_prices").fetchone()["c"]
                conn.close()
        else:
            conn = _get_db()
            total_txns = conn.execute("SELECT COUNT(*) as c FROM resale_prices").fetchone()["c"]
            conn.close()
    except Exception:
        total_txns = 970000

    # Model MAPE
    try:
        artefact_mape = round(ARTEFACTS["metrics"]["winner"]["test_mape"], 1)
    except Exception:
        artefact_mape = 6.5

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
        artefact_mape=artefact_mape,
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

    selected_saved_ids = _get_comparison_saved_prediction_ids() if g.user else []
    left_id = request.values.get("left_id", "").strip() or (
        str(selected_saved_ids[0]) if len(selected_saved_ids) >= 1 else ""
    )
    right_id = request.values.get("right_id", "").strip() or (
        str(selected_saved_ids[1]) if len(selected_saved_ids) >= 2 else ""
    )

    left_saved = _get_saved_prediction_by_id(saved_predictions, left_id)
    right_saved = _get_saved_prediction_by_id(saved_predictions, right_id)

    left_form_data = _prediction_form_from_saved(left_saved)
    right_form_data = _prediction_form_from_saved(right_saved)
    left_result = None
    right_result = None
    comparison_summary = None

    if request.method == "POST":
        left_form_data = _extract_prediction_form_data(request.form, "left", seed=left_form_data)
        right_form_data = _extract_prediction_form_data(request.form, "right", seed=right_form_data)
    should_compare = request.method == "POST" or bool(left_saved and right_saved)

    if should_compare and left_form_data.get("town") and right_form_data.get("town"):
        left_form_data, left_result, left_payload = _run_prediction_form(left_form_data)
        right_form_data, right_result, right_payload = _run_prediction_form(right_form_data)
        comparison_summary = _build_prediction_comparison(left_payload, right_payload)

    return render_template(
        "comparison.html",
        saved_predictions=saved_predictions,
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        storey_ranges=STOREY_RANGES,
        left_prefilled=bool(left_saved),
        right_prefilled=bool(right_saved),
        left_form_data=left_form_data,
        right_form_data=right_form_data,
        left_result=left_result,
        right_result=right_result,
        comparison_summary=comparison_summary,
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
    if len(updated_ids) == 1:
        flash("Saved prediction added to comparison. Choose one more to auto-fill the second side.", "info")
    else:
        flash("Saved prediction added. The latest two saved selections are now auto-filled in comparison.", "success")

    return redirect(url_for("comparison"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    form_data = {}
    timeline = None
    flat_age = None
    remaining_lease = None
    town_avg_price = None
    recent_transactions = None

    if request.method == "POST":
        floor_area_raw = request.form.get("floor_area", "").strip()
        lease_commence_raw = request.form.get("lease_commence", "").strip()
        floor_area, lease_commence, assumptions = _resolve_prediction_inputs(
            request.form["town"],
            request.form["flat_type"],
            floor_area_raw,
            lease_commence_raw,
        )

        form_data = {
            "town": request.form["town"],
            "flat_type": request.form["flat_type"],
            "flat_model": request.form["flat_model"],
            "floor_area": floor_area,
            "storey_range": request.form["storey_range"],
            "lease_commence": lease_commence,
            "street_name": request.form.get("street_name", "").strip(),
            "block": request.form.get("block", "").strip(),
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
            form_data["town"], form_data["flat_type"]
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

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_api_transactions", {
                "p_town": town or None, "p_limit": limit
            }) or []
            return jsonify(rows)
        except SupabaseError:
            pass

    conn = _get_db()
    query = """
        SELECT town, flat_type, block, street_name, storey_range,
               floor_area_sqm, resale_price, month, year,
               latitude, longitude
        FROM resale_prices
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """
    params = []
    if town:
        query += " AND town = ?"
        params.append(town)
    query += " ORDER BY year DESC, month DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/district_summary")
@api_login_required
def api_district_summary():
    """Return per-town summary stats for district heatmap."""
    return jsonify(_get_district_summary_data())


@app.route("/api/predicted_heatmap")
@api_login_required
def api_predicted_heatmap():
    """Run the prediction model for a representative flat in each town.

    Query params (all optional — defaults to a typical 4 Room flat):
      flat_type       e.g. "4 Room"
      flat_model      e.g. "Model A"
      floor_area      e.g. 93
      storey_range    e.g. "07 TO 09"
      lease_commence  e.g. 1995
    """
    flat_type = request.args.get("flat_type", "4 Room")
    flat_model = request.args.get("flat_model", "Model A")
    floor_area_raw = request.args.get("floor_area", "")
    storey_range = request.args.get("storey_range", "07 TO 09")
    lease_commence_raw = request.args.get("lease_commence", "")

    district_data = _get_district_summary_data()
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
            floor_area = 90
            lease_commence = 1990

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

    conn = _get_db()
    query = """
        SELECT year, ROUND(AVG(resale_price)) as avg_price,
               ROUND(PERCENTILE(resale_price, 25)) as q1,
               ROUND(PERCENTILE(resale_price, 75)) as q3,
               COUNT(*) as txn_count
        FROM resale_prices WHERE 1=1
    """
    params = []
    if town:
        query += " AND town = ?"
        params.append(town)
    if flat_type:
        query += " AND flat_type = ?"
        params.append(flat_type)

    query += " GROUP BY year ORDER BY year"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/price_trend_simple")
@api_login_required
def api_price_trend_simple():
    """Yearly avg price trend (SQLite-compatible, no PERCENTILE)."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")

    if SUPABASE_ENABLED:
        try:
            return jsonify(_supabase_rpc("rpc_api_price_trend_simple", {
                "p_town": town or None, "p_flat_type": flat_type or None
            }) or [])
        except SupabaseError:
            pass  # fall through to SQLite

    conn = _get_db()
    query = """
        SELECT year,
               ROUND(AVG(resale_price)) as avg_price,
               ROUND(MIN(resale_price)) as min_price,
               ROUND(MAX(resale_price)) as max_price,
               COUNT(*) as txn_count
        FROM resale_prices WHERE 1=1
    """
    params = []
    if town:
        query += " AND town = ?"
        params.append(town)
    if flat_type:
        query += " AND flat_type = ?"
        params.append(flat_type)
    query += " GROUP BY year ORDER BY year"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/district_comparison")
@api_login_required
def api_district_comparison():
    """Return per-town avg prices for the most recent year."""
    if SUPABASE_ENABLED:
        try:
            return jsonify(_supabase_rpc("rpc_api_district_comparison") or [])
        except SupabaseError:
            pass

    conn = _get_db()
    rows = conn.execute("""
        SELECT town,
               ROUND(AVG(resale_price)) as avg_price,
               COUNT(*) as txn_count,
               ROUND(AVG(floor_area_sqm)) as avg_area,
               ROUND(AVG(resale_price / floor_area_sqm)) as psf
        FROM resale_prices
        WHERE year = (SELECT MAX(year) FROM resale_prices)
        GROUP BY town
        ORDER BY avg_price DESC
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/flat_type_breakdown")
@api_login_required
def api_flat_type_breakdown():
    """Return flat type breakdown for a town."""
    town = request.args.get("town", "")
    return jsonify(_get_flat_type_breakdown_data(town))


@app.route("/api/monthly_volume")
@api_login_required
def api_monthly_volume():
    """Return monthly transaction volume."""
    town = request.args.get("town", "")

    if SUPABASE_ENABLED:
        try:
            return jsonify(_supabase_rpc("rpc_api_monthly_volume", {"p_town": town or None}) or [])
        except SupabaseError:
            pass

    conn = _get_db()
    query = """
        SELECT month, COUNT(*) as txn_count,
               ROUND(AVG(resale_price)) as avg_price
        FROM resale_prices WHERE 1=1
    """
    params = []
    if town:
        query += " AND town = ?"
        params.append(town)
    query += " GROUP BY month ORDER BY month"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# API endpoints: Public (no auth required)
# ---------------------------------------------------------------------------

@app.route("/api/public/location_summary")
def api_public_location_summary():
    """Per-town centroids with blurred price bucket (1-5) for guest teaser map."""
    if SUPABASE_ENABLED:
        try:
            return jsonify(_supabase_rpc("rpc_api_public_location_summary") or [])
        except SupabaseError:
            pass  # fall through to SQLite

    conn = _get_db()
    rows = conn.execute("""
        SELECT town,
               AVG(latitude) as lat,
               AVG(longitude) as lng,
               COUNT(*) as total_txns,
               AVG(resale_price) as avg_price
        FROM resale_prices
        WHERE latitude IS NOT NULL
        GROUP BY town
        ORDER BY town
    """).fetchall()
    conn.close()

    town_list = [dict(r) for r in rows]
    # Compute quintile buckets manually
    town_list.sort(key=lambda x: x["avg_price"] or 0)
    n = len(town_list)
    for i, t in enumerate(town_list):
        t["price_bucket"] = min(5, int(i / max(n, 1) * 5) + 1)
        del t["avg_price"]  # Don't expose actual prices to guests
    town_list.sort(key=lambda x: x["town"])
    return jsonify(town_list)


@app.route("/api/public/recent_ticker")
def api_public_recent_ticker():
    """20 most recent transactions for homepage ticker. No auth required."""
    if SUPABASE_ENABLED:
        try:
            return jsonify(_supabase_rpc("rpc_api_public_recent_ticker") or [])
        except SupabaseError:
            pass

    conn = _get_db()
    rows = conn.execute("""
        SELECT town, flat_type, resale_price, year
        FROM resale_prices
        ORDER BY year DESC, month DESC
        LIMIT 20
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


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

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_available_streets", {"p_town": town}) or []
            return jsonify({"streets": [r["street_name"] for r in rows]})
        except SupabaseError:
            pass

    if _has_local_resale_db():
        conn = _get_db()
        rows = conn.execute(
            "SELECT DISTINCT street_name FROM resale_prices WHERE town = ? ORDER BY street_name",
            (town,),
        ).fetchall()
        conn.close()
        return jsonify({"streets": [r["street_name"] for r in rows]})

    return jsonify({"streets": []})


@app.route("/api/available_blocks")
@api_login_required
def api_available_blocks():
    """Returns blocks for a given town + street."""
    town = request.args.get("town", "")
    street = request.args.get("street_name", "")
    if not town or not street:
        return jsonify({"blocks": []})

    if SUPABASE_ENABLED:
        try:
            rows = _supabase_rpc("rpc_available_blocks", {"p_town": town, "p_street": street}) or []
            return jsonify({"blocks": [r["block"] for r in rows]})
        except SupabaseError:
            pass

    if _has_local_resale_db():
        conn = _get_db()
        rows = conn.execute(
            "SELECT DISTINCT block FROM resale_prices WHERE town = ? AND street_name = ? ORDER BY block",
            (town, street),
        ).fetchall()
        conn.close()
        return jsonify({"blocks": [r["block"] for r in rows]})

    return jsonify({"blocks": []})


@app.route("/api/prediction_context")
@api_login_required
def api_prediction_context():
    """Returns lease decay + recent transactions for prediction analytics."""
    town = request.args.get("town", "")
    flat_type = request.args.get("flat_type", "")
    predicted_price = request.args.get("predicted_price", type=float, default=0)

    # Lease decay
    lease_decay = []
    if SUPABASE_ENABLED:
        try:
            lease_decay = _supabase_rpc("rpc_lease_decay", {
                "p_town": town, "p_flat_type": flat_type or None,
            }) or []
        except SupabaseError:
            pass

    if not lease_decay and _has_local_resale_db() and town:
        try:
            conn = _get_db()
            query = """
                SELECT CAST(remaining_lease/10 AS INT)*10 as lease_bucket,
                       ROUND(AVG(resale_price)) as avg_price,
                       COUNT(*) as txn_count
                FROM resale_prices WHERE town = ?
            """
            params = [town]
            if flat_type:
                query += " AND flat_type = ?"
                params.append(flat_type)
            query += " GROUP BY lease_bucket ORDER BY lease_bucket DESC"
            rows = conn.execute(query, params).fetchall()
            conn.close()
            lease_decay = [dict(r) for r in rows]
        except sqlite3.Error:
            pass

    # Recent transactions
    recent = _get_recent_similar_transactions(town, flat_type, limit=20)

    return jsonify({
        "lease_decay": lease_decay,
        "recent_transactions": recent,
        "predicted_price": predicted_price,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
