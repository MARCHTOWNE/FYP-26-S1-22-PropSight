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
from functools import wraps

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


DB_PATH = _first_existing_path([
    os.environ.get("DB_PATH", ""),
    os.path.join(PROJECT_DIR, "hdb_resale.db"),
    os.path.join(PROJECT_DIR, "Database ", "hdb_resale.db"),
])

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
    """)
    conn.commit()
    conn.close()


_init_user_db()


def _get_towns():
    conn = _get_db()
    rows = conn.execute(
        "SELECT DISTINCT town FROM resale_prices ORDER BY town"
    ).fetchall()
    conn.close()
    return [r["town"] for r in rows]


def _get_flat_models():
    conn = _get_db()
    rows = conn.execute(
        "SELECT DISTINCT flat_model FROM resale_prices ORDER BY flat_model"
    ).fetchall()
    conn.close()
    return [r["flat_model"] for r in rows]


def _get_town_avg_distances():
    """Pre-compute average distances per town for prediction fallback."""
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


@app.before_request
def load_user():
    g.user = None
    if "user_id" in session:
        conn = _get_user_db()
        g.user = conn.execute(
            "SELECT * FROM users WHERE id = ?", (session["user_id"],)
        ).fetchone()
        conn.close()


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

def predict_price(town, flat_type, flat_model, floor_area, storey_range,
                  lease_commence):
    """
    Run the full feature engineering + prediction pipeline for a single property.
    Returns dict with predicted_price, price_low, price_high.
    """
    model = ARTEFACTS["model"]
    scaler = ARTEFACTS["scaler"]
    encoders = ARTEFACTS["encoders"]
    price_index = ARTEFACTS["price_index"]

    now = datetime.now()
    year = now.year
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

    # Distances (use town averages)
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


def _resolve_prediction_inputs(town, flat_type, floor_area_raw, lease_commence_raw):
    """
    Resolve optional prediction inputs.
    If floor_area or lease_commence is missing, infer from historical averages.
    """
    assumptions = []
    conn = _get_db()

    floor_area = None
    if floor_area_raw:
        floor_area = float(floor_area_raw)
    else:
        row = conn.execute(
            """
            SELECT ROUND(AVG(floor_area_sqm), 1) AS v
            FROM resale_prices
            WHERE town = ? AND flat_type = ?
            """,
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
        assumptions.append(f"Used inferred floor area: {floor_area} sqm")

    lease_commence = None
    if lease_commence_raw:
        lease_commence = int(lease_commence_raw)
    else:
        row = conn.execute(
            """
            SELECT ROUND(AVG(lease_commence_date), 0) AS v
            FROM resale_prices
            WHERE town = ? AND flat_type = ?
            """,
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
        assumptions.append(f"Used inferred lease start year: {lease_commence}")

    conn.close()
    return floor_area, lease_commence, assumptions


# ---------------------------------------------------------------------------
# Routes: Auth
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        conn = _get_user_db()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password)),
            )
            conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = _get_user_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]

        if new_password != confirm_password:
            flash("New passwords do not match.", "danger")
            return render_template("forgot_password.html")

        if len(new_password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("forgot_password.html")

        conn = _get_user_db()
        user = conn.execute(
            "SELECT id FROM users WHERE username = ? AND email = ?",
            (username, email),
        ).fetchone()

        if not user:
            conn.close()
            flash("No account found for that username and email.", "danger")
            return render_template("forgot_password.html")

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(new_password), user["id"]),
        )
        conn.commit()
        conn.close()

        flash("Password updated. Please log in with your new password.", "success")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Routes: Pages
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "home.html",
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        storey_ranges=STOREY_RANGES,
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    form_data = {}

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
        }

        result = predict_price(
            form_data["town"],
            form_data["flat_type"],
            form_data["flat_model"],
            form_data["floor_area"],
            form_data["storey_range"],
            form_data["lease_commence"],
        )
        result["assumptions"] = assumptions

        # Get trend data for context
        conn = _get_db()
        trend = conn.execute("""
            SELECT year, ROUND(AVG(resale_price)) as avg_price,
                   COUNT(*) as txn_count
            FROM resale_prices
            WHERE town = ? AND flat_type = ?
            GROUP BY year ORDER BY year
        """, (form_data["town"], form_data["flat_type"])).fetchall()
        conn.close()

        result["trend"] = [dict(r) for r in trend]

        # Momentum indicator
        if len(result["trend"]) >= 2:
            recent = result["trend"][-1]["avg_price"]
            prev = result["trend"][-2]["avg_price"]
            pct_change = (recent - prev) / prev * 100
            if pct_change > 3:
                result["momentum"] = "Rising"
                result["momentum_class"] = "success"
            elif pct_change < -3:
                result["momentum"] = "Declining"
                result["momentum_class"] = "danger"
            else:
                result["momentum"] = "Stable"
                result["momentum_class"] = "warning"
            result["momentum_pct"] = round(pct_change, 1)

        # Nearby benchmarks
        conn = _get_db()
        benchmarks = conn.execute("""
            SELECT flat_type, ROUND(AVG(resale_price)) as avg_price,
                   COUNT(*) as txn_count,
                   ROUND(AVG(floor_area_sqm)) as avg_area
            FROM resale_prices
            WHERE town = ? AND year >= 2023
            GROUP BY flat_type ORDER BY flat_type
        """, (form_data["town"],)).fetchall()
        conn.close()
        result["benchmarks"] = [dict(r) for r in benchmarks]

    return render_template(
        "predict.html",
        result=result,
        form_data=form_data,
        towns=TOWNS,
        flat_types=list(FLAT_TYPE_ORDINAL.keys()),
        flat_models=FLAT_MODELS,
        storey_ranges=STOREY_RANGES,
    )


@app.route("/save_prediction", methods=["POST"])
@login_required
def save_prediction():
    conn = _get_user_db()
    conn.execute(
        """INSERT INTO saved_predictions
           (user_id, town, flat_type, flat_model, floor_area,
            storey_range, lease_commence, predicted_price, price_low, price_high)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session["user_id"],
            request.form["town"],
            request.form["flat_type"],
            request.form["flat_model"],
            float(request.form["floor_area"]),
            request.form["storey_range"],
            int(request.form["lease_commence"]),
            float(request.form["predicted_price"]),
            float(request.form["price_low"]),
            float(request.form["price_high"]),
        ),
    )
    conn.commit()
    conn.close()
    flash("Prediction saved!", "success")
    return redirect(url_for("my_predictions"))


@app.route("/my_predictions")
@login_required
def my_predictions():
    conn = _get_user_db()
    preds = conn.execute(
        "SELECT * FROM saved_predictions WHERE user_id = ? ORDER BY created_at DESC",
        (session["user_id"],),
    ).fetchall()
    conn.close()
    return render_template("my_predictions.html", predictions=preds)


@app.route("/delete_prediction/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    conn = _get_user_db()
    conn.execute(
        "DELETE FROM saved_predictions WHERE id = ? AND user_id = ?",
        (pred_id, session["user_id"]),
    )
    conn.commit()
    conn.close()
    flash("Prediction deleted.", "info")
    return redirect(url_for("my_predictions"))


@app.route("/map")
@login_required
def map_view():
    return render_template("map.html", towns=TOWNS)


@app.route("/analytics")
@login_required
def analytics():
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
    return jsonify([dict(r) for r in rows])


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
    return jsonify([dict(r) for r in rows])


@app.route("/api/monthly_volume")
@api_login_required
def api_monthly_volume():
    """Return monthly transaction volume."""
    town = request.args.get("town", "")
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
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
