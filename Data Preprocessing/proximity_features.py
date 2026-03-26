"""
proximity_features.py
=====================
Single responsibility: compute dist_mrt, dist_cbd, dist_primary_school,
and dist_major_mall for every geocoded row in resale_prices and write them
back to hdb_resale.db.

Design decisions:
  - Pure-Python haversine formula; no external geospatial libraries required.
  - Reference data is loaded from JSON files under /reference_data.
  - Columns already exist in the schema from data_pipeline.py — no ALTER TABLE.
  - Rows with null coordinates are skipped; their distance columns remain NULL.
  - Uses a temporary table + UPDATE FROM pattern for efficient bulk writes.

Execution order context:
  Step 4 of 6. Reads/writes: hdb_resale.db (resale_prices).
  Previous step: geocoding.py. Next step: feature_engineering.py.

Run:
    python proximity_features.py
"""

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).resolve().parent
DB_PATH         = os.environ.get("HDB_SQLITE_PATH", str(BASE_DIR / "hdb_resale.db"))
EARTH_RADIUS_KM = 6371.0                    # mean Earth radius, km
RAFFLES_PLACE   = (1.2832, 103.8517)        # CBD anchor point (lat, lng)
REFERENCE_DATA_DIR = Path(
    os.environ.get("HDB_REFERENCE_DATA_DIR", str(BASE_DIR / "reference_data"))
)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

MRT_STATIONS: list[dict[str, Any]] = []
PRIMARY_SCHOOLS: list[dict[str, Any]] = []
MAJOR_SHOPPING_MALLS: list[dict[str, Any]] = []


def load_reference_data() -> None:
    """Load MRT, school, and mall reference JSON files into module globals."""
    global MRT_STATIONS, PRIMARY_SCHOOLS, MAJOR_SHOPPING_MALLS

    sources = {
        "mrt_stations.json": "MRT_STATIONS",
        "primary_schools.json": "PRIMARY_SCHOOLS",
        "major_shopping_malls.json": "MAJOR_SHOPPING_MALLS",
    }
    loaded: dict[str, list[dict[str, Any]]] = {}

    for filename, label in sources.items():
        path = REFERENCE_DATA_DIR / filename
        if not path.exists():
            raise RuntimeError(
                f"Missing reference file: {path}. Run fetch_reference_data.py once "
                "before running proximity_features.py."
            )
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not records:
            raise RuntimeError(
                f"Reference file {path} is empty or malformed for {label}."
            )
        loaded[label] = records

    MRT_STATIONS = loaded["MRT_STATIONS"]
    PRIMARY_SCHOOLS = loaded["PRIMARY_SCHOOLS"]
    MAJOR_SHOPPING_MALLS = loaded["MAJOR_SHOPPING_MALLS"]


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth (km).

    Uses the Haversine formula; suitable for the short distances typical
    of Singapore urban geography (< 50 km).

    Parameters:
        lat1, lon1: Coordinates of the first point (decimal degrees).
        lat2, lon2: Coordinates of the second point (decimal degrees).

    Returns:
        Distance in kilometres.
    """
    # Convert degrees to radians
    phi1    = math.radians(lat1)
    phi2    = math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def nearest_distance(
    lat: float,
    lon: float,
    locations: list[dict[str, Any]],
) -> float:
    """
    Find the minimum great-circle distance from (lat, lon) to any point
    in a list of named locations.

    Parameters:
        lat:       Latitude of the query point (decimal degrees).
        lon:       Longitude of the query point (decimal degrees).
        locations: List of dicts, each with keys "lat" and "lng".

    Returns:
        Minimum distance in km, rounded to 4 decimal places.
    """
    min_dist = min(
        haversine(lat, lon, loc["lat"], loc["lng"])
        for loc in locations
    )
    return round(min_dist, 4)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all four distance columns for rows with non-null coordinates.

    Rows with null latitude or longitude receive NaN for all distance columns.

    Parameters:
        df: DataFrame containing at least "latitude" and "longitude" columns.

    Returns:
        Input DataFrame with dist_mrt, dist_cbd, dist_primary_school,
        dist_major_mall
        columns added or overwritten.
    """
    # Pre-build CBD tuple for vectorised use
    cbd_lat, cbd_lng = RAFFLES_PLACE

    geocoded_mask = df["latitude"].notna() & df["longitude"].notna()
    n_geocoded = geocoded_mask.sum()
    print(f"  Computing proximity features for {n_geocoded:,} geocoded rows ...")

    # Initialise with NaN; only geocoded rows will be filled
    df["dist_mrt"]            = float("nan")
    df["dist_cbd"]            = float("nan")
    df["dist_primary_school"] = float("nan")
    df["dist_major_mall"]     = float("nan")

    if n_geocoded == 0:
        print("  WARNING: No geocoded rows found. All distance columns remain null.")
        return df

    def _compute_row(row) -> tuple[float, float, float, float]:
        lat, lon = row["latitude"], row["longitude"]
        return (
            nearest_distance(lat, lon, MRT_STATIONS),
            haversine(lat, lon, cbd_lat, cbd_lng),
            nearest_distance(lat, lon, PRIMARY_SCHOOLS),
            nearest_distance(lat, lon, MAJOR_SHOPPING_MALLS),
        )

    computed = df.loc[geocoded_mask].apply(_compute_row, axis=1, result_type="expand")
    computed.columns = ["dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"]

    df.loc[
        geocoded_mask,
        ["dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"],
    ] = computed
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ensure_proximity_columns(conn: sqlite3.Connection) -> None:
    """
    Ensure proximity output columns exist in resale_prices.

    This keeps the script compatible with older DB files that still have
    dist_school/dist_mall but not the renamed columns.
    """
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(resale_prices)").fetchall()
    }
    required_columns = [
        "dist_mrt",
        "dist_cbd",
        "dist_primary_school",
        "dist_major_mall",
    ]
    missing = [column for column in required_columns if column not in existing_columns]

    for column in missing:
        conn.execute(f"ALTER TABLE resale_prices ADD COLUMN {column} REAL")
        print(f"  Added missing column: {column}")

    if missing:
        conn.commit()


def run_proximity_features(db_path: str = DB_PATH) -> None:
    """
    Read geocoded rows from resale_prices, compute distances, and write
    the four distance columns back to the DB using a temporary table.

    Parameters:
        db_path: Path to the SQLite database file.
    """
    load_reference_data()
    conn = sqlite3.connect(db_path)
    ensure_proximity_columns(conn)

    # Read only geocoded rows (avoids loading NULL lat/lng rows needlessly)
    try:
        df = pd.read_sql(
            "SELECT rowid, latitude, longitude FROM resale_prices WHERE latitude IS NOT NULL",
            conn,
        )
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            "resale_prices table not found in the database. "
            "Run data_pipeline.py and geocoding.py first."
        ) from exc
    n_read = len(df)
    print(f"\n  Geocoded rows loaded: {n_read:,}")

    if n_read == 0:
        print("  No geocoded rows. Run geocoding.py first.")
        conn.close()
        return

    df = compute_proximity_features(df)

    # Write distances back via temporary table + UPDATE FROM to avoid
    # row-by-row Python loops against SQLite
    conn.execute("DROP TABLE IF EXISTS _prox_tmp")
    conn.execute("""
        CREATE TEMPORARY TABLE _prox_tmp (
            rowid               INTEGER PRIMARY KEY,
            dist_mrt            REAL,
            dist_cbd            REAL,
            dist_primary_school REAL,
            dist_major_mall     REAL
        )
    """)
    df[["rowid", "dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"]].to_sql(
        "_prox_tmp", conn, if_exists="append", index=False
    )

    conn.execute("""
        UPDATE resale_prices
        SET
            dist_mrt            = t.dist_mrt,
            dist_cbd            = t.dist_cbd,
            dist_primary_school = t.dist_primary_school,
            dist_major_mall     = t.dist_major_mall
        FROM _prox_tmp AS t
        WHERE resale_prices.rowid = t.rowid
    """)
    conn.execute("DROP TABLE IF EXISTS _prox_tmp")
    conn.commit()

    # Summary statistics
    stats = df[["dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"]].describe()
    print(f"\n  Proximity feature summary (km):")
    for col in ["dist_mrt", "dist_cbd", "dist_primary_school", "dist_major_mall"]:
        print(
            f"    {col:<14}  "
            f"min={stats.loc['min', col]:.3f}  "
            f"mean={stats.loc['mean', col]:.3f}  "
            f"max={stats.loc['max', col]:.3f}"
        )
    print(f"\n  Rows updated: {n_read:,}")
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute and write proximity features for all geocoded rows."""
    print("=" * 60)
    print("HDB Resale — Proximity Features")
    print("=" * 60)
    run_proximity_features()
    print(f"\nDone. hdb_resale.db updated with distance columns.")


if __name__ == "__main__":
    main()
