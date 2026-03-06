"""
proximity_features.py
=====================
Single responsibility: compute dist_mrt, dist_cbd, dist_school, dist_mall
for every geocoded row in resale_prices and write them back to hdb_resale.db.

Design decisions:
  - Pure-Python haversine formula; no external geospatial libraries required.
  - Reference data embedded as module-level constants for portability.
  - Columns already exist in the schema from data_pipeline.py — no ALTER TABLE.
  - Rows with null coordinates are skipped; their distance columns remain NULL.
  - Uses a temporary table + UPDATE FROM pattern for efficient bulk writes.

Execution order context:
  Step 4 of 6. Reads/writes: hdb_resale.db (resale_prices).
  Previous step: geocoding.py. Next step: feature_engineering.py.

Run:
    python proximity_features.py
"""

import math
import sqlite3
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH         = "hdb_resale.db"
EARTH_RADIUS_KM = 6371.0                    # mean Earth radius, km
RAFFLES_PLACE   = (1.2832, 103.8517)        # CBD anchor point (lat, lng)

# ---------------------------------------------------------------------------
# Reference data
# Representative sample — replace with full dataset before production use.
# ---------------------------------------------------------------------------

# MRT / LRT stations (Singapore, 2024) — name, lat, lng
MRT_STATIONS: list[dict[str, Any]] = [
    {"name": "Jurong East",     "lat": 1.3329, "lng": 103.7424},
    {"name": "Bukit Batok",     "lat": 1.3490, "lng": 103.7497},
    {"name": "Bukit Gombak",    "lat": 1.3588, "lng": 103.7517},
    {"name": "Choa Chu Kang",   "lat": 1.3853, "lng": 103.7444},
    {"name": "Yew Tee",         "lat": 1.3972, "lng": 103.7470},
    {"name": "Kranji",          "lat": 1.4252, "lng": 103.7619},
    {"name": "Marsiling",       "lat": 1.4322, "lng": 103.7746},
    {"name": "Woodlands",       "lat": 1.4370, "lng": 103.7865},
    {"name": "Admiralty",       "lat": 1.4408, "lng": 103.8005},
    {"name": "Sembawang",       "lat": 1.4489, "lng": 103.8197},
    {"name": "Yishun",          "lat": 1.4294, "lng": 103.8354},
    {"name": "Khatib",          "lat": 1.4173, "lng": 103.8329},
    {"name": "Yio Chu Kang",    "lat": 1.3817, "lng": 103.8449},
    {"name": "Ang Mo Kio",      "lat": 1.3699, "lng": 103.8497},
    {"name": "Bishan",          "lat": 1.3508, "lng": 103.8487},
    {"name": "Braddell",        "lat": 1.3407, "lng": 103.8472},
    {"name": "Toa Payoh",       "lat": 1.3327, "lng": 103.8473},
    {"name": "Novena",          "lat": 1.3204, "lng": 103.8437},
    {"name": "Newton",          "lat": 1.3127, "lng": 103.8385},
    {"name": "Orchard",         "lat": 1.3039, "lng": 103.8319},
    {"name": "Somerset",        "lat": 1.3006, "lng": 103.8389},
    {"name": "Dhoby Ghaut",     "lat": 1.2995, "lng": 103.8457},
    {"name": "City Hall",       "lat": 1.2931, "lng": 103.8520},
    {"name": "Raffles Place",   "lat": 1.2832, "lng": 103.8517},
    {"name": "Marina Bay",      "lat": 1.2762, "lng": 103.8547},
    {"name": "Outram Park",     "lat": 1.2802, "lng": 103.8394},
    {"name": "Tanjong Pagar",   "lat": 1.2762, "lng": 103.8454},
    {"name": "Redhill",         "lat": 1.2893, "lng": 103.8168},
    {"name": "Queenstown",      "lat": 1.2943, "lng": 103.8057},
    {"name": "Commonwealth",    "lat": 1.3023, "lng": 103.7981},
    {"name": "Buona Vista",     "lat": 1.3072, "lng": 103.7898},
    {"name": "Dover",           "lat": 1.3114, "lng": 103.7789},
    {"name": "Clementi",        "lat": 1.3152, "lng": 103.7652},
    {"name": "Boon Lay",        "lat": 1.3386, "lng": 103.7058},
    {"name": "Lakeside",        "lat": 1.3440, "lng": 103.7208},
    {"name": "Tampines",        "lat": 1.3538, "lng": 103.9453},
    {"name": "Pasir Ris",       "lat": 1.3721, "lng": 103.9494},
    {"name": "Bedok",           "lat": 1.3240, "lng": 103.9301},
    {"name": "Kembangan",       "lat": 1.3202, "lng": 103.9130},
    {"name": "Eunos",           "lat": 1.3196, "lng": 103.9032},
    {"name": "Paya Lebar",      "lat": 1.3175, "lng": 103.8920},
    {"name": "Aljunied",        "lat": 1.3163, "lng": 103.8831},
    {"name": "Kallang",         "lat": 1.3115, "lng": 103.8712},
    {"name": "Lavender",        "lat": 1.3074, "lng": 103.8634},
    {"name": "Bugis",           "lat": 1.3009, "lng": 103.8556},
    {"name": "Sengkang",        "lat": 1.3915, "lng": 103.8954},
    {"name": "Punggol",         "lat": 1.4053, "lng": 103.9023},
    {"name": "Hougang",         "lat": 1.3713, "lng": 103.8920},
    {"name": "Serangoon",       "lat": 1.3504, "lng": 103.8731},
    {"name": "Potong Pasir",    "lat": 1.3313, "lng": 103.8693},
]

# Primary schools (Singapore) — name, lat, lng
PRIMARY_SCHOOLS: list[dict[str, Any]] = [
    {"name": "Anglo-Chinese School (Primary)",         "lat": 1.3191, "lng": 103.8296},
    {"name": "Catholic High School (Primary)",         "lat": 1.3508, "lng": 103.8427},
    {"name": "Henry Park Primary",                     "lat": 1.3206, "lng": 103.7756},
    {"name": "Nanyang Primary",                        "lat": 1.3284, "lng": 103.8013},
    {"name": "Raffles Girls' Primary",                 "lat": 1.3707, "lng": 103.8462},
    {"name": "Singapore Chinese Girls' Primary",       "lat": 1.3213, "lng": 103.8214},
    {"name": "St. Hilda's Primary",                    "lat": 1.3526, "lng": 103.9367},
    {"name": "Tao Nan School",                         "lat": 1.3015, "lng": 103.9057},
    {"name": "Maris Stella High (Primary)",            "lat": 1.3159, "lng": 103.8566},
    {"name": "CHIJ St. Nicholas Girls' Primary",       "lat": 1.3713, "lng": 103.8430},
    {"name": "Ai Tong School",                         "lat": 1.3631, "lng": 103.8477},
    {"name": "Bukit Timah Primary",                    "lat": 1.3488, "lng": 103.7891},
    {"name": "Fairfield Methodist Primary",            "lat": 1.3150, "lng": 103.7822},
    {"name": "Greenridge Primary",                     "lat": 1.3811, "lng": 103.7499},
    {"name": "Holy Innocents' Primary",                "lat": 1.3639, "lng": 103.8944},
    {"name": "Jurong Primary",                         "lat": 1.3490, "lng": 103.7140},
    {"name": "Keming Primary",                         "lat": 1.3462, "lng": 103.7512},
    {"name": "Lakeside Primary",                       "lat": 1.3441, "lng": 103.7208},
    {"name": "North Vista Primary",                    "lat": 1.3878, "lng": 103.9019},
    {"name": "Opera Estate Primary",                   "lat": 1.3251, "lng": 103.9254},
    {"name": "Pasir Ris Primary",                      "lat": 1.3737, "lng": 103.9504},
    {"name": "Pioneer Primary",                        "lat": 1.3460, "lng": 103.6991},
    {"name": "Queenstown Primary",                     "lat": 1.2930, "lng": 103.8035},
    {"name": "Rosyth School",                          "lat": 1.3737, "lng": 103.8877},
    {"name": "Sembawang Primary",                      "lat": 1.4489, "lng": 103.8155},
    {"name": "Tampines Primary",                       "lat": 1.3541, "lng": 103.9385},
    {"name": "Teck Ghee Primary",                      "lat": 1.3763, "lng": 103.8622},
    {"name": "West Spring Primary",                    "lat": 1.3780, "lng": 103.7716},
    {"name": "Yishun Primary",                         "lat": 1.4239, "lng": 103.8351},
    {"name": "Zhangde Primary",                        "lat": 1.3327, "lng": 103.8567},
    {"name": "Punggol Primary",                        "lat": 1.4048, "lng": 103.9138},
    {"name": "Bedok Green Primary",                    "lat": 1.3393, "lng": 103.9362},
]

# Shopping malls (Singapore) — name, lat, lng
SHOPPING_MALLS: list[dict[str, Any]] = [
    {"name": "VivoCity",                    "lat": 1.2644, "lng": 103.8222},
    {"name": "ION Orchard",                 "lat": 1.3042, "lng": 103.8318},
    {"name": "Paragon",                     "lat": 1.3039, "lng": 103.8330},
    {"name": "Plaza Singapura",             "lat": 1.3003, "lng": 103.8454},
    {"name": "Bugis Junction",              "lat": 1.2993, "lng": 103.8554},
    {"name": "Marina Square",               "lat": 1.2909, "lng": 103.8571},
    {"name": "Suntec City",                 "lat": 1.2933, "lng": 103.8582},
    {"name": "Tampines Mall",               "lat": 1.3528, "lng": 103.9453},
    {"name": "Century Square",              "lat": 1.3531, "lng": 103.9445},
    {"name": "Jurong Point",                "lat": 1.3399, "lng": 103.7058},
    {"name": "IMM",                         "lat": 1.3329, "lng": 103.7424},
    {"name": "Westgate",                    "lat": 1.3333, "lng": 103.7428},
    {"name": "NEX",                         "lat": 1.3504, "lng": 103.8731},
    {"name": "Heartland Mall (Kovan)",      "lat": 1.3611, "lng": 103.8856},
    {"name": "AMK Hub",                     "lat": 1.3699, "lng": 103.8497},
    {"name": "Junction 8",                  "lat": 1.3505, "lng": 103.8487},
    {"name": "Causeway Point",              "lat": 1.4370, "lng": 103.7865},
    {"name": "Northpoint City",             "lat": 1.4294, "lng": 103.8354},
    {"name": "Sun Plaza",                   "lat": 1.4489, "lng": 103.8197},
    {"name": "Eastpoint Mall",              "lat": 1.3430, "lng": 103.9540},
    {"name": "Bedok Mall",                  "lat": 1.3240, "lng": 103.9301},
    {"name": "White Sands",                 "lat": 1.3737, "lng": 103.9494},
    {"name": "Lot One",                     "lat": 1.3853, "lng": 103.7444},
    {"name": "Bukit Panjang Plaza",         "lat": 1.3780, "lng": 103.7716},
    {"name": "Clementi Mall",               "lat": 1.3152, "lng": 103.7652},
    {"name": "The Star Vista",              "lat": 1.3072, "lng": 103.7898},
    {"name": "Tiong Bahru Plaza",           "lat": 1.2861, "lng": 103.8272},
    {"name": "Queensway Shopping Centre",   "lat": 1.2943, "lng": 103.8057},
    {"name": "Hougang Mall",                "lat": 1.3639, "lng": 103.8894},
    {"name": "Compass One",                 "lat": 1.3915, "lng": 103.8954},
    {"name": "Punggol Plaza",               "lat": 1.3985, "lng": 103.9044},
    {"name": "Waterway Point",              "lat": 1.4053, "lng": 103.9023},
]


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
        Input DataFrame with dist_mrt, dist_cbd, dist_school, dist_mall
        columns added or overwritten.
    """
    # Pre-build CBD tuple for vectorised use
    cbd_lat, cbd_lng = RAFFLES_PLACE

    geocoded_mask = df["latitude"].notna() & df["longitude"].notna()
    n_geocoded = geocoded_mask.sum()
    print(f"  Computing proximity features for {n_geocoded:,} geocoded rows ...")

    # Initialise with NaN; only geocoded rows will be filled
    df["dist_mrt"]    = float("nan")
    df["dist_cbd"]    = float("nan")
    df["dist_school"] = float("nan")
    df["dist_mall"]   = float("nan")

    if n_geocoded == 0:
        print("  WARNING: No geocoded rows found. All distance columns remain null.")
        return df

    def _compute_row(row) -> tuple[float, float, float, float]:
        lat, lon = row["latitude"], row["longitude"]
        return (
            nearest_distance(lat, lon, MRT_STATIONS),
            haversine(lat, lon, cbd_lat, cbd_lng),
            nearest_distance(lat, lon, PRIMARY_SCHOOLS),
            nearest_distance(lat, lon, SHOPPING_MALLS),
        )

    computed = df.loc[geocoded_mask].apply(_compute_row, axis=1, result_type="expand")
    computed.columns = ["dist_mrt", "dist_cbd", "dist_school", "dist_mall"]

    df.loc[geocoded_mask, ["dist_mrt", "dist_cbd", "dist_school", "dist_mall"]] = computed
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_proximity_features(db_path: str = DB_PATH) -> None:
    """
    Read geocoded rows from resale_prices, compute distances, and write
    the four distance columns back to the DB using a temporary table.

    Parameters:
        db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)

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
            rowid      INTEGER PRIMARY KEY,
            dist_mrt   REAL,
            dist_cbd   REAL,
            dist_school REAL,
            dist_mall  REAL
        )
    """)
    df[["rowid", "dist_mrt", "dist_cbd", "dist_school", "dist_mall"]].to_sql(
        "_prox_tmp", conn, if_exists="append", index=False
    )

    conn.execute("""
        UPDATE resale_prices
        SET
            dist_mrt    = t.dist_mrt,
            dist_cbd    = t.dist_cbd,
            dist_school = t.dist_school,
            dist_mall   = t.dist_mall
        FROM _prox_tmp AS t
        WHERE resale_prices.rowid = t.rowid
    """)
    conn.execute("DROP TABLE IF EXISTS _prox_tmp")
    conn.commit()

    # Summary statistics
    stats = df[["dist_mrt", "dist_cbd", "dist_school", "dist_mall"]].describe()
    print(f"\n  Proximity feature summary (km):")
    for col in ["dist_mrt", "dist_cbd", "dist_school", "dist_mall"]:
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
