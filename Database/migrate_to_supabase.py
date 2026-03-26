"""
migrate_to_supabase.py — Migrate hdb_resale.db → Supabase (PostgreSQL)
=======================================================================
Reads data from the local SQLite database and inserts it into the
normalized Supabase tables (run supabase_schema.sql first).

Requirements:
    pip install psycopg2-binary python-dotenv

Setup:
    Add SUPABASE_DB_URL to your .env file:
    SUPABASE_DB_URL=postgresql://postgres.[project-ref]:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres

    Find it in: Supabase Dashboard → Settings → Database → Connection string (URI)

Usage:
    cd "Database " && python migrate_to_supabase.py
"""

import os
import sqlite3

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

SQLITE_PATH = os.environ.get(
    "HDB_SQLITE_PATH",
    os.path.join(
        PROJECT_ROOT,
        "Data Preprocessing",
        "hdb_resale.db",
    ),
)
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "")
BATCH_SIZE = 5000

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


def _raise_if_missing_dimension_ids(rows, block_id_map, flat_type_id_map, flat_model_id_map):
    missing_blocks = set()
    missing_flat_types = set()
    missing_flat_models = set()

    for r in rows:
        if (r["block"], r["street_name"]) not in block_id_map:
            missing_blocks.add((r["block"], r["street_name"]))
        if r["flat_type"] not in flat_type_id_map:
            missing_flat_types.add(r["flat_type"])
        if r["flat_model"] not in flat_model_id_map:
            missing_flat_models.add(r["flat_model"])

    if missing_blocks or missing_flat_types or missing_flat_models:
        details = []
        if missing_blocks:
            sample = ", ".join(f"{block} {street}" for block, street in sorted(missing_blocks)[:3])
            details.append(f"missing block IDs (sample: {sample})")
        if missing_flat_types:
            sample = ", ".join(sorted(missing_flat_types)[:3])
            details.append(f"missing flat types ({sample})")
        if missing_flat_models:
            sample = ", ".join(sorted(missing_flat_models)[:3])
            details.append(f"missing flat models ({sample})")
        raise RuntimeError("Failed to resolve dimension IDs: " + "; ".join(details))


def migrate():
    if not SUPABASE_DB_URL:
        raise ValueError(
            "SUPABASE_DB_URL is not set.\n"
            "Add it to your .env file:\n"
            "  SUPABASE_DB_URL=postgresql://postgres.[ref]:[password]@..."
        )

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg2.connect(SUPABASE_DB_URL)
    pg_cur = pg_conn.cursor()

    print("Connected to SQLite and Supabase PostgreSQL.")
    print(f"SQLite path: {SQLITE_PATH}\n")

    # ── 1. Towns ─────────────────────────────────────────────────
    print("Step 1/5: Migrating towns...")
    rows = sqlite_conn.execute(
        "SELECT DISTINCT town FROM resale_prices ORDER BY town"
    ).fetchall()
    data = [(r["town"], r["town"] in MATURE_ESTATES) for r in rows]
    psycopg2.extras.execute_values(
        pg_cur,
        "INSERT INTO towns (name, is_mature_estate) VALUES %s ON CONFLICT (name) DO NOTHING",
        data,
    )
    pg_conn.commit()
    print(f"  Done — {len(data)} towns.\n")

    pg_cur.execute("SELECT id, name FROM towns")
    town_id_map = {name: id_ for id_, name in pg_cur.fetchall()}

    # ── 2. Flat types ─────────────────────────────────────────────
    print("Step 2/5: Migrating flat types...")
    rows = sqlite_conn.execute(
        "SELECT DISTINCT flat_type FROM resale_prices ORDER BY flat_type"
    ).fetchall()
    data = [(r["flat_type"], FLAT_TYPE_ORDINAL.get(r["flat_type"], 0)) for r in rows]
    psycopg2.extras.execute_values(
        pg_cur,
        "INSERT INTO flat_types (name, ordinal) VALUES %s ON CONFLICT (name) DO NOTHING",
        data,
    )
    pg_conn.commit()
    print(f"  Done — {len(data)} flat types.\n")

    pg_cur.execute("SELECT id, name FROM flat_types")
    flat_type_id_map = {name: id_ for id_, name in pg_cur.fetchall()}

    # ── 3. Flat models ────────────────────────────────────────────
    print("Step 3/5: Migrating flat models...")
    rows = sqlite_conn.execute(
        "SELECT DISTINCT flat_model FROM resale_prices ORDER BY flat_model"
    ).fetchall()
    data = [(r["flat_model"],) for r in rows]
    psycopg2.extras.execute_values(
        pg_cur,
        "INSERT INTO flat_models (name) VALUES %s ON CONFLICT (name) DO NOTHING",
        data,
    )
    pg_conn.commit()
    print(f"  Done — {len(data)} flat models.\n")

    pg_cur.execute("SELECT id, name FROM flat_models")
    flat_model_id_map = {name: id_ for id_, name in pg_cur.fetchall()}

    # ── 4. Blocks ─────────────────────────────────────────────────
    print("Step 4/5: Migrating blocks (unique addresses)...")
    rows = sqlite_conn.execute("""
        SELECT
            block, street_name, town, full_address,
            AVG(latitude)            AS latitude,
            AVG(longitude)           AS longitude,
            AVG(dist_mrt)            AS dist_mrt,
            AVG(dist_cbd)            AS dist_cbd,
            AVG(dist_primary_school) AS dist_primary_school,
            AVG(dist_major_mall)     AS dist_major_mall
        FROM resale_prices
        GROUP BY block, street_name, town
    """).fetchall()

    data = [
        (
            r["block"], r["street_name"],
            town_id_map.get(r["town"]),
            r["full_address"],
            r["latitude"], r["longitude"],
            r["dist_mrt"], r["dist_cbd"],
            r["dist_primary_school"], r["dist_major_mall"],
        )
        for r in rows
    ]
    returned_blocks = psycopg2.extras.execute_values(
        pg_cur,
        """
        INSERT INTO blocks
            (block, street_name, town_id, full_address, latitude, longitude,
             dist_mrt, dist_cbd, dist_primary_school, dist_major_mall)
        VALUES %s
        ON CONFLICT (block, street_name) DO UPDATE SET
        town_id = EXCLUDED.town_id,
        full_address = EXCLUDED.full_address,
        latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude,
        dist_mrt = EXCLUDED.dist_mrt,
        dist_cbd = EXCLUDED.dist_cbd,
        dist_primary_school = EXCLUDED.dist_primary_school,
        dist_major_mall = EXCLUDED.dist_major_mall
        RETURNING id, block, street_name
        """,
        data,
        page_size=1000,
        fetch=True,
    )
    pg_conn.commit()
    print(f"  Done — {len(data)} blocks.\n")

    block_id_map = {(block, street_name): id_ for id_, block, street_name in returned_blocks}

    # ── 5. Transactions ───────────────────────────────────────────
    print("  Truncating existing transactions...")
    pg_cur.execute("TRUNCATE TABLE transactions RESTART IDENTITY")
    pg_conn.commit()
    total = sqlite_conn.execute("SELECT COUNT(*) FROM resale_prices").fetchone()[0]
    print(f"Step 5/5: Migrating {total:,} transactions (batch size={BATCH_SIZE})...")

    last_rowid = 0
    inserted = 0

    while True:
        rows = sqlite_conn.execute("""
            SELECT
                rowid,
                block, street_name, flat_type, flat_model,
                storey_range, storey_midpoint, floor_area_sqm,
                lease_commence_date, remaining_lease, remaining_lease_months,
                resale_price, month, month_num, year
            FROM resale_prices
            WHERE rowid > ?
            ORDER BY rowid
            LIMIT ?
        """, (last_rowid, BATCH_SIZE)).fetchall()

        if not rows:
            break

        _raise_if_missing_dimension_ids(rows, block_id_map, flat_type_id_map, flat_model_id_map)

        data = []
        for r in rows:
            data.append((
                block_id_map.get((r["block"], r["street_name"])),
                flat_type_id_map.get(r["flat_type"]),
                flat_model_id_map.get(r["flat_model"]),
                r["storey_range"], r["storey_midpoint"], r["floor_area_sqm"],
                r["lease_commence_date"], r["remaining_lease"], r["remaining_lease_months"],
                r["resale_price"], r["month"], r["month_num"], r["year"],
            ))

        psycopg2.extras.execute_values(
            pg_cur,
            """
            INSERT INTO transactions
                (block_id, flat_type_id, flat_model_id,
                 storey_range, storey_midpoint, floor_area_sqm,
                 lease_commence_date, remaining_lease, remaining_lease_months,
                 resale_price, month, month_num, year)
            VALUES %s
            """,
            data,
            page_size=1000,
        )
        pg_conn.commit()

        inserted += len(rows)
        last_rowid = rows[-1]["rowid"]
        pct = inserted * 100 // total
        print(f"  {inserted:,}/{total:,} ({pct}%)", end="\r")

    print(f"\n  Done — {inserted:,} transactions.\n")
    print("Migration complete!")

    sqlite_conn.close()
    pg_conn.close()


if __name__ == "__main__":
    migrate()
