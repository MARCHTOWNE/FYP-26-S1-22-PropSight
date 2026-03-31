from __future__ import annotations

import os
import sqlite3
import time
import uuid
from collections.abc import Sequence

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some local setups
    load_dotenv = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if load_dotenv is not None:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "").strip()
SQLITE_DB_PATH = os.environ.get(
    "HDB_SQLITE_PATH",
    os.path.join(PROJECT_ROOT, "Data Preprocessing", "hdb_resale.db"),
)
TRAINING_DATA_SOURCE = os.environ.get("TRAINING_DATA_SOURCE", "auto").strip().lower()
SUPABASE_CONNECT_TIMEOUT = int(os.environ.get("SUPABASE_CONNECT_TIMEOUT", "15"))
SUPABASE_STATEMENT_TIMEOUT_MS = int(
    os.environ.get("SUPABASE_STATEMENT_TIMEOUT_MS", "0")
)
SUPABASE_FETCH_BATCH_SIZE = int(os.environ.get("SUPABASE_FETCH_BATCH_SIZE", "50000"))
SQLITE_FETCH_BATCH_SIZE = int(os.environ.get("SQLITE_FETCH_BATCH_SIZE", "100000"))

_SUPABASE_TRAINING_QUERY = """
    SELECT
        tx.month,
        tx.year,
        tx.month_num,
        t.name AS town,
        ft.name AS flat_type,
        b.block,
        b.street_name,
        tx.storey_range,
        tx.storey_midpoint,
        tx.floor_area_sqm,
        fm.name AS flat_model,
        tx.lease_commence_date,
        tx.remaining_lease,
        tx.remaining_lease_months,
        tx.resale_price,
        b.full_address,
        b.latitude,
        b.longitude,
        b.dist_mrt,
        b.dist_cbd,
        b.dist_primary_school,
        b.dist_major_mall
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    JOIN flat_models fm ON tx.flat_model_id = fm.id
"""

_SUPABASE_TRAINING_COUNT_QUERY = """
    SELECT COUNT(*)
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    JOIN flat_models fm ON tx.flat_model_id = fm.id
"""

_SQLITE_TRAINING_QUERY = """
    SELECT
        month,
        year,
        month_num,
        town,
        flat_type,
        block,
        street_name,
        storey_range,
        storey_midpoint,
        floor_area_sqm,
        flat_model,
        lease_commence_date,
        remaining_lease,
        remaining_lease_months,
        resale_price,
        full_address,
        latitude,
        longitude,
        dist_mrt,
        dist_cbd,
        dist_primary_school,
        dist_major_mall
    FROM resale_prices
"""


def _resolve_requested_source() -> str:
    source = TRAINING_DATA_SOURCE or "auto"
    if source not in {"auto", "supabase", "sqlite"}:
        raise RuntimeError(
            "TRAINING_DATA_SOURCE must be one of: auto, supabase, sqlite."
        )
    if source == "auto":
        return "supabase" if SUPABASE_DB_URL else "sqlite"
    return source


def _sqlite_available() -> bool:
    return os.path.exists(SQLITE_DB_PATH)


def _select_source_for_fallback(preferred_source: str, exc: Exception) -> str:
    if preferred_source == "supabase" and TRAINING_DATA_SOURCE == "auto" and _sqlite_available():
        print(
            f"  WARNING: Supabase training source unavailable ({exc}). "
            f"Falling back to local SQLite at {SQLITE_DB_PATH}.",
            flush=True,
        )
        return "sqlite"
    raise exc


def get_training_data_source_name() -> str:
    source = _resolve_requested_source()
    if source == "supabase":
        if not SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL is not configured.")
        return "supabase"
    if source == "sqlite":
        if not _sqlite_available():
            raise RuntimeError(f"SQLite training DB not found at '{SQLITE_DB_PATH}'.")
        return "sqlite"
    raise RuntimeError(f"Unsupported training data source: {source}")


def _connect_supabase():
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not configured.")

    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "psycopg2 is required to use Supabase as the training data source."
        ) from exc

    conn = psycopg2.connect(
        SUPABASE_DB_URL,
        connect_timeout=SUPABASE_CONNECT_TIMEOUT,
        application_name="hdb_resale_ml_pipeline",
    )
    if SUPABASE_STATEMENT_TIMEOUT_MS > 0:
        with conn.cursor() as cur:
            cur.execute(
                "SET statement_timeout = %s",
                (int(SUPABASE_STATEMENT_TIMEOUT_MS),),
            )
    return conn


def _connect_sqlite() -> sqlite3.Connection:
    if not _sqlite_available():
        raise RuntimeError(f"SQLite training DB not found at '{SQLITE_DB_PATH}'.")
    return sqlite3.connect(SQLITE_DB_PATH)


def _fetch_training_row_count(conn) -> int | None:
    try:
        with conn.cursor() as cur:
            cur.execute(_SUPABASE_TRAINING_COUNT_QUERY)
            row = cur.fetchone()
    except Exception:
        return None
    return int(row[0]) if row and row[0] is not None else None


def _fetch_sqlite_row_count(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute("SELECT COUNT(*) FROM resale_prices").fetchone()
    except Exception:
        return None
    return int(row[0]) if row and row[0] is not None else None


def _get_query_columns(
    conn,
    query: str,
    params: Sequence[object] | None = None,
) -> list[str]:
    probe_query = f"SELECT * FROM ({query}) AS training_extract LIMIT 0"
    with conn.cursor() as cur:
        cur.execute(probe_query, params or ())
        return [desc[0] for desc in cur.description]


def _read_query_via_server_cursor(
    conn,
    query: str,
    params: Sequence[object] | None = None,
    *,
    batch_size: int = SUPABASE_FETCH_BATCH_SIZE,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("SUPABASE_FETCH_BATCH_SIZE must be greater than 0.")

    cursor_name = f"training_extract_{uuid.uuid4().hex}"
    columns = _get_query_columns(conn, query, params)
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    started_at = time.time()

    with conn.cursor(name=cursor_name) as cur:
        cur.itersize = batch_size
        cur.execute(query, params or ())

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            chunks.append(pd.DataFrame.from_records(rows, columns=columns))
            rows_read += len(rows)

            elapsed = time.time() - started_at
            rate = rows_read / elapsed if elapsed > 0 else 0.0
            if expected_rows:
                pct = rows_read / expected_rows * 100
                print(
                    f"  Fetched {rows_read:,}/{expected_rows:,} rows "
                    f"({pct:.1f}%) at ~{rate:,.0f} rows/s.",
                    flush=True,
                )
            else:
                print(
                    f"  Fetched {rows_read:,} rows at ~{rate:,.0f} rows/s.",
                    flush=True,
                )

    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def _read_sqlite_dataframe(
    conn: sqlite3.Connection,
    query: str,
    *,
    batch_size: int = SQLITE_FETCH_BATCH_SIZE,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("SQLITE_FETCH_BATCH_SIZE must be greater than 0.")

    chunks: list[pd.DataFrame] = []
    rows_read = 0
    started_at = time.time()

    for chunk in pd.read_sql_query(query, conn, chunksize=batch_size):
        chunks.append(chunk)
        rows_read += len(chunk)
        elapsed = time.time() - started_at
        rate = rows_read / elapsed if elapsed > 0 else 0.0
        if expected_rows:
            pct = rows_read / expected_rows * 100
            print(
                f"  Fetched {rows_read:,}/{expected_rows:,} rows "
                f"({pct:.1f}%) at ~{rate:,.0f} rows/s.",
                flush=True,
            )
        else:
            print(
                f"  Fetched {rows_read:,} rows at ~{rate:,.0f} rows/s.",
                flush=True,
            )

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_training_dataframe() -> tuple[pd.DataFrame, str]:
    preferred_source = _resolve_requested_source()

    if preferred_source == "supabase":
        try:
            conn = _connect_supabase()
            try:
                expected_rows = _fetch_training_row_count(conn)
                if expected_rows is not None:
                    print(
                        f"  Supabase training extract expects {expected_rows:,} joined rows.",
                        flush=True,
                    )
                df = _read_query_via_server_cursor(
                    conn,
                    _SUPABASE_TRAINING_QUERY,
                    expected_rows=expected_rows,
                )
                return df, "supabase"
            finally:
                conn.close()
        except Exception as exc:
            fallback_source = _select_source_for_fallback(preferred_source, exc)
            if fallback_source != "sqlite":
                raise

    conn = _connect_sqlite()
    try:
        expected_rows = _fetch_sqlite_row_count(conn)
        if expected_rows is not None:
            print(
                f"  SQLite training extract expects {expected_rows:,} rows.",
                flush=True,
            )
        df = _read_sqlite_dataframe(
            conn,
            _SQLITE_TRAINING_QUERY,
            expected_rows=expected_rows,
        )
    finally:
        conn.close()
    return df, "sqlite"


def get_training_source_summary(
    *,
    test_start_year: int | None = None,
    test_end_year: int | None = None,
    train_end_year: int | None = None,
) -> dict[str, object]:
    preferred_source = _resolve_requested_source()
    if preferred_source == "supabase":
        try:
            get_training_data_source_name()
            return _get_supabase_summary(
                test_start_year=test_start_year,
                test_end_year=test_end_year,
                train_end_year=train_end_year,
            )
        except Exception as exc:
            fallback_source = _select_source_for_fallback(preferred_source, exc)
            if fallback_source != "sqlite":
                raise

    return _get_sqlite_summary(
        test_start_year=test_start_year,
        test_end_year=test_end_year,
        train_end_year=train_end_year,
    )


def _normalise_year_counts(rows: Sequence[Sequence[object]]) -> list[dict[str, int]]:
    return [
        {"year": int(year), "count": int(count)}
        for year, count in rows
        if year is not None
    ]


def _normalise_flat_type_breakdown(
    rows: Sequence[Sequence[object]],
) -> list[dict[str, int | str]]:
    return [
        {
            "flat_type": str(flat_type),
            "n_train": int(n_train or 0),
            "n_test": int(n_test or 0),
        }
        for flat_type, n_train, n_test in rows
    ]


def _get_supabase_summary(
    *,
    test_start_year: int | None,
    test_end_year: int | None,
    train_end_year: int | None,
) -> dict[str, object]:
    conn = _connect_supabase()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    SUM(CASE WHEN b.latitude IS NOT NULL THEN 1 ELSE 0 END) AS geocoded_rows,
                    SUM(
                        CASE
                            WHEN b.dist_mrt IS NOT NULL
                             AND b.dist_cbd IS NOT NULL
                             AND b.dist_primary_school IS NOT NULL
                             AND b.dist_major_mall IS NOT NULL
                            THEN 1 ELSE 0
                        END
                    ) AS proximity_rows
                FROM transactions tx
                JOIN blocks b ON tx.block_id = b.id
                """
            )
            row = cur.fetchone() or (0, 0, 0)
            total_rows = int(row[0] or 0)
            geocoded_rows = int(row[1] or 0)
            proximity_rows = int(row[2] or 0)

            year_rows: list[dict[str, int]] = []
            if test_start_year is not None:
                if test_end_year is not None:
                    cur.execute(
                        """
                        SELECT tx.year, COUNT(*)
                        FROM transactions tx
                        WHERE tx.year BETWEEN %s AND %s
                        GROUP BY tx.year
                        ORDER BY tx.year
                        """,
                        (int(test_start_year), int(test_end_year)),
                    )
                else:
                    cur.execute(
                        """
                        SELECT tx.year, COUNT(*)
                        FROM transactions tx
                        WHERE tx.year >= %s
                        GROUP BY tx.year
                        ORDER BY tx.year
                        """,
                        (int(test_start_year),),
                    )
                year_rows = _normalise_year_counts(cur.fetchall())

            flat_type_breakdown: list[dict[str, int | str]] = []
            if test_start_year is not None and train_end_year is not None:
                if test_end_year is not None:
                    cur.execute(
                        """
                        SELECT
                            ft.name AS flat_type,
                            SUM(CASE WHEN tx.year <= %s THEN 1 ELSE 0 END) AS n_train,
                            SUM(
                                CASE
                                    WHEN tx.year BETWEEN %s AND %s THEN 1
                                    ELSE 0
                                END
                            ) AS n_test
                        FROM transactions tx
                        JOIN flat_types ft ON tx.flat_type_id = ft.id
                        GROUP BY ft.name
                        ORDER BY ft.name
                        """,
                        (
                            int(train_end_year),
                            int(test_start_year),
                            int(test_end_year),
                        ),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            ft.name AS flat_type,
                            SUM(CASE WHEN tx.year <= %s THEN 1 ELSE 0 END) AS n_train,
                            SUM(CASE WHEN tx.year >= %s THEN 1 ELSE 0 END) AS n_test
                        FROM transactions tx
                        JOIN flat_types ft ON tx.flat_type_id = ft.id
                        GROUP BY ft.name
                        ORDER BY ft.name
                        """,
                        (int(train_end_year), int(test_start_year)),
                    )
                flat_type_breakdown = _normalise_flat_type_breakdown(cur.fetchall())
    finally:
        conn.close()

    return {
        "source": "supabase",
        "total_rows": total_rows,
        "geocoded_rows": geocoded_rows,
        "proximity_rows": proximity_rows,
        "test_rows_by_year": year_rows,
        "flat_type_breakdown": flat_type_breakdown,
    }


def _get_sqlite_summary(
    *,
    test_start_year: int | None = None,
    test_end_year: int | None = None,
    train_end_year: int | None = None,
) -> dict[str, object]:
    conn = _connect_sqlite()
    try:
        total_rows = int(conn.execute("SELECT COUNT(*) FROM resale_prices").fetchone()[0])
        geocoded_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM resale_prices WHERE latitude IS NOT NULL"
            ).fetchone()[0]
        )
        proximity_rows = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM resale_prices
                WHERE dist_mrt IS NOT NULL
                  AND dist_cbd IS NOT NULL
                  AND dist_primary_school IS NOT NULL
                  AND dist_major_mall IS NOT NULL
                """
            ).fetchone()[0]
        )

        year_rows: list[dict[str, int]] = []
        if test_start_year is not None:
            if test_end_year is not None:
                rows = conn.execute(
                    """
                    SELECT year, COUNT(*)
                    FROM resale_prices
                    WHERE year BETWEEN ? AND ?
                    GROUP BY year
                    ORDER BY year
                    """,
                    (int(test_start_year), int(test_end_year)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT year, COUNT(*)
                    FROM resale_prices
                    WHERE year >= ?
                    GROUP BY year
                    ORDER BY year
                    """,
                    (int(test_start_year),),
                ).fetchall()
            year_rows = _normalise_year_counts(rows)

        flat_type_breakdown: list[dict[str, int | str]] = []
        if test_start_year is not None and train_end_year is not None:
            if test_end_year is not None:
                rows = conn.execute(
                    """
                    SELECT
                        flat_type,
                        SUM(CASE WHEN year <= ? THEN 1 ELSE 0 END) AS n_train,
                        SUM(CASE WHEN year BETWEEN ? AND ? THEN 1 ELSE 0 END) AS n_test
                    FROM resale_prices
                    GROUP BY flat_type
                    ORDER BY flat_type
                    """,
                    (
                        int(train_end_year),
                        int(test_start_year),
                        int(test_end_year),
                    ),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        flat_type,
                        SUM(CASE WHEN year <= ? THEN 1 ELSE 0 END) AS n_train,
                        SUM(CASE WHEN year >= ? THEN 1 ELSE 0 END) AS n_test
                    FROM resale_prices
                    GROUP BY flat_type
                    ORDER BY flat_type
                    """,
                    (int(train_end_year), int(test_start_year)),
                ).fetchall()
            flat_type_breakdown = _normalise_flat_type_breakdown(rows)
    finally:
        conn.close()

    return {
        "source": "sqlite",
        "total_rows": total_rows,
        "geocoded_rows": geocoded_rows,
        "proximity_rows": proximity_rows,
        "test_rows_by_year": year_rows,
        "flat_type_breakdown": flat_type_breakdown,
    }
