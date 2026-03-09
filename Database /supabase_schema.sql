-- ============================================================
-- Supabase Normalized Schema — HDB Resale Analytics Platform
-- Run this in Supabase SQL Editor (Settings > SQL Editor)
-- ============================================================

-- ── 1. Dimension Tables ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS towns (
    id               SERIAL PRIMARY KEY,
    name             TEXT    UNIQUE NOT NULL,
    is_mature_estate BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS flat_types (
    id      SERIAL PRIMARY KEY,
    name    TEXT    UNIQUE NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS flat_models (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- ── 2. Address / Location Table ───────────────────────────────

CREATE TABLE IF NOT EXISTS blocks (
    id                  SERIAL PRIMARY KEY,
    block               TEXT   NOT NULL,
    street_name         TEXT   NOT NULL,
    town_id             INTEGER REFERENCES towns(id),
    full_address        TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    dist_mrt            DOUBLE PRECISION,
    dist_cbd            DOUBLE PRECISION,
    dist_primary_school DOUBLE PRECISION,
    dist_major_mall     DOUBLE PRECISION,
    UNIQUE (block, street_name)
);

CREATE INDEX IF NOT EXISTS idx_blocks_town     ON blocks(town_id);
CREATE INDEX IF NOT EXISTS idx_blocks_location ON blocks(latitude, longitude);

-- ── 3. Fact Table (Transactions) ─────────────────────────────

CREATE TABLE IF NOT EXISTS transactions (
    id                     SERIAL PRIMARY KEY,
    block_id               INTEGER REFERENCES blocks(id),
    flat_type_id           INTEGER REFERENCES flat_types(id),
    flat_model_id          INTEGER REFERENCES flat_models(id),
    storey_range           TEXT             NOT NULL,
    storey_midpoint        DOUBLE PRECISION,
    floor_area_sqm         DOUBLE PRECISION NOT NULL,
    lease_commence_date    INTEGER          NOT NULL,
    remaining_lease        DOUBLE PRECISION,
    remaining_lease_months DOUBLE PRECISION,
    resale_price           DOUBLE PRECISION NOT NULL,
    month                  TEXT             NOT NULL,
    month_num              INTEGER,
    year                   INTEGER          NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_txn_year      ON transactions(year);
CREATE INDEX IF NOT EXISTS idx_txn_block     ON transactions(block_id);
CREATE INDEX IF NOT EXISTS idx_txn_flat_type ON transactions(flat_type_id);
CREATE INDEX IF NOT EXISTS idx_txn_year_block ON transactions(year, block_id);

-- ── 4. User Tables ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS saved_predictions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    town            TEXT             NOT NULL,
    flat_type       TEXT             NOT NULL,
    flat_model      TEXT             NOT NULL,
    floor_area      DOUBLE PRECISION NOT NULL,
    storey_range    TEXT             NOT NULL,
    lease_commence  INTEGER          NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL,
    price_low       DOUBLE PRECISION NOT NULL,
    price_high      DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── 5. Model Versions Table ───────────────────────────────────

CREATE TABLE IF NOT EXISTS model_versions (
    id         SERIAL PRIMARY KEY,
    version    TEXT    NOT NULL,
    run_dir    TEXT,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    test_mape  DOUBLE PRECISION,
    test_rmse  DOUBLE PRECISION,
    test_r2    DOUBLE PRECISION,
    notes      TEXT,
    is_active  BOOLEAN DEFAULT FALSE
);

-- ── 6. RPC Functions (called by Flask app) ────────────────────

-- Towns list
CREATE OR REPLACE FUNCTION rpc_get_towns()
RETURNS TABLE(town TEXT) LANGUAGE SQL STABLE AS $$
    SELECT name FROM towns ORDER BY name;
$$;

-- Flat models list
CREATE OR REPLACE FUNCTION rpc_get_flat_models()
RETURNS TABLE(flat_model TEXT) LANGUAGE SQL STABLE AS $$
    SELECT name FROM flat_models ORDER BY name;
$$;

-- Town average distances (used by prediction engine)
CREATE OR REPLACE FUNCTION rpc_get_town_avg_distances()
RETURNS TABLE(
    town TEXT, avg_dist_mrt DOUBLE PRECISION, avg_dist_cbd DOUBLE PRECISION,
    avg_dist_school DOUBLE PRECISION, avg_dist_mall DOUBLE PRECISION,
    avg_lat DOUBLE PRECISION, avg_lng DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        AVG(b.dist_mrt),
        AVG(b.dist_cbd),
        AVG(b.dist_primary_school),
        AVG(b.dist_major_mall),
        AVG(b.latitude),
        AVG(b.longitude)
    FROM blocks b
    JOIN towns t ON b.town_id = t.id
    WHERE b.dist_mrt IS NOT NULL
    GROUP BY t.name;
$$;

-- Recent transactions for map
CREATE OR REPLACE FUNCTION rpc_api_transactions(p_town TEXT DEFAULT NULL, p_limit INTEGER DEFAULT 500)
RETURNS TABLE(
    town TEXT, flat_type TEXT, block TEXT, street_name TEXT,
    storey_range TEXT, floor_area_sqm DOUBLE PRECISION,
    resale_price DOUBLE PRECISION, month TEXT, year INTEGER,
    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name, ft.name, b.block, b.street_name,
        tx.storey_range, tx.floor_area_sqm,
        tx.resale_price, tx.month, tx.year,
        b.latitude, b.longitude
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE b.latitude IS NOT NULL AND b.longitude IS NOT NULL
      AND (p_town IS NULL OR t.name = p_town)
    ORDER BY tx.year DESC, tx.month_num DESC
    LIMIT p_limit;
$$;

-- District summary for heatmap
CREATE OR REPLACE FUNCTION rpc_api_district_summary()
RETURNS TABLE(
    town TEXT, avg_price DOUBLE PRECISION, recent_avg DOUBLE PRECISION,
    total_txns BIGINT, recent_txns BIGINT,
    lat DOUBLE PRECISION, lng DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(AVG(CASE WHEN tx.year >= 2023 THEN tx.resale_price END)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        SUM(CASE WHEN tx.year >= 2023 THEN 1 ELSE 0 END),
        AVG(b.latitude),
        AVG(b.longitude)
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns  t ON b.town_id   = t.id
    WHERE b.latitude IS NOT NULL
    GROUP BY t.name
    ORDER BY t.name;
$$;

-- Yearly price trend (simple, no percentile)
CREATE OR REPLACE FUNCTION rpc_api_price_trend_simple(p_town TEXT DEFAULT NULL, p_flat_type TEXT DEFAULT NULL)
RETURNS TABLE(
    year INTEGER, avg_price DOUBLE PRECISION,
    min_price DOUBLE PRECISION, max_price DOUBLE PRECISION, txn_count BIGINT
) LANGUAGE SQL STABLE AS $$
    SELECT
        tx.year,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(MIN(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(MAX(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*)
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE (p_town IS NULL OR t.name = p_town)
      AND (p_flat_type IS NULL OR ft.name = p_flat_type)
    GROUP BY tx.year
    ORDER BY tx.year;
$$;

-- District comparison (most recent year)
CREATE OR REPLACE FUNCTION rpc_api_district_comparison()
RETURNS TABLE(
    town TEXT, avg_price DOUBLE PRECISION, txn_count BIGINT,
    avg_area DOUBLE PRECISION, psf DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION,
        ROUND(AVG(tx.resale_price / NULLIF(tx.floor_area_sqm, 0))::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns  t ON b.town_id   = t.id
    WHERE tx.year = (SELECT MAX(year) FROM transactions)
    GROUP BY t.name
    ORDER BY AVG(tx.resale_price) DESC;
$$;

-- Flat type breakdown for a town
CREATE OR REPLACE FUNCTION rpc_api_flat_type_breakdown(p_town TEXT DEFAULT NULL)
RETURNS TABLE(
    flat_type TEXT, avg_price DOUBLE PRECISION,
    txn_count BIGINT, avg_area DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        ft.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE tx.year >= 2023
      AND (p_town IS NULL OR t.name = p_town)
    GROUP BY ft.name
    ORDER BY ft.name;
$$;

-- Monthly transaction volume
CREATE OR REPLACE FUNCTION rpc_api_monthly_volume(p_town TEXT DEFAULT NULL)
RETURNS TABLE(
    month INTEGER, txn_count BIGINT, avg_price DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        tx.month_num,
        COUNT(*),
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns  t ON b.town_id   = t.id
    WHERE (p_town IS NULL OR t.name = p_town)
    GROUP BY tx.month_num
    ORDER BY tx.month_num;
$$;

-- Price trend for predict page
CREATE OR REPLACE FUNCTION rpc_predict_trend(p_town TEXT, p_flat_type TEXT)
RETURNS TABLE(year INTEGER, avg_price DOUBLE PRECISION, txn_count BIGINT)
LANGUAGE SQL STABLE AS $$
    SELECT
        tx.year,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*)
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND ft.name = p_flat_type
    GROUP BY tx.year
    ORDER BY tx.year;
$$;

-- Benchmarks for predict page
CREATE OR REPLACE FUNCTION rpc_predict_benchmarks(p_town TEXT)
RETURNS TABLE(
    flat_type TEXT, avg_price DOUBLE PRECISION,
    txn_count BIGINT, avg_area DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        ft.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND tx.year >= 2023
    GROUP BY ft.name
    ORDER BY ft.name;
$$;

-- Resolve floor area for prediction (town + flat_type average)
CREATE OR REPLACE FUNCTION rpc_resolve_floor_area(p_town TEXT, p_flat_type TEXT)
RETURNS DOUBLE PRECISION LANGUAGE SQL STABLE AS $$
    SELECT ROUND(AVG(tx.floor_area_sqm)::NUMERIC, 1)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND ft.name = p_flat_type;
$$;

-- Resolve lease commence for prediction (town + flat_type average)
CREATE OR REPLACE FUNCTION rpc_resolve_lease_commence(p_town TEXT, p_flat_type TEXT)
RETURNS INTEGER LANGUAGE SQL STABLE AS $$
    SELECT ROUND(AVG(tx.lease_commence_date))::INTEGER
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND ft.name = p_flat_type;
$$;

-- ── 7. Block-Level & Analytics RPC Functions ────────────────────

-- Streets for a town (block-level comparison)
CREATE OR REPLACE FUNCTION rpc_available_streets(p_town TEXT)
RETURNS TABLE(street_name TEXT) LANGUAGE SQL STABLE AS $$
    SELECT DISTINCT b.street_name
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town
    ORDER BY b.street_name;
$$;

-- Blocks for a town + street
CREATE OR REPLACE FUNCTION rpc_available_blocks(p_town TEXT, p_street TEXT)
RETURNS TABLE(block TEXT) LANGUAGE SQL STABLE AS $$
    SELECT DISTINCT b.block
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town AND b.street_name = p_street
    ORDER BY b.block;
$$;

-- Block-level distances for prediction
CREATE OR REPLACE FUNCTION rpc_block_distances(p_town TEXT, p_street TEXT, p_block TEXT)
RETURNS TABLE(dist_mrt DOUBLE PRECISION, dist_cbd DOUBLE PRECISION,
              dist_school DOUBLE PRECISION, dist_mall DOUBLE PRECISION)
LANGUAGE SQL STABLE AS $$
    SELECT b.dist_mrt, b.dist_cbd, b.dist_primary_school, b.dist_major_mall
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town AND b.street_name = p_street AND b.block = p_block
    LIMIT 1;
$$;

-- Lease decay data for analytics
CREATE OR REPLACE FUNCTION rpc_lease_decay(p_town TEXT, p_flat_type TEXT DEFAULT NULL)
RETURNS TABLE(lease_bucket INTEGER, avg_price DOUBLE PRECISION, txn_count BIGINT)
LANGUAGE SQL STABLE AS $$
    SELECT
        (CAST(tx.remaining_lease AS INT) / 10) * 10,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*)
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town
      AND (p_flat_type IS NULL OR ft.name = p_flat_type)
    GROUP BY (CAST(tx.remaining_lease AS INT) / 10) * 10
    ORDER BY 1 DESC;
$$;

-- Recent similar transactions for prediction context
CREATE OR REPLACE FUNCTION rpc_recent_similar_transactions(p_town TEXT, p_flat_type TEXT, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(
    block TEXT, street_name TEXT, storey_range TEXT,
    floor_area_sqm DOUBLE PRECISION, resale_price DOUBLE PRECISION, month TEXT
) LANGUAGE SQL STABLE AS $$
    SELECT b.block, b.street_name, tx.storey_range,
           tx.floor_area_sqm, tx.resale_price, tx.month
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND ft.name = p_flat_type
    ORDER BY tx.year DESC, tx.month_num DESC
    LIMIT p_limit;
$$;
