"""
geocoding.py
============
Single responsibility: resolve every unique full_address in resale_prices
to (latitude, longitude) via the OneMap API, write results back to the DB,
and cache all results in geocode_cache to avoid redundant API calls on
re-runs.

Design decisions:
  - Cache-first strategy: only addresses with latitude IS NULL that are not
    already in geocode_cache are sent to the OneMap API.
  - Batch processing with BATCH_SIZE to flush cache periodically and reduce
    data-loss exposure on interruption.
  - Rate limiting enforced by RATE_LIMIT_RPS (calls per second).
  - Exponential backoff on HTTP 429 up to MAX_RETRIES.
  - A single bulk UPDATE FROM geocode_cache is used at the end to minimise
    individual SQLite writes.

Execution order context:
  Step 3 of 6. Reads/writes: hdb_resale.db (resale_prices, geocode_cache).
  Previous step: data_pipeline.py. Next step: proximity_features.py.

Run:
    python geocoding.py
"""

import logging
import itertools
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DB_PATH          = os.environ.get("HDB_SQLITE_PATH", os.path.join(BASE_DIR, "hdb_resale.db"))
ONEMAP_API_URL   = "https://www.onemap.gov.sg/api/common/elastic/search"
BATCH_SIZE       = 50     # addresses per batch before flushing cache to DB
RATE_LIMIT_RPS   = 3      # maximum API calls per second
MAX_RETRIES      = 5      # retries on HTTP 429
RETRY_BASE_DELAY = 2.0    # seconds; doubled each retry attempt
MAX_SEARCH_VARIANTS = 12  # cap derived OneMap query variants per address

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Minimum interval between API calls derived from RATE_LIMIT_RPS
_MIN_CALL_INTERVAL = 1.0 / RATE_LIMIT_RPS

_TOKEN_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "C'WEALTH": ("COMMONWEALTH",),
    "CWEALTH": ("COMMONWEALTH",),
    "PK": ("PARK",),
    "RD": ("ROAD",),
    "AVE": ("AVENUE",),
    "CRES": ("CRESCENT",),
    "CL": ("CLOSE",),
    "DR": ("DRIVE",),
    "CTR": ("CENTRE",),
    "CTRL": ("CENTRAL",),
    "MKT": ("MARKET",),
    "GDNS": ("GARDENS",),
    "UPP": ("UPPER",),
    "NTH": ("NORTH",),
    "STH": ("SOUTH",),
    "BT": ("BUKIT",),
    "KG": ("KAMPONG",),
    "JLN": ("JALAN",),
    "LOR": ("LORONG",),
}
_BLOCK_TOKEN_RE = re.compile(r"^\d+[A-Z]?$")
_TOKEN_RE = re.compile(r"[A-Z0-9]+(?:'[A-Z0-9]+)?")


def _normalise_spaces(text: str) -> str:
    """Collapse repeated whitespace and trim ends."""
    return re.sub(r"\s+", " ", text).strip()


def _split_address(address: str) -> tuple[str, str, str]:
    """
    Split an address into block token, street portion, and optional country suffix.
    """
    cleaned = _normalise_spaces(address)
    suffix = ""
    core = cleaned
    if cleaned.upper().endswith(" SINGAPORE"):
        suffix = "SINGAPORE"
        core = cleaned[:-len(" SINGAPORE")].strip()

    parts = core.split(maxsplit=1)
    if len(parts) == 2 and _BLOCK_TOKEN_RE.fullmatch(parts[0].upper()):
        return parts[0].upper(), parts[1], suffix
    return "", core, suffix


def _tokenise_street(street: str) -> list[str]:
    """Tokenise the street name into uppercase address words."""
    upper = _normalise_spaces(street).upper().replace(".", " ")
    return _TOKEN_RE.findall(upper)


def _street_token_options(tokens: list[str], idx: int) -> list[str]:
    """
    Return ordered token alternatives for dynamic address normalization.

    `ST` is handled contextually:
      - mid-street token -> prefer SAINT
      - final token      -> prefer STREET
    """
    token = tokens[idx]
    options: list[str] = []

    if token == "ST":
        preferred = "SAINT" if idx < len(tokens) - 1 else "STREET"
        options.extend([preferred, token])
    else:
        expanded = _TOKEN_EXPANSIONS.get(token, ())
        options.extend(expanded)
        options.append(token)

    if "'" in token:
        options.append(token.replace("'", ""))

    deduped: list[str] = []
    seen: set[str] = set()
    for option in options:
        cleaned = _normalise_spaces(option).upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped or [token]


def _street_variants(street: str) -> list[str]:
    """
    Build a small set of dynamic street-name variants from token expansions.
    """
    tokens = _tokenise_street(street)
    if not tokens:
        return []

    option_lists = [_street_token_options(tokens, idx) for idx in range(len(tokens))]
    variants: list[str] = []
    seen: set[str] = set()

    for combo in itertools.product(*option_lists):
        variant = " ".join(combo)
        if variant in seen:
            continue
        seen.add(variant)
        variants.append(variant)
        if len(variants) >= MAX_SEARCH_VARIANTS:
            break

    return variants


def _build_search_candidates(address: str) -> list[str]:
    """
    Generate dynamic OneMap query variants from the address tokens.

    This avoids maintaining a manual per-address fix list and instead
    derives cleaner search strings by expanding known abbreviations and
    punctuation patterns generically.
    """
    raw = _normalise_spaces(address)
    raw_upper = raw.upper()
    block, street, suffix = _split_address(raw)

    candidates = [raw, raw_upper]

    for street_variant in _street_variants(street):
        full = " ".join(part for part in [block, street_variant, suffix] if part)
        no_country = " ".join(part for part in [block, street_variant] if part)
        candidates.extend([full, no_country])
        if "'" in full:
            candidates.extend([full.replace("'", ""), no_country.replace("'", "")])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _normalise_spaces(candidate)
        if not cleaned:
            continue
        key = cleaned.upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


# ---------------------------------------------------------------------------
# Core geocoding
# ---------------------------------------------------------------------------

def geocode_address(
    session: requests.Session,
    address: str,
) -> tuple[float, float] | None:
    """
    Resolve a single address string to (latitude, longitude) via OneMap API.

    Enforces RATE_LIMIT_RPS with a minimum sleep between calls. Applies
    exponential backoff on HTTP 429 up to MAX_RETRIES.

    Parameters:
        session: Shared requests.Session for connection reuse.
        address: Address string to geocode (e.g. '123 ANG MO KIO AVE 3 SINGAPORE').

    Returns:
        (latitude, longitude) tuple on success, or None if the address
        could not be resolved (no results or API error).
    """
    candidates = _build_search_candidates(address)

    for candidate in candidates:
        params: dict[str, Any] = {
            "searchVal":      candidate,
            "returnGeom":     "Y",
            "getAddrDetails": "N",
            "pageNum":        1,
        }

        # Enforce rate limit before each call
        time.sleep(_MIN_CALL_INTERVAL)

        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(ONEMAP_API_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Rate limited (429) for address '%s'. "
                        "Waiting %.1fs ... (retry %d/%d)",
                        candidate, wait, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Request error for address '%s': %s. Retrying in %.1fs ...",
                        candidate, exc, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.warning(
                    "All retries exhausted for address '%s': %s", candidate, exc
                )
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            try:
                lat = float(results[0]["LATITUDE"])
                lng = float(results[0]["LONGITUDE"])
                if candidate != address:
                    logger.info(
                        "Resolved '%s' using normalised query '%s'",
                        address, candidate,
                    )
                return (lat, lng)
            except (KeyError, ValueError, IndexError) as exc:
                logger.warning(
                    "Could not parse coordinates for address '%s': %s", candidate, exc
                )
                break

    logger.warning(
        "No results from OneMap for address: '%s' (tried %d variants)",
        address,
        len(candidates),
    )
    return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(conn: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    """
    Read all rows from geocode_cache and return as an in-memory lookup dict.

    Parameters:
        conn: Open SQLite connection with geocode_cache table present.

    Returns:
        Dict mapping full_address → (latitude, longitude).
    """
    rows = conn.execute(
        "SELECT full_address, latitude, longitude FROM geocode_cache"
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def save_cache_batch(
    conn: sqlite3.Connection,
    batch: list[tuple[str, float, float, str]],
) -> None:
    """
    Insert a batch of geocode results into geocode_cache using INSERT OR IGNORE
    (existing entries from previous runs are preserved).

    Parameters:
        conn:  Open SQLite connection.
        batch: List of (full_address, latitude, longitude, fetched_at) tuples.
    """
    conn.executemany(
        "INSERT OR IGNORE INTO geocode_cache (full_address, latitude, longitude, fetched_at) "
        "VALUES (?, ?, ?, ?)",
        batch,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_geocoding(db_path: str = DB_PATH) -> None:
    """
    Resolve latitude/longitude for every ungeocoded address in resale_prices.

    Workflow:
      1. Load all unique full_address values where latitude IS NULL.
      2. Apply cache hits from geocode_cache without touching the API.
      3. Geocode remaining addresses in batches, flushing cache each batch.
      4. Run a single SQL UPDATE to write lat/lng from geocode_cache into
         resale_prices for all newly resolved addresses.

    Parameters:
        db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)

    # 1. Load addresses that still need geocoding
    try:
        pending_rows = conn.execute(
            "SELECT DISTINCT full_address FROM resale_prices WHERE latitude IS NULL"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        conn.close()
        raise RuntimeError(
            "resale_prices table not found in the database. "
            "Run data_pipeline.py first to populate the DB."
        ) from exc
    pending = [row[0] for row in pending_rows if row[0] is not None]
    total = len(pending)
    print(f"\n  Addresses pending geocoding: {total:,}")

    if total == 0:
        print("  All addresses already geocoded. Nothing to do.")
        conn.close()
        return

    # 2. Apply cache hits
    cache = load_cache(conn)
    cache_hits = [addr for addr in pending if addr in cache]
    to_geocode = [addr for addr in pending if addr not in cache]
    print(f"  Cache hits: {len(cache_hits):,}  |  API calls needed: {len(to_geocode):,}")

    # 3. Geocode remaining in batches
    api_calls = 0
    failed = 0
    new_cache_entries: list[tuple[str, float, float, str]] = []
    session = requests.Session()

    for i, address in enumerate(to_geocode):
        result = geocode_address(session, address)
        api_calls += 1

        if result is not None:
            lat, lng = result
            fetched_at = datetime.now(timezone.utc).isoformat()
            new_cache_entries.append((address, lat, lng, fetched_at))
            cache[address] = (lat, lng)
        else:
            failed += 1

        # Flush cache batch to DB periodically
        if len(new_cache_entries) >= BATCH_SIZE or i == len(to_geocode) - 1:
            if new_cache_entries:
                save_cache_batch(conn, new_cache_entries)
                print(
                    f"    Flushed {len(new_cache_entries)} entries to geocode_cache "
                    f"(progress: {i + 1}/{len(to_geocode)})"
                )
                new_cache_entries = []

    session.close()

    # 4. Bulk UPDATE resale_prices from geocode_cache
    conn.execute("""
        UPDATE resale_prices
        SET
            latitude  = gc.latitude,
            longitude = gc.longitude
        FROM geocode_cache AS gc
        WHERE resale_prices.full_address = gc.full_address
          AND resale_prices.latitude IS NULL
    """)
    conn.commit()

    updated = conn.execute(
        "SELECT COUNT(*) FROM resale_prices WHERE latitude IS NOT NULL"
    ).fetchone()[0]
    still_null = conn.execute(
        "SELECT COUNT(*) FROM resale_prices WHERE latitude IS NULL"
    ).fetchone()[0]
    conn.close()

    print(f"\n  Geocoding complete.")
    print(f"    Total addresses processed : {total:,}")
    print(f"    Cache hits                : {len(cache_hits):,}")
    print(f"    API calls made            : {api_calls:,}")
    print(f"    Failed / unresolved       : {failed:,}")
    print(f"    resale_prices rows geocoded: {updated:,}")
    print(f"    resale_prices still null   : {still_null:,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Resolve all pending addresses in the DB and update resale_prices."""
    print("=" * 60)
    print("HDB Resale — Geocoding")
    print("=" * 60)
    run_geocoding()
    print(f"\nDone. hdb_resale.db updated.")


if __name__ == "__main__":
    main()
