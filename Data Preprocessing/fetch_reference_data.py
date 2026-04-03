"""
fetch_reference_data.py
=======================
Single responsibility: fetch full MRT station, primary school, hawker
centre, and major shopping mall reference datasets and save them as JSON
files under /reference_data.

Design decisions:
  - MRT stations are fetched from the latest LTA static geospatial TrainStation
    ZIP and converted from SVY21 to WGS84 in pure Python.
  - The MOE school directory is downloaded from data.gov.sg without an API key,
    filtered to primary schools, then geocoded via OneMap.
  - Hawker centre coordinates are fetched from the official NEA data.gov.sg
    geojson dataset.
  - Major shopping mall coordinates are built from a fixed curated mall list
    and geocoded via OneMap.
  - JSON output is overwritten on each run to keep the reference files
    deterministic and easy to refresh.

Run once before:
    python proximity_features.py

Run:
    python fetch_reference_data.py
"""

from __future__ import annotations

import io
import html
import json
import math
import os
import re
import struct
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = os.environ.get("HDB_REFERENCE_DATA_DIR", str(BASE_DIR / "reference_data"))

ONEMAP_API_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
DATA_GOV_API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
PRIMARY_SCHOOL_DATASET_ID = "d_688b934f82c1059ed0a6993d2a829089"
HAWKER_CENTRE_DATASET_ID = "d_4a086da0a5553be1d89383cd90d07ecd"
PRIMARY_SCHOOL_RANKING_URL = "https://www.property2b2c.com/school-ranking/2025-pop"
HIGH_DEMAND_PRIMARY_TOP_N = 100
LTA_STATIC_DATA_URL = "https://datamall.lta.gov.sg/content/datamall/en/static-data.html"
LTA_PAGE_SIZE = 500
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
ONEMAP_RATE_LIMIT_RPS = 3

SHOPPING_MALL_NAMES: list[str] = [
    "VivoCity", "ION Orchard", "Bugis Junction", "Tampines Mall",
    "Jurong Point", "Causeway Point", "Northpoint City", "NEX",
    "AMK Hub", "Bedok Mall", "Waterway Point", "Eastpoint Mall",
    "White Sands", "Lot One", "West Mall", "Clementi Mall",
    "The Star Vista", "Holland Village", "Rochester Mall",
    "Buona Vista MRT", "Plaza Singapura", "Orchard Gateway",
    "313@somerset", "Far East Plaza", "Lucky Plaza",
    "Ngee Ann City", "Wisma Atria", "Mandarin Gallery",
    "The Cathay", "Cineleisure Orchard", "Liat Towers",
    "Shaw Centre", "Wheelock Place", "Paragon", "The Heeren",
    "Raffles City", "Suntec City",
    "Marina Square", "Millenia Walk", "The Shoppes at MBS",
    "Parkway Parade", "112 Katong", "Siglap Centre",
    "Tampines 1", "Century Square", "Our Tampines Hub",
    "Changi City Point", "Jewel Changi Airport", "Downtown East",
    "Pasir Ris Mall", "White Sands", "Hougang Mall",
    "Heartland Mall", "Greenwich V", "Compass One",
    "Rivervale Mall", "Punggol Plaza", "Waterway Point",
    "Junction 8", "Thomson Plaza", "Velocity@Novena Square",
    "United Square", "Novena Square", "Square 2",
    "Toa Payoh HDB Hub", "Balestier Hill", "Mustafa Centre",
    "City Square Mall", "Farrer Park", "Paya Lebar Quarter",
    "Kinex", "Geylang Serai Market", "Kallang Wave Mall",
    "Leisure Park Kallang", "Queensway Shopping Centre",
    "IKEA Alexandra", "Anchorpoint", "Alexandra Retail Centre",
    "Harbourfront Centre", "Telok Blangah Mall",
]

MALL_QUERY_ALIASES: dict[str, list[str]] = {
    "313@somerset": ["313 Somerset", "313@Somerset", "Somerset 313"],
    "Heartland Mall": ["Heartland Mall", "Heartland Mall Kovan", "Kovan Heartland Mall"],
    "Velocity@Novena Square": ["Velocity Novena Square", "Velocity @ Novena Square", "Novena Square Velocity"],
    "Telok Blangah Mall": ["Telok Blangah Mall", "Telok Blangah", "Blk 55 Telok Blangah"],
}

HIGH_DEMAND_PRIMARY_NAME_ALIASES: dict[str, str] = {
    "ANGLO-CHINESE": "ANGLO-CHINESE SCHOOL (PRIMARY)",
    "ANGLO-CHINESE (JUNIOR)": "ANGLO-CHINESE SCHOOL (JUNIOR)",
    "CHIJ (TOA PAYOH)": "CHIJ PRIMARY (TOA PAYOH)",
    "CHIJ (KATONG)": "CHIJ (KATONG) PRIMARY",
    "CHIJ (KELLOCK)": "CHIJ (KELLOCK)",
}

HIGH_DEMAND_PRIMARY_GEOCODE_ALIASES: dict[str, list[str]] = {
    "CHIJ St. Nicholas Girls'": ["CHIJ St. Nicholas Girls' School"],
    "Catholic High": ["Catholic High School (Primary)"],
    "Maris Stella High": ["Maris Stella High School (Primary)"],
    "St. Andrew's Junior": ["St Andrew's School (Junior)"],
    "St. Gabriel's": ["St. Gabriel's Primary School"],
}

_MIN_ONEMAP_INTERVAL = 1.0 / ONEMAP_RATE_LIMIT_RPS

_SVY21_A = 6378137.0
_SVY21_F = 1 / 298.257223563
_SVY21_O_LAT = 1.366666
_SVY21_O_LON = 103.833333
_SVY21_NO = 38744.572
_SVY21_EO = 28001.642
_SVY21_K = 1.0
_SVY21_B = _SVY21_A * (1 - _SVY21_F)
_SVY21_E2 = (2 * _SVY21_F) - (_SVY21_F ** 2)
_SVY21_E4 = _SVY21_E2 ** 2
_SVY21_E6 = _SVY21_E4 * _SVY21_E2
_SVY21_A0 = 1 - (_SVY21_E2 / 4) - (3 * _SVY21_E4 / 64) - (5 * _SVY21_E6 / 256)
_SVY21_A2 = (3 / 8) * (_SVY21_E2 + (_SVY21_E4 / 4) + (15 * _SVY21_E6 / 128))
_SVY21_A4 = (15 / 256) * (_SVY21_E4 + (3 * _SVY21_E6 / 4))
_SVY21_A6 = 35 * _SVY21_E6 / 3072


def _ensure_output_dir() -> Path:
    """Create the output directory if needed and return it as a Path."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_json(records: list[dict[str, float | str]], filename: str) -> Path:
    """Write a list of reference records to JSON in OUTPUT_DIR."""
    output_path = _ensure_output_dir() / filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=True)
    return output_path


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> requests.Response:
    """Issue a GET request with retries on transient request failures and 429s."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"    Rate limited (429) for {url}. Waiting {wait:.1f}s "
                    f"(retry {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError):
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise RuntimeError(
                        f"Non-retryable HTTP error for {url}: {status_code}"
                    ) from exc
            if attempt == MAX_RETRIES - 1:
                break
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"    Request failed for {url}: {exc}. Waiting {wait:.1f}s "
                f"(retry {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed for {url}") from last_error


def _extract_float(record: dict[str, Any], *keys: str) -> float:
    """Return the first parseable float found under any of the provided keys."""
    for key in keys:
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    raise KeyError(f"Missing numeric field in record. Tried keys: {keys}")


def _extract_text(record: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string found under any of the provided keys."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Missing text field in record. Tried keys: {keys}")


def _download_data_gov_csv(dataset_id: str, session: requests.Session) -> pd.DataFrame:
    """Download a full data.gov.sg dataset as a DataFrame via async export."""
    init_url = f"{DATA_GOV_API_BASE}/{dataset_id}/initiate-download"
    poll_url = f"{DATA_GOV_API_BASE}/{dataset_id}/poll-download"

    print(f"  Requesting data.gov.sg export for {dataset_id} ...")
    init_response = _request_with_retry(session, init_url)
    download_url = init_response.json().get("data", {}).get("url")

    if not download_url:
        print("    Export not ready. Polling until download URL is available ...")
        for attempt in range(30):
            time.sleep(3)
            poll_response = _request_with_retry(session, poll_url)
            poll_data = poll_response.json().get("data", {})
            if poll_data.get("status") == "DOWNLOAD_SUCCESS":
                download_url = poll_data.get("url")
                print(f"    Export ready after {attempt + 1} poll(s).")
                break
            print(
                f"    Poll attempt {attempt + 1}/30 - "
                f"status: {poll_data.get('status', 'UNKNOWN')}"
            )

    if not download_url:
        raise RuntimeError(
            f"Unable to obtain download URL for data.gov.sg dataset {dataset_id}."
        )

    csv_response = _request_with_retry(session, download_url, timeout=300)
    dataframe = pd.read_csv(io.StringIO(csv_response.text))
    print(f"    Downloaded {len(dataframe):,} rows from data.gov.sg.")
    return dataframe


def _geocode_query(session: requests.Session, query: str) -> tuple[float, float] | None:
    """Resolve a query string to latitude and longitude using OneMap."""
    params: dict[str, Any] = {
        "searchVal": query,
        "returnGeom": "Y",
        "getAddrDetails": "N",
        "pageNum": 1,
    }

    time.sleep(_MIN_ONEMAP_INTERVAL)
    response = _request_with_retry(session, ONEMAP_API_URL, params=params, timeout=30)
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return None

    result = results[0]
    try:
        lat = float(result["LATITUDE"])
        lng = float(result["LONGITUDE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Unable to parse OneMap result for query: {query}") from exc
    return lat, lng


def _svy21_meridional_arc(latitude_radians: float) -> float:
    """Compute the meridional arc used by the SVY21 inverse projection."""
    return _SVY21_A * (
        (_SVY21_A0 * latitude_radians)
        - (_SVY21_A2 * math.sin(2 * latitude_radians))
        + (_SVY21_A4 * math.sin(4 * latitude_radians))
        - (_SVY21_A6 * math.sin(6 * latitude_radians))
    )


def _svy21_to_latlon(northing: float, easting: float) -> tuple[float, float]:
    """Convert SVY21 northing/easting coordinates to WGS84 latitude/longitude."""
    origin_lat = math.radians(_SVY21_O_LAT)
    origin_lon = math.radians(_SVY21_O_LON)
    mo = _svy21_meridional_arc(origin_lat)
    m_prime = mo + (northing - _SVY21_NO) / _SVY21_K

    sigma = m_prime / (_SVY21_A * _SVY21_A0)
    e1 = (1 - math.sqrt(1 - _SVY21_E2)) / (1 + math.sqrt(1 - _SVY21_E2))

    latitude_prime = (
        sigma
        + ((3 * e1 / 2) - (27 * e1 ** 3 / 32)) * math.sin(2 * sigma)
        + ((21 * e1 ** 2 / 16) - (55 * e1 ** 4 / 32)) * math.sin(4 * sigma)
        + (151 * e1 ** 3 / 96) * math.sin(6 * sigma)
        + (1097 * e1 ** 4 / 512) * math.sin(8 * sigma)
    )

    sin_lat_prime = math.sin(latitude_prime)
    cos_lat_prime = math.cos(latitude_prime)
    tan_lat_prime = math.tan(latitude_prime)

    rho_prime = (
        _SVY21_A * (1 - _SVY21_E2)
        / ((1 - _SVY21_E2 * sin_lat_prime ** 2) ** 1.5)
    )
    v_prime = _SVY21_A / math.sqrt(1 - _SVY21_E2 * sin_lat_prime ** 2)
    psi_prime = v_prime / rho_prime
    t_prime = tan_lat_prime
    x = (easting - _SVY21_EO) / (_SVY21_K * v_prime)

    latitude = latitude_prime - (
        (t_prime / (_SVY21_K * rho_prime))
        * (
            ((easting - _SVY21_EO) * x / 2)
            - (((easting - _SVY21_EO) * (x ** 3)) / 24)
            * (
                -4 * psi_prime ** 2
                + 9 * psi_prime * (1 - t_prime ** 2)
                + 12 * t_prime ** 2
            )
            + (((easting - _SVY21_EO) * (x ** 5)) / 720)
            * (
                8 * psi_prime ** 4 * (11 - 24 * t_prime ** 2)
                - 12 * psi_prime ** 3 * (21 - 71 * t_prime ** 2)
                + 15 * psi_prime ** 2 * (15 - 98 * t_prime ** 2 + 15 * t_prime ** 4)
                + 180 * psi_prime * (5 * t_prime ** 2 - 3 * t_prime ** 4)
                + 360 * t_prime ** 4
            )
            - (((easting - _SVY21_EO) * (x ** 7)) / 40320)
            * (
                1385
                + 3633 * t_prime ** 2
                + 4095 * t_prime ** 4
                + 1575 * t_prime ** 6
            )
        )
    )

    longitude = origin_lon + (
        x
        - ((x ** 3) / 6) * (psi_prime + 2 * t_prime ** 2)
        + ((x ** 5) / 120)
        * (
            -4 * psi_prime ** 3 * (1 - 6 * t_prime ** 2)
            + psi_prime ** 2 * (9 - 68 * t_prime ** 2)
            + 72 * psi_prime * t_prime ** 2
            + 24 * t_prime ** 4
        )
        - ((x ** 7) / 5040)
        * (
            61
            + 662 * t_prime ** 2
            + 1320 * t_prime ** 4
            + 720 * t_prime ** 6
        )
    ) / cos_lat_prime

    return math.degrees(latitude), math.degrees(longitude)


def _find_latest_train_station_zip(session: requests.Session) -> str:
    """Scrape the LTA static-data page and return the latest train station ZIP URL."""
    response = _request_with_retry(session, LTA_STATIC_DATA_URL, timeout=60)
    html = response.text

    absolute_matches = re.findall(
        r"https://datamall\.lta\.gov\.sg/content/dam/datamall/datasets/Geospatial/TrainStation_[A-Za-z]{3}\d{4}\.zip",
        html,
    )
    if absolute_matches:
        return absolute_matches[0]

    relative_matches = re.findall(
        r'/content/dam/datamall/datasets/Geospatial/TrainStation_[A-Za-z]{3}\d{4}\.zip',
        html,
    )
    if relative_matches:
        return f"https://datamall.lta.gov.sg{relative_matches[0]}"

    block_match = re.search(
        r"Train Station.*?Last Update:\s*([A-Za-z]{3})\s+(\d{4})",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if block_match:
        month_abbrev, year = block_match.groups()
        return (
            "https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/"
            f"TrainStation_{month_abbrev.title()}{year}.zip"
        )

    month_abbrevs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    current = datetime.now()
    for offset in range(12):
        zero_based_month = current.month - 1 - offset
        month_index = zero_based_month % 12
        year = current.year + (zero_based_month // 12)
        month_abbrev = month_abbrevs[month_index]
        candidate = (
            "https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/"
            f"TrainStation_{month_abbrev}{year}.zip"
        )
        try:
            probe = session.head(candidate, timeout=30, allow_redirects=True)
            if probe.ok:
                return candidate
        except requests.RequestException:
            continue

    raise RuntimeError(
        "Could not find a TrainStation ZIP link on the LTA static-data page."
    )


def _parse_dbf_records(dbf_bytes: bytes) -> list[dict[str, str]]:
    """Parse a DBF file into a list of dictionaries using the standard DBF layout."""
    num_records = struct.unpack("<I", dbf_bytes[4:8])[0]
    header_length = struct.unpack("<H", dbf_bytes[8:10])[0]
    record_length = struct.unpack("<H", dbf_bytes[10:12])[0]

    fields: list[tuple[str, int]] = []
    offset = 32
    while dbf_bytes[offset] != 0x0D:
        name = dbf_bytes[offset:offset + 11].split(b"\x00", 1)[0].decode("ascii").strip()
        field_length = dbf_bytes[offset + 16]
        fields.append((name, field_length))
        offset += 32

    records: list[dict[str, str]] = []
    for index in range(num_records):
        start = header_length + (index * record_length)
        record_bytes = dbf_bytes[start:start + record_length]
        if not record_bytes or record_bytes[0] == 0x2A:
            continue
        cursor = 1
        record: dict[str, str] = {}
        for field_name, field_length in fields:
            raw_value = record_bytes[cursor:cursor + field_length]
            record[field_name] = raw_value.decode("utf-8", errors="ignore").strip()
            cursor += field_length
        records.append(record)
    return records


def _parse_polygon_bbox_centers(shp_bytes: bytes) -> list[tuple[float, float]]:
    """Parse polygon shapefile records and return bbox center points in SVY21."""
    shape_type = struct.unpack("<I", shp_bytes[32:36])[0]
    if shape_type != 5:
        raise RuntimeError(f"Unexpected TrainStation shapefile type: {shape_type}. Expected polygon (5).")

    centers: list[tuple[float, float]] = []
    offset = 100
    total_length = len(shp_bytes)

    while offset + 8 <= total_length:
        record_number = struct.unpack(">I", shp_bytes[offset:offset + 4])[0]
        content_length_words = struct.unpack(">I", shp_bytes[offset + 4:offset + 8])[0]
        content_length_bytes = content_length_words * 2
        offset += 8
        content = shp_bytes[offset:offset + content_length_bytes]
        offset += content_length_bytes

        if not content:
            continue

        record_shape_type = struct.unpack("<I", content[:4])[0]
        if record_shape_type == 0:
            continue
        if record_shape_type != 5:
            raise RuntimeError(
                f"Unexpected TrainStation record shape type at record {record_number}: {record_shape_type}"
            )

        xmin, ymin, xmax, ymax = struct.unpack("<4d", content[4:36])
        centers.append(((ymin + ymax) / 2, (xmin + xmax) / 2))

    return centers


def _normalize_station_name(raw_name: str) -> str:
    """Clean LTA station labels into a shorter title-cased station name."""
    cleaned = raw_name.strip()
    for suffix in (" MRT STATION", " LRT STATION", " MRT", " LRT"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break
    return cleaned.title()


def _build_school_queries(school_name: str) -> list[str]:
    """Build OneMap fallback queries for school names with punctuation variants."""
    base = school_name.strip()
    without_parentheses = re.sub(r"\s*\([^)]*\)", "", base).strip()
    variants = [base]
    variants.extend(HIGH_DEMAND_PRIMARY_GEOCODE_ALIASES.get(base, []))
    variants.extend([
        base,
        without_parentheses,
        base.replace("ST.", "ST"),
        base.replace("ST. ", "SAINT "),
        base.replace("'", ""),
        base.replace(".", ""),
        base.replace("'", "").replace(".", ""),
        base.replace("ST. ", "SAINT ").replace("'", ""),
        base.replace("ST. ", "SAINT ").replace("'", "").replace(".", ""),
        without_parentheses.replace("ST.", "ST"),
        without_parentheses.replace("ST. ", "SAINT "),
        without_parentheses.replace("'", ""),
        without_parentheses.replace(".", ""),
        without_parentheses.replace("'", "").replace(".", ""),
        without_parentheses.replace("ST. ", "SAINT ").replace("'", ""),
        without_parentheses.replace("ST. ", "SAINT ").replace("'", "").replace(".", ""),
    ])

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = " ".join(variant.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return [f"{variant} SINGAPORE" for variant in deduped]


def _build_mall_queries(mall_name: str) -> list[str]:
    """Build OneMap fallback queries for malls with punctuation or alias issues."""
    base = mall_name.strip()
    variants = [base]
    variants.extend(MALL_QUERY_ALIASES.get(base, []))
    variants.extend([
        base.replace("@", " "),
        base.replace("@", " @ "),
        base.replace("Centre", "Center"),
        base.replace("Shopping Centre", "Shopping Center"),
    ])

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = " ".join(variant.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return [f"{variant} SINGAPORE" for variant in deduped]


def fetch_mrt_stations() -> list[dict[str, float | str]]:
    """
    Fetch all MRT/LRT stations from LTA DataMall and save them to JSON.

    Returns:
        A deduplicated list of station dicts with keys: name, lat, lng.
    """
    with requests.Session() as session:
        zip_url = _find_latest_train_station_zip(session)
        print(f"  MRT source ZIP: {zip_url}")
        response = _request_with_retry(session, zip_url, timeout=300)

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        dbf_name = next(name for name in archive.namelist() if name.endswith(".dbf"))
        shp_name = next(name for name in archive.namelist() if name.endswith(".shp"))
        dbf_records = _parse_dbf_records(archive.read(dbf_name))
        polygon_centers = _parse_polygon_bbox_centers(archive.read(shp_name))

    print(f"  MRT page 1: fetched {len(dbf_records)} rows from static geospatial dataset")

    if len(dbf_records) != len(polygon_centers):
        raise RuntimeError(
            "TrainStation shapefile record count does not match DBF record count."
        )

    stations_by_name: dict[str, dict[str, float | str]] = {}
    for record, (northing, easting) in zip(dbf_records, polygon_centers):
        name = _normalize_station_name(_extract_text(record, "STN_NAM_DE", "STN_NAM"))
        if name in stations_by_name:
            continue
        lat, lng = _svy21_to_latlon(northing, easting)
        stations_by_name[name] = {"name": name, "lat": lat, "lng": lng}

    stations = sorted(stations_by_name.values(), key=lambda item: str(item["name"]))
    path = _save_json(stations, "mrt_stations.json")
    print(f"  Saved {len(stations):,} MRT stations to {path}")
    return stations


def fetch_primary_schools() -> list[dict[str, float | str]]:
    """
    Fetch primary schools from data.gov.sg, geocode them via OneMap, and save JSON.

    Returns:
        A list of school dicts with keys: name, lat, lng.
    """
    with requests.Session() as session:
        dataframe = _download_data_gov_csv(PRIMARY_SCHOOL_DATASET_ID, session)

        normalized_columns = {column.lower(): column for column in dataframe.columns}
        level_column = normalized_columns.get("mainlevel_code")
        name_column = normalized_columns.get("school_name") or normalized_columns.get("name")

        if level_column is None or name_column is None:
            raise RuntimeError(
                "Unexpected school dataset schema. Expected columns including "
                "'mainlevel_code' and 'school_name' or 'name'."
            )

        filtered = dataframe[dataframe[level_column].astype(str).str.upper() == "PRIMARY"]
        school_names = sorted(
            {
                str(name).strip()
                for name in filtered[name_column].dropna().tolist()
                if str(name).strip()
            }
        )
        print(f"  Primary schools to geocode: {len(school_names):,}")

        schools: list[dict[str, float | str]] = []
        failures: list[str] = []

        for index, school_name in enumerate(school_names, start=1):
            print(f"    Geocoding school {index}/{len(school_names)}: {school_name}")
            result: tuple[float, float] | None = None
            for query in _build_school_queries(school_name):
                result = _geocode_query(session, query)
                if result is not None:
                    break
            if result is None:
                print(f"      No OneMap result for school: {school_name}")
                failures.append(school_name)
                continue

            lat, lng = result
            schools.append({"name": school_name, "lat": lat, "lng": lng})

        if failures:
            raise RuntimeError(
                f"Failed to geocode {len(failures)} primary school(s). "
                f"First few: {failures[:5]}"
            )

    path = _save_json(schools, "primary_schools.json")
    print(f"  Saved {len(schools):,} primary schools to {path}")
    return schools


def _normalize_school_name(value: str) -> str:
    text = html.unescape(value).upper().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_school_lookup_name(value: str) -> str:
    text = _normalize_school_name(value)
    text = text.replace("&", " AND ")
    text = text.replace("ST.", "SAINT ")
    text = re.sub(r"\bST\b", "SAINT", text)
    text = re.sub(r"[()'\.-]", " ", text)
    for token in ["PRIMARY", "SCHOOL", "JUNIOR", "HIGH"]:
        text = re.sub(rf"\b{token}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_high_demand_primary_school_names(top_n: int = HIGH_DEMAND_PRIMARY_TOP_N) -> list[str]:
    response = requests.get(PRIMARY_SCHOOL_RANKING_URL, timeout=60)
    response.raise_for_status()
    rows = re.findall(
        r'<tr><td[^>]*>(\d+)</td><td[^>]*><a [^>]+>([^<]+)</a>',
        response.text,
    )
    if len(rows) < top_n:
        raise RuntimeError(
            f"Expected at least {top_n} ranked schools from {PRIMARY_SCHOOL_RANKING_URL}, "
            f"found {len(rows)}."
        )
    return [html.unescape(name).strip() for _, name in rows[:top_n]]


def fetch_high_demand_primary_schools(
    primary_schools: list[dict[str, float | str]],
) -> list[dict[str, float | str | int]]:
    """
    Build a top-100 high-demand primary-school reference set.

    Coordinates are reused from the canonical primary school list when possible.
    Missing schools are geocoded directly from their ranked labels.
    """
    ranked_names = _extract_high_demand_primary_school_names()
    canonical_by_name = {
        _normalize_school_lookup_name(str(item["name"])): item
        for item in primary_schools
    }

    ranked_schools: list[dict[str, float | str | int]] = []
    failures: list[str] = []

    with requests.Session() as session:
        for rank, school_name in enumerate(ranked_names, start=1):
            normalized = _normalize_school_name(school_name)
            alias = HIGH_DEMAND_PRIMARY_NAME_ALIASES.get(normalized, school_name)
            canonical_match = canonical_by_name.get(
                _normalize_school_lookup_name(alias)
            )

            if canonical_match is not None:
                ranked_schools.append(
                    {
                        "rank": rank,
                        "name": school_name,
                        "canonical_name": canonical_match["name"],
                        "lat": canonical_match["lat"],
                        "lng": canonical_match["lng"],
                    }
                )
                continue

            print(f"    Geocoding high-demand school {rank}/{len(ranked_names)}: {school_name}")
            result: tuple[float, float] | None = None
            for query in _build_school_queries(school_name):
                result = _geocode_query(session, query)
                if result is not None:
                    break
            if result is None:
                failures.append(school_name)
                continue

            lat, lng = result
            ranked_schools.append(
                {
                    "rank": rank,
                    "name": school_name,
                    "canonical_name": school_name,
                    "lat": lat,
                    "lng": lng,
                }
            )

    if failures:
        raise RuntimeError(
            f"Failed to geocode {len(failures)} high-demand primary school(s). "
            f"First few: {failures[:5]}"
        )

    path = _save_json(ranked_schools, "high_demand_primary_schools.json")
    print(f"  Saved {len(ranked_schools):,} high-demand primary schools to {path}")
    return ranked_schools


def fetch_hawker_centres() -> list[dict[str, float | str]]:
    """
    Fetch hawker centres from data.gov.sg geojson and save them to JSON.

    Returns:
        A deduplicated list of hawker centre dicts with keys: name, lat, lng.
    """
    with requests.Session() as session:
        response = _request_with_retry(
            session,
            f"{DATA_GOV_API_BASE}/{HAWKER_CENTRE_DATASET_ID}/initiate-download",
            timeout=120,
        )
        payload = response.json()
        download_url = payload.get("data", {}).get("url")
        if not download_url:
            raise RuntimeError("Hawker centre dataset did not return a download URL.")

        geojson = _request_with_retry(session, download_url, timeout=300).json()

    features = geojson.get("features")
    if not isinstance(features, list) or not features:
        raise RuntimeError("Hawker centre geojson is empty or malformed.")

    hawkers_by_name: dict[str, dict[str, float | str]] = {}
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        properties = feature.get("properties") or {}
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue

        name = (
            str(properties.get("NAME") or properties.get("ADDRESSBUILDINGNAME") or "").strip()
        )
        if not name:
            continue

        try:
            lng = float(coordinates[0])
            lat = float(coordinates[1])
        except (TypeError, ValueError):
            continue

        hawkers_by_name.setdefault(name, {"name": name, "lat": lat, "lng": lng})

    hawkers = sorted(hawkers_by_name.values(), key=lambda item: str(item["name"]))
    path = _save_json(hawkers, "hawker_centres.json")
    print(f"  Saved {len(hawkers):,} hawker centres to {path}")
    return hawkers


def fetch_major_shopping_malls() -> list[dict[str, float | str]]:
    """
    Geocode the configured major shopping mall list via OneMap and save JSON.

    Returns:
        A list of major shopping mall dicts with keys: name, lat, lng.
    """
    mall_names = list(dict.fromkeys(SHOPPING_MALL_NAMES))
    malls: list[dict[str, float | str]] = []
    failures: list[str] = []

    with requests.Session() as session:
        for index, mall_name in enumerate(mall_names, start=1):
            print(f"    Geocoding mall {index}/{len(mall_names)}: {mall_name}")
            result: tuple[float, float] | None = None
            for query in _build_mall_queries(mall_name):
                result = _geocode_query(session, query)
                if result is not None:
                    break
            if result is None:
                print(f"      No OneMap result for mall: {mall_name}")
                failures.append(mall_name)
                continue

            lat, lng = result
            malls.append({"name": mall_name, "lat": lat, "lng": lng})

    if failures:
        print(
            f"  WARNING: Failed to geocode {len(failures)} major shopping mall(s). "
            f"Skipping them. First few: {failures[:5]}"
        )

    path = _save_json(malls, "major_shopping_malls.json")
    print(f"  Saved {len(malls):,} major shopping malls to {path}")
    return malls


def main() -> None:
    """Fetch all reference datasets and save them as JSON files."""
    print("=" * 60)
    print("HDB Resale - Reference Data Fetch")
    print("=" * 60)

    try:
        mrt_stations = fetch_mrt_stations()
        primary_schools = fetch_primary_schools()
        high_demand_primary_schools = fetch_high_demand_primary_schools(primary_schools)
        hawker_centres = fetch_hawker_centres()
        major_shopping_malls = fetch_major_shopping_malls()
    except Exception as exc:
        raise SystemExit(f"Reference data fetch failed: {exc}") from exc

    print("\nSummary")
    print(f"  MRT stations saved:     {len(mrt_stations):,}")
    print(f"  Primary schools saved:  {len(primary_schools):,}")
    print(f"  Top primary schools:    {len(high_demand_primary_schools):,}")
    print(f"  Hawker centres saved:   {len(hawker_centres):,}")
    print(f"  Major malls saved:      {len(major_shopping_malls):,}")
    print(f"  Output folder:          {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
