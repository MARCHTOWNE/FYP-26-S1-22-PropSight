"""
fetch_sora.py — MAS SORA data loader
=====================================
Provides load_sora_monthly() which returns a DataFrame with columns:
  year_month  (int, YYYYMM)
  sora_3m     (float, monthly average of daily 3-month compounded SORA)

Primary source: ML/data/sora_monthly.csv (covers Jan 2020 – present).
  - First 6 rows are metadata; data starts at row 7.
  - Header row columns used: "SORA Publication Date", "Compound SORA - 3 month"

Fallback: MAS eservices API (chunked, retried).
  - Used only when the CSV is missing or empty.
  - The API is known to be intermittently unavailable (scheduled maintenance).
  - If both sources fail, returns an empty DataFrame; feature engineering
    handles this with median fill.
"""

import json
import os
import time
from datetime import date
from urllib import request as urllib_request

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SORA_CSV_PATH = os.path.join(BASE_DIR, "data", "sora_monthly.csv")

_MAS_SORA_BASE = (
    "https://eservices.mas.gov.sg/api/action/datastore/search.json"
    "?resource_id=9a0bf149-308c-4bd2-832d-76c8e6cb47ed"
)
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_TIMEOUT = 30
_CHUNK_YEARS = 2
_MAX_RETRIES = 3
_RETRY_SLEEP = 5
_START_YEAR = 2020  # SORA data begins Jan 2020


def _load_sora_from_csv() -> pd.DataFrame:
    """
    Load daily SORA data from the local CSV file.

    The file has 6 metadata rows before the actual header (row index 6).
    Columns used:
      "SORA Publication Date"  — date string like "03 Jan 2020"
      "Compound SORA - 3 month" — daily rate as float
    """
    df = pd.read_csv(SORA_CSV_PATH, skiprows=6, header=0)
    df = df.rename(columns={
        "SORA Publication Date": "end_of_day",
        "Compound SORA - 3 month": "comp_sora_3m",
    })
    df["end_of_day"] = pd.to_datetime(df["end_of_day"], errors="coerce")
    df["comp_sora_3m"] = pd.to_numeric(df["comp_sora_3m"], errors="coerce")
    df = df.dropna(subset=["end_of_day", "comp_sora_3m"])
    return df[["end_of_day", "comp_sora_3m"]]


def _fetch_sora_from_api() -> pd.DataFrame:
    """
    Fallback: fetch daily SORA records from MAS API in 2-year date range
    chunks with retry logic per chunk.

    Raises ValueError if any chunk fails all retries or API returns HTML
    (e.g. maintenance page).
    """
    today = date.today()
    current_year = today.year
    all_records: list[dict] = []

    for chunk_start in range(_START_YEAR, current_year + 1, _CHUNK_YEARS):
        chunk_end = min(chunk_start + _CHUNK_YEARS - 1, current_year)
        start_date = f"{chunk_start}-01-01"
        end_date = f"{chunk_end}-12-31"

        # Literal brackets are valid in query strings (RFC 3986 gen-delims)
        url = (
            f"{_MAS_SORA_BASE}"
            f"&fields=end_of_day,comp_sora_3m"
            f"&between[end_of_day]={start_date},{end_date}"
            f"&limit=10000"
        )

        chunk_records = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                req = urllib_request.Request(
                    url, headers={"User-Agent": _USER_AGENT}
                )
                with urllib_request.urlopen(req, timeout=_TIMEOUT) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    raw = resp.read()

                if "html" in content_type.lower():
                    raise ValueError(
                        f"API returned HTML instead of JSON "
                        f"(maintenance or redirect); Content-Type: {content_type}"
                    )

                data = json.loads(raw.decode("utf-8"))
                chunk_records = data.get("result", {}).get("records", [])
                break
            except Exception as exc:
                print(
                    f"  SORA API: chunk {start_date}–{end_date} "
                    f"attempt {attempt}/{_MAX_RETRIES} failed: {exc}"
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP)

        if chunk_records is None:
            raise ValueError(
                f"MAS API failed for chunk {start_date}–{end_date} "
                f"after {_MAX_RETRIES} attempts."
            )

        all_records.extend(chunk_records)

    if not all_records:
        raise ValueError("MAS API returned no SORA records.")

    df = pd.DataFrame(all_records)[["end_of_day", "comp_sora_3m"]]
    df["end_of_day"] = pd.to_datetime(df["end_of_day"], errors="coerce")
    df["comp_sora_3m"] = pd.to_numeric(df["comp_sora_3m"], errors="coerce")
    df = df.dropna(subset=["end_of_day", "comp_sora_3m"])
    return df


def load_sora_monthly() -> pd.DataFrame:
    """
    Return monthly average 3-month compounded SORA.

    Tries the local CSV first (fast, no network required, covers full SORA
    history Jan 2020–present). Falls back to MAS API if the CSV is missing
    or empty. Returns an empty DataFrame if both fail.

    Returns DataFrame with columns:
      year_month  int   YYYYMM (e.g. 202401)
      sora_3m     float monthly average of daily comp_sora_3m
    """
    daily: pd.DataFrame | None = None

    # --- Primary: local CSV ---
    if os.path.exists(SORA_CSV_PATH):
        try:
            daily = _load_sora_from_csv()
            if daily.empty:
                print("  SORA: CSV is empty, trying API fallback.")
                daily = None
            else:
                print(
                    f"  SORA: loaded {len(daily):,} daily rows from CSV "
                    f"({daily['end_of_day'].min().date()} – "
                    f"{daily['end_of_day'].max().date()})."
                )
        except Exception as exc:
            print(f"  SORA: CSV load failed ({exc}), trying API fallback.")
            daily = None

    # --- Fallback: MAS API ---
    if daily is None:
        try:
            daily = _fetch_sora_from_api()
            print(f"  SORA: loaded {len(daily):,} daily rows from MAS API.")
        except Exception as exc:
            print(
                f"  SORA: API also failed ({exc}); "
                "returning empty DataFrame (median fill will be applied)."
            )
            return pd.DataFrame(columns=["year_month", "sora_3m"])

    daily["year_month"] = (
        daily["end_of_day"].dt.year * 100 + daily["end_of_day"].dt.month
    ).astype(int)

    monthly = (
        daily.groupby("year_month")["comp_sora_3m"]
        .mean()
        .reset_index()
        .rename(columns={"comp_sora_3m": "sora_3m"})
    )
    return monthly
