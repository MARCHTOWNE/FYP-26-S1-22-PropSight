"""
fetch_sora.py — MAS SORA data loader
=====================================
Provides load_sora_monthly() which returns a DataFrame with columns:
  year_month  (int, YYYYMM)
  sora_3m     (float, monthly average of daily 3-month compounded SORA)

Primary source: MAS API (paginated, 10-second timeout).
Fallback: ML/data/sora_monthly.csv
  - First 6 rows are metadata; skip them.
  - Header row is row 7 (index 6).
  - Date column: "SORA Publication Date" (format "03 Jan 2020").
  - Rate column: "Compound SORA - 3 month".
"""

import json
import os
from urllib import request as urllib_request

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SORA_CSV_PATH = os.path.join(BASE_DIR, "data", "sora_monthly.csv")

_MAS_SORA_URL = (
    "https://eservices.mas.gov.sg/api/action/datastore/search.json"
    "?resource_id=9a0bf149-308c-4bd2-832d-76c8e6cb47ed"
)
_USER_AGENT = "PropSight/1.0 (HDB Resale Price Prediction)"
_TIMEOUT = 10
_PAGE_LIMIT = 1000


def _fetch_sora_from_api() -> pd.DataFrame:
    """Fetch all daily SORA records from MAS API with pagination."""
    records = []
    offset = 0

    while True:
        url = f"{_MAS_SORA_URL}&limit={_PAGE_LIMIT}&offset={offset}"
        req = urllib_request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib_request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        page_records = data.get("result", {}).get("records", [])
        if not page_records:
            break

        records.extend(page_records)
        if len(page_records) < _PAGE_LIMIT:
            break
        offset += _PAGE_LIMIT

    if not records:
        raise ValueError("MAS API returned no SORA records.")

    df = pd.DataFrame(records)[["end_of_day", "comp_sora_3m"]]
    df["end_of_day"] = pd.to_datetime(df["end_of_day"], errors="coerce")
    df["comp_sora_3m"] = pd.to_numeric(df["comp_sora_3m"], errors="coerce")
    df = df.dropna(subset=["end_of_day", "comp_sora_3m"])
    return df


def _load_sora_from_csv() -> pd.DataFrame:
    """
    Load SORA fallback CSV.

    The file has 6 metadata rows before the actual header (row index 6).
    Relevant columns:
      "SORA Publication Date"  — date string like "03 Jan 2020"
      "Compound SORA - 3 month" — rate as float
    These are renamed to end_of_day and comp_sora_3m respectively.
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


def load_sora_monthly() -> pd.DataFrame:
    """
    Return monthly average 3-month compounded SORA.

    Tries MAS API first (10-second timeout, paginated).
    Falls back to ML/data/sora_monthly.csv on any failure.

    Returns DataFrame with columns:
      year_month  int   YYYYMM (e.g. 202401)
      sora_3m     float monthly average of daily comp_sora_3m
    """
    try:
        daily = _fetch_sora_from_api()
        print("  SORA: loaded from MAS API.")
    except Exception as exc:
        print(f"  SORA: API failed ({exc}); falling back to CSV.")
        try:
            daily = _load_sora_from_csv()
            print("  SORA: loaded from local CSV.")
        except Exception as csv_exc:
            print(f"  SORA: CSV also failed ({csv_exc}); returning empty DataFrame.")
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
