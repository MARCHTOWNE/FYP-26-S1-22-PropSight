"""
api_fetcher.py
==============
Single responsibility: fetch raw HDB resale datasets from data.gov.sg API
and save each as a CSV to the /raw directory.

Design decisions:
  - No data cleaning or transformation — raw API output only.
  - Async export + polling pattern required by data.gov.sg v1 API.
  - Exponential backoff on HTTP 429 to respect rate limits.
  - save_raw_csv() is idempotent — overwrites existing files.

Execution order context:
  Step 1 of 6. Outputs: raw/<dataset_id>.csv for each dataset.
  Next step: data_pipeline.py reads /raw and consolidates into SQLite.

Run:
    python api_fetcher.py
"""

import io
import os
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_IDS = [
    "d_ebc5ab87086db484f88045b47411ebc5",  # 1990–1999
    "d_43f493c6c50d54243cc1eab0df142d6a",  # 2000–Feb 2012
    "d_2d5ff9ea31397b66239f245f57751537",  # Mar 2012–Dec 2014
    "d_ea9ed51da2787afaf8e51f827c304208",  # Jan 2015–Dec 2016
    "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",  # Jan 2017–present
]

OPEN_API_BASE    = "https://api-open.data.gov.sg/v1/public/api/datasets"
RAW_DIR          = "raw"
POLL_INTERVAL    = 3       # seconds between poll attempts
POLL_MAX_TRIES   = 30      # max polling attempts per dataset
REQUEST_PAUSE    = 3       # seconds to wait between datasets
MAX_RETRIES      = 5       # retries on HTTP 429
RETRY_BASE_DELAY = 5.0     # seconds; doubled each retry attempt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, timeout: int = 60) -> requests.Response:
    """
    HTTP GET with exponential backoff on HTTP 429 Too Many Requests.

    Parameters:
        url:     Full URL to GET.
        timeout: Request timeout in seconds.

    Returns:
        Successful requests.Response object.

    Raises:
        requests.HTTPError: On non-429 errors, or on 429 after MAX_RETRIES exhausted.
    """
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 429:
            wait = RETRY_BASE_DELAY * (2 ** attempt)  # 5, 10, 20, 40, 80 s
            print(f"    Rate limited (429). Waiting {wait:.0f}s ... (retry {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp

    # Final attempt — raise naturally on any error
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Download one dataset from the data.gov.sg v1 API.

    Initiates an asynchronous export, polls until DOWNLOAD_SUCCESS, then
    streams and parses the resulting CSV. No transformations are applied.

    Parameters:
        dataset_id: Data.gov.sg dataset identifier string.

    Returns:
        Raw DataFrame exactly as returned by the API.

    Raises:
        TimeoutError: If export does not complete within POLL_MAX_TRIES attempts.
        requests.HTTPError: On non-recoverable HTTP errors.
    """
    print(f"  Fetching: {dataset_id}")
    init_resp = _get_with_retry(f"{OPEN_API_BASE}/{dataset_id}/initiate-download")
    download_url = init_resp.json().get("data", {}).get("url")

    if not download_url:
        print("    Export not ready — polling ...")
        for attempt in range(POLL_MAX_TRIES):
            time.sleep(POLL_INTERVAL)
            poll_resp = _get_with_retry(f"{OPEN_API_BASE}/{dataset_id}/poll-download")
            poll_data = poll_resp.json().get("data", {})
            if poll_data.get("status") == "DOWNLOAD_SUCCESS":
                download_url = poll_data["url"]
                print(f"    Export ready after {attempt + 1} poll(s).")
                break
            print(f"    Poll attempt {attempt + 1}/{POLL_MAX_TRIES} — status: {poll_data.get('status')}")
        else:
            raise TimeoutError(
                f"Export for {dataset_id} did not complete after {POLL_MAX_TRIES} attempts."
            )

    csv_resp = requests.get(download_url, timeout=300)
    csv_resp.raise_for_status()
    df = pd.read_csv(io.StringIO(csv_resp.text))
    print(f"    Rows: {len(df):,}  Columns: {list(df.columns)}")
    return df


def save_raw_csv(df: pd.DataFrame, dataset_id: str) -> str:
    """
    Save a raw DataFrame to RAW_DIR/<dataset_id>.csv.

    Creates RAW_DIR if it does not already exist. Overwrites any
    existing file with the same dataset_id (idempotent).

    Parameters:
        df:         DataFrame to save.
        dataset_id: Used as the output filename stem.

    Returns:
        Absolute path to the saved CSV file.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DIR, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)
    size_kb = os.path.getsize(file_path) / 1024
    print(f"    Saved → {file_path}  ({size_kb:,.1f} KB, {len(df):,} rows)")
    return file_path


def get_latest_available_month(dataset_id: str = DATASET_IDS[-1]) -> str | None:
    """
    Query the metadata endpoint for the most recent dataset to determine
    the latest available month without downloading any data.

    Used by retrain_pipeline.py to decide whether new data has arrived
    since the last pipeline run.

    Parameters:
        dataset_id: Dataset to query; defaults to the Jan 2017–present dataset.

    Returns:
        Latest month as "YYYY-MM" string, or None on any failure.
    """
    try:
        resp = _get_with_retry(f"{OPEN_API_BASE}/{dataset_id}/metadata")
        data = resp.json().get("data", {})
        # The metadata response includes a 'coverageEnd' or similar field;
        # fall back to checking available columns in dataset info.
        # Attempt common field names used by data.gov.sg metadata API.
        for field in ("coverageEnd", "coverage_end", "lastUpdated", "last_updated"):
            value = data.get(field)
            if value and len(str(value)) >= 7:
                # Normalise to YYYY-MM (trim day portion if present)
                return str(value)[:7]
        print(f"  [get_latest_available_month] Unrecognised metadata fields: {list(data.keys())}")
        return None
    except Exception as exc:
        print(f"  [get_latest_available_month] WARNING: metadata check failed: {exc}")
        return None


def run_fetch(dataset_ids: list[str] = DATASET_IDS) -> list[str]:
    """
    Main entry point: fetch and save all specified datasets.

    Called by retrain_pipeline.py as Step 1 of the pipeline.

    Parameters:
        dataset_ids: List of data.gov.sg dataset IDs to fetch. Defaults
                     to the full chronological set (1990–present).

    Returns:
        List of file paths for every saved CSV, in the same order as dataset_ids.
    """
    print(f"\n  Fetching {len(dataset_ids)} dataset(s) from data.gov.sg ...")
    saved_paths: list[str] = []
    total_rows = 0

    for i, dataset_id in enumerate(dataset_ids):
        df = fetch_dataset(dataset_id)
        path = save_raw_csv(df, dataset_id)
        saved_paths.append(path)
        total_rows += len(df)

        if i < len(dataset_ids) - 1:
            print(f"    Pausing {REQUEST_PAUSE}s before next dataset ...")
            time.sleep(REQUEST_PAUSE)

    print(f"\n  Fetch complete. Datasets: {len(saved_paths)}  Total rows: {total_rows:,}")
    return saved_paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch all datasets and save raw CSVs to /raw."""
    print("=" * 60)
    print("HDB Resale — API Fetcher")
    print("=" * 60)
    paths = run_fetch()
    print(f"\nDone. {len(paths)} file(s) saved to: {RAW_DIR}/")


if __name__ == "__main__":
    main()
