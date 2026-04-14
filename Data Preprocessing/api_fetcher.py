"""
api_fetcher.py
==============
Single responsibility: fetch raw HDB resale datasets from data.gov.sg API
and save each as a CSV to the /raw directory.

Design decisions:
  - No data cleaning or transformation — raw API output only.
  - Async export + polling pattern required by data.gov.sg v1 API.
  - Historical split files are reused when already present locally.
  - The live Jan 2017-present dataset is refreshed when data.gov.sg reports
    a newer update date or the local cache reaches the weekly refresh window,
    unless HDB_FORCE_FETCH=1 is set.
  - If the live dataset refresh fails after retries, the latest cached live
    CSV is reused when available so the broader pipeline can keep running.
  - Exponential backoff is used for rate limits and transient request errors.
  - save_raw_csv() is idempotent — overwrites existing files.

Execution order context:
  Step 1 of 6. Outputs: raw/<dataset_id>.csv for each dataset.
  Next step: data_pipeline.py reads /raw and consolidates into SQLite.

Run:
    python api_fetcher.py
"""

import os
import tempfile
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_IDS = [
    "d_ebc5ab87086db484f88045b47411ebc5",  # 1990–1999
    "d_43f493c6c50d54243cc1eab0df142d6a",  # 2000–Feb 2012
    "d_2d5ff9ea31397b66239f245f57751537",  # Mar 2012–Dec 2014
    "d_ea9ed51da2787afaf8e51f827c304208",  # Jan 2015–Dec 2016
    "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",  # Jan 2017–present
]

OPEN_API_BASE    = "https://api-open.data.gov.sg/v1/public/api/datasets"
METADATA_API_BASE = "https://api-production.data.gov.sg/v2/public/api/datasets"
RAW_DIR          = os.environ.get("HDB_RAW_DIR", os.path.join(BASE_DIR, "raw"))
POLL_INTERVAL    = 3       # seconds between poll attempts
POLL_MAX_TRIES   = 30      # max polling attempts per dataset
REQUEST_PAUSE    = 10      # seconds to wait between datasets
MAX_RETRIES      = 7       # retries on HTTP 429
RETRY_BASE_DELAY = 5.0     # seconds; doubled each retry attempt
DOWNLOAD_TIMEOUT = (30, 600)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
PROGRESS_LOG_BYTES = 5 * 1024 * 1024
CURRENT_DATASET_ID = DATASET_IDS[-1]
DEFAULT_LIVE_REFRESH_DAYS = 7


class DataGovRateLimitError(RuntimeError):
    """Raised when data.gov.sg responds with HTTP 429."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer environment variable with a safe fallback."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        print(f"  WARNING: Ignoring invalid {name}={raw_value!r}; using {default}.")
        return default
    return max(value, minimum)


def _raw_csv_path(dataset_id: str, raw_dir: str | None = None) -> str:
    """Return the output CSV path for a dataset ID."""
    if raw_dir is None:
        raw_dir = RAW_DIR
    return os.path.join(raw_dir, f"{dataset_id}.csv")


def _format_bytes(num_bytes: int) -> str:
    """Render a byte count using a compact human-friendly unit."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _parse_iso_datetime(value: object) -> datetime | None:
    """Parse an ISO-like datetime or date string into an aware datetime."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    elif len(text) == 10 and text.count("-") == 2:
        text = f"{text}T00:00:00+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_date_label(value: datetime | None) -> str | None:
    """Render an aware datetime as YYYY-MM-DD for logs."""
    if value is None:
        return None
    return value.date().isoformat()


def _get_file_modified_at(file_path: str) -> datetime | None:
    """Return the last modified timestamp for a file as an aware UTC datetime."""
    if not os.path.exists(file_path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)


def _get_file_age_days(file_path: str) -> float | None:
    """Return the age of a file in days, or None when the file is missing."""
    modified_at = _get_file_modified_at(file_path)
    if modified_at is None:
        return None
    return (datetime.now(timezone.utc) - modified_at).total_seconds() / 86400


def _get_with_retry(
    session: requests.Session,
    url: str,
    timeout: int | tuple[int, int] = 60,
    *,
    stream: bool = False,
    retry_on_429: bool = True,
) -> requests.Response:
    """
    HTTP GET with exponential backoff on HTTP 429 and transient failures.

    Parameters:
        session: Reused requests session.
        url:     Full URL to GET.
        timeout: Request timeout in seconds.
        stream:  Whether to stream the response body.
        retry_on_429: Whether HTTP 429 should be retried with backoff.

    Returns:
        Successful requests.Response object.

    Raises:
        requests.HTTPError: On non-retryable HTTP errors.
        RuntimeError: On repeated transient request failures.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=timeout, stream=stream)
            if resp.status_code == 429:
                if not retry_on_429:
                    resp.close()
                    raise DataGovRateLimitError(f"Rate limited (429) for {url}")
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                resp.close()
                print(
                    f"    Rate limited (429). Waiting {wait:.0f}s ... "
                    f"(retry {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError):
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise
            if attempt == MAX_RETRIES - 1:
                break
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"    Request failed for {url}: {exc}. Waiting {wait:.0f}s ... "
                f"(retry {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed for {url}") from last_error


def _download_csv_as_dataframe(
    session: requests.Session,
    download_url: str,
    *,
    retry_on_429: bool = True,
) -> pd.DataFrame:
    """Download a CSV to a temporary file with progress logging, then parse it."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        response: requests.Response | None = None
        temp_path: str | None = None
        bytes_downloaded = 0

        try:
            response = _get_with_retry(
                session,
                download_url,
                timeout=DOWNLOAD_TIMEOUT,
                stream=True,
                retry_on_429=retry_on_429,
            )
            total_bytes = int(response.headers.get("Content-Length", "0") or 0)
            next_progress_mark = PROGRESS_LOG_BYTES

            with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as handle:
                temp_path = handle.name
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_downloaded += len(chunk)

                    if bytes_downloaded >= next_progress_mark:
                        if total_bytes:
                            pct = (bytes_downloaded / total_bytes) * 100
                            print(
                                f"    Downloaded {_format_bytes(bytes_downloaded)} / "
                                f"{_format_bytes(total_bytes)} ({pct:.0f}%)"
                            )
                        else:
                            print(f"    Downloaded {_format_bytes(bytes_downloaded)} ...")
                        next_progress_mark += PROGRESS_LOG_BYTES

            if total_bytes:
                print(
                    f"    Download complete: {_format_bytes(bytes_downloaded)} / "
                    f"{_format_bytes(total_bytes)}"
                )
            else:
                print(f"    Download complete: {_format_bytes(bytes_downloaded)}")

            df = pd.read_csv(temp_path)
            return df
        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                break
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"    CSV download failed: {exc}. Waiting {wait:.0f}s ... "
                f"(retry {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)
        finally:
            if response is not None:
                response.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    raise RuntimeError(f"Failed to download CSV from {download_url}") from last_error


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_dataset(
    dataset_id: str,
    session: requests.Session,
    *,
    retry_on_429: bool = True,
) -> pd.DataFrame:
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
    init_resp = _get_with_retry(
        session,
        f"{OPEN_API_BASE}/{dataset_id}/initiate-download",
        retry_on_429=retry_on_429,
    )
    download_url = init_resp.json().get("data", {}).get("url")

    if not download_url:
        print("    Export not ready — polling ...")
        for attempt in range(POLL_MAX_TRIES):
            time.sleep(POLL_INTERVAL)
            poll_resp = _get_with_retry(
                session,
                f"{OPEN_API_BASE}/{dataset_id}/poll-download",
                retry_on_429=retry_on_429,
            )
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

    df = _download_csv_as_dataframe(
        session,
        download_url,
        retry_on_429=retry_on_429,
    )
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


def raw_snapshot_is_complete(
    dataset_ids: list[str] = DATASET_IDS,
    raw_dir: str | None = None,
) -> bool:
    """Return True when every expected raw CSV already exists locally."""
    return all(os.path.exists(_raw_csv_path(dataset_id, raw_dir)) for dataset_id in dataset_ids)


def get_latest_local_raw_month(
    dataset_id: str = CURRENT_DATASET_ID,
    raw_dir: str | None = None,
) -> str | None:
    """
    Inspect the locally cached Jan 2017-present raw CSV and return its latest month.

    Returns None if the file is missing or the month column cannot be read.
    """
    file_path = _raw_csv_path(dataset_id, raw_dir)
    if not os.path.exists(file_path):
        return None

    try:
        month_series = pd.read_csv(
            file_path,
            usecols=["month"],
            dtype={"month": "string"},
        )["month"].dropna()
    except Exception as exc:
        print(f"  WARNING: Could not inspect cached dataset month for {dataset_id}: {exc}")
        return None

    if month_series.empty:
        return None
    return month_series.astype(str).str[:7].max()


def get_live_dataset_update_info(
    dataset_id: str = CURRENT_DATASET_ID,
) -> dict[str, str | datetime | None]:
    """
    Return the latest coverage month and update timestamp for the live dataset.

    The fetch policy now uses the exact update date for weekly refresh checks
    while keeping the YYYY-MM coverage marker available for context.
    """
    metadata_url = f"{METADATA_API_BASE}/{dataset_id}/metadata"
    try:
        with requests.Session() as session:
            resp = session.get(metadata_url, timeout=15)
            resp.raise_for_status()
        data = resp.json().get("data", {})
        coverage_end = data.get("coverageEnd")
        coverage_month = None
        if coverage_end and len(str(coverage_end)) >= 7:
            coverage_month = str(coverage_end)[:7]

        last_updated_at = _parse_iso_datetime(data.get("lastUpdatedAt"))
        fallback_date = (
            last_updated_at
            or _parse_iso_datetime(coverage_end)
            or _parse_iso_datetime(data.get("createdAt"))
        )

        return {
            "latest_available_month": coverage_month,
            "last_updated_at": last_updated_at,
            "latest_update_date": _format_date_label(fallback_date),
        }
    except Exception as exc:
        print(f"  [get_live_dataset_update_info] WARNING: metadata check failed: {exc}")
        return {
            "latest_available_month": None,
            "last_updated_at": None,
            "latest_update_date": None,
        }


def get_latest_available_month(dataset_id: str = DATASET_IDS[-1]) -> str | None:
    """Compatibility wrapper returning the latest available coverage month."""
    update_info = get_live_dataset_update_info(dataset_id)
    latest_available_month = update_info.get("latest_available_month")
    if isinstance(latest_available_month, str):
        return latest_available_month
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
    print(f"\n  Preparing {len(dataset_ids)} dataset(s) from data.gov.sg ...")
    saved_paths: list[str] = []
    downloaded_count = 0
    reused_count = 0
    force_refresh = _env_flag("HDB_FORCE_FETCH")
    live_refresh_days = _env_int(
        "HDB_LIVE_REFRESH_DAYS",
        DEFAULT_LIVE_REFRESH_DAYS,
    )
    latest_available_month: str | None = None
    latest_update_date: str | None = None
    latest_updated_at: datetime | None = None

    if force_refresh:
        print("  HDB_FORCE_FETCH=1 detected — refreshing every dataset.")
    elif CURRENT_DATASET_ID in dataset_ids:
        update_info = get_live_dataset_update_info(CURRENT_DATASET_ID)
        latest_available_month = update_info.get("latest_available_month")
        latest_update_date = update_info.get("latest_update_date")
        last_updated_value = update_info.get("last_updated_at")
        if isinstance(last_updated_value, datetime):
            latest_updated_at = last_updated_value

        if isinstance(latest_update_date, str) and latest_update_date:
            print(f"  Latest data.gov.sg update date: {latest_update_date}")
        elif isinstance(latest_available_month, str) and latest_available_month:
            print(f"  Latest data.gov.sg coverage month: {latest_available_month}")
        else:
            print("  Could not verify the latest data.gov.sg update date — cached live data will be reused when available.")
        print(f"  Live dataset refresh window: every {live_refresh_days} day(s)")

    with requests.Session() as session:
        for i, dataset_id in enumerate(dataset_ids):
            cached_path = _raw_csv_path(dataset_id)
            is_live_dataset = dataset_id == CURRENT_DATASET_ID

            if not force_refresh and os.path.exists(cached_path):
                if not is_live_dataset:
                    print(f"  Using cached historical dataset: {dataset_id}")
                    saved_paths.append(cached_path)
                    reused_count += 1
                    continue

                local_month = get_latest_local_raw_month(dataset_id)
                local_refreshed_at = _get_file_modified_at(cached_path)
                local_refresh_date = _format_date_label(local_refreshed_at) or "unknown"
                cache_age_days = _get_file_age_days(cached_path)
                refresh_reason: str | None = None

                if latest_available_month and local_month is None:
                    refresh_reason = (
                        f"local latest transaction month is unknown "
                        f"(remote coverage month {latest_available_month})"
                    )
                elif latest_available_month and local_month and local_month < latest_available_month:
                    refresh_reason = (
                        f"remote coverage month {latest_available_month} "
                        f"is newer than local {local_month}"
                    )
                elif latest_updated_at and local_refreshed_at is None:
                    refresh_reason = (
                        f"remote update date {_format_date_label(latest_updated_at) or 'unknown'} "
                        "is known but the local refresh date is unknown"
                    )
                elif (
                    latest_updated_at
                    and local_refreshed_at
                    and local_refreshed_at < latest_updated_at.astimezone(timezone.utc)
                ):
                    refresh_reason = (
                        f"remote update date {_format_date_label(latest_updated_at) or 'unknown'} "
                        f"is newer than local refresh {local_refresh_date}"
                    )
                elif cache_age_days is not None and cache_age_days >= live_refresh_days:
                    refresh_reason = (
                        f"cache age {cache_age_days:.1f} day(s) reached the "
                        f"{live_refresh_days}-day refresh window"
                    )

                if refresh_reason is None:
                    print(
                        f"  Using cached live dataset: {dataset_id} "
                        f"(latest txn month {local_month or 'unknown'}, "
                        f"last refreshed {local_refresh_date})"
                    )
                    saved_paths.append(cached_path)
                    reused_count += 1
                    continue

                print(f"  Refreshing live dataset: {dataset_id} ({refresh_reason})")

            try:
                df = fetch_dataset(
                    dataset_id,
                    session,
                    retry_on_429=not (is_live_dataset and not force_refresh and os.path.exists(cached_path)),
                )
            except DataGovRateLimitError as exc:
                if is_live_dataset and not force_refresh and os.path.exists(cached_path):
                    print(
                        f"  WARNING: Live dataset refresh was rate limited for {dataset_id}: {exc}"
                    )
                    print(
                        f"  Reusing cached live dataset instead: {cached_path}"
                    )
                    saved_paths.append(cached_path)
                    reused_count += 1
                    continue
                raise
            except Exception as exc:
                if is_live_dataset and not force_refresh and os.path.exists(cached_path):
                    print(
                        f"  WARNING: Live dataset refresh failed for {dataset_id}: {exc}"
                    )
                    print(
                        f"  Falling back to cached live dataset: {cached_path}"
                    )
                    saved_paths.append(cached_path)
                    reused_count += 1
                    continue
                raise

            path = save_raw_csv(df, dataset_id)
            saved_paths.append(path)
            downloaded_count += 1

            if i < len(dataset_ids) - 1:
                print(f"    Pausing {REQUEST_PAUSE}s before next dataset ...")
                time.sleep(REQUEST_PAUSE)

    print(
        f"\n  Fetch complete. Datasets ready: {len(saved_paths)}  "
        f"Downloaded: {downloaded_count}  Reused: {reused_count}"
    )
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
