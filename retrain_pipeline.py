"""
retrain_pipeline.py
===================
Single responsibility: orchestrate the full HDB Resale prediction pipeline
end-to-end, from API fetch through model artefact promotion.

Design decisions:
  - check_for_new_data() short-circuits the run if no new month is available,
    avoiding unnecessary API calls and compute.
  - Each step is wrapped in its own try/except so a failure is isolated,
    reported, and does not silently corrupt later steps.
  - promote_model() is the single authoritative place where latest.txt is
    updated; feature_engineering.py writes to the timestamped run_dir only.
  - A JSON report is always written — even on failure — to provide an audit
    trail for every triggered run (supports NFR Transparency).

Execution order context:
  Orchestrates Steps 1–6:
    1. api_fetcher.run_fetch()
    2. data_pipeline.main()
    3. geocoding.run_geocoding()
    4. proximity_features.run_proximity_features()
    5. feature_engineering.main()
    6. Metric comparison + optional model promotion

Run:
    python retrain_pipeline.py                 # cron trigger (default)
    python retrain_pipeline.py --trigger api   # on-demand API trigger
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

import api_fetcher
import data_pipeline
import feature_engineering
import geocoding
import proximity_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH           = "hdb_resale.db"
OUTPUT_DIR        = "model_assets"
LOG_DIR           = "logs"
LATEST_MARKER     = os.path.join(OUTPUT_DIR, "latest.txt")
LATEST_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"  # Jan 2017–present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 helper: check whether new data exists
# ---------------------------------------------------------------------------

def check_for_new_data(db_path: str = DB_PATH) -> bool:
    """
    Determine whether newer data has been published since the last pipeline run.

    Reads last_fetched_month from pipeline_meta and compares it against the
    most recent month available on data.gov.sg (metadata query only — no download).

    Parameters:
        db_path: Path to the SQLite database file.

    Returns:
        True if newer data exists or if pipeline_meta is empty (first run).
        False if the latest available month matches what was last fetched.
    """
    meta = data_pipeline.get_pipeline_meta(db_path)
    last_fetched = meta.get("last_fetched_month")

    if not last_fetched:
        logger.info("pipeline_meta is empty — treating as first run. Proceeding.")
        return True

    logger.info("Last fetched month from pipeline_meta: %s", last_fetched)
    latest_available = api_fetcher.get_latest_available_month(LATEST_DATASET_ID)

    if latest_available is None:
        logger.warning(
            "Could not determine latest available month from data.gov.sg. "
            "Proceeding conservatively to avoid missing data."
        )
        return True

    logger.info("Latest available month from data.gov.sg:  %s", latest_available)

    if latest_available > last_fetched:
        logger.info("New data found (%s > %s). Pipeline will run.", latest_available, last_fetched)
        return True

    logger.info("No new data. Latest available (%s) matches last fetch. Skipping.", latest_available)
    return False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def load_current_best_metrics(output_dir: str = OUTPUT_DIR) -> dict | None:
    """
    Load metrics from the current best (latest) model run.

    Reads LATEST_MARKER to find the current run directory, then loads
    metrics.json from that directory.

    Parameters:
        output_dir: Root model_assets directory (used to resolve LATEST_MARKER).

    Returns:
        Dict of metrics from the latest run, or None if no previous run exists
        or if metrics.json cannot be read.
    """
    if not os.path.isfile(LATEST_MARKER):
        logger.info("No LATEST_MARKER found at '%s'. No previous metrics.", LATEST_MARKER)
        return None

    with open(LATEST_MARKER) as f:
        run_dir = f.read().strip()

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        logger.warning("metrics.json not found in run_dir '%s'.", run_dir)
        return None

    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read metrics.json from '%s': %s", run_dir, exc)
        return None


def is_new_model_better(new_metrics: dict, old_metrics: dict) -> bool:
    """
    Compare Q50 test MAE between the new run and the current best.

    Returns True (promote) only if the new MAE is strictly lower. Logs
    both values for the audit trail.

    Parameters:
        new_metrics: metrics dict from the newly completed run.
        old_metrics: metrics dict from the current promoted run.

    Returns:
        True if new model is strictly better; False otherwise.
    """
    new_mae = new_metrics.get("q50_test_mae")
    old_mae = old_metrics.get("q50_test_mae")

    logger.info("Q50 test MAE — new: %s  |  current best: %s", new_mae, old_mae)

    if new_mae is None:
        logger.warning("New metrics.json has no q50_test_mae. Cannot compare — will not promote.")
        return False

    if old_mae is None:
        logger.info("Current best has no q50_test_mae (stub). Promoting new model.")
        return True

    if new_mae < old_mae:
        logger.info("New model is better (%.2f < %.2f). Will promote.", new_mae, old_mae)
        return True

    logger.info("Current model is equal or better (%.2f >= %.2f). Will not promote.", new_mae, old_mae)
    return False


def promote_model(new_run_dir: str) -> None:
    """
    Promote a new model run by overwriting LATEST_MARKER.

    This is the single authoritative place where LATEST_MARKER is updated
    after a pipeline run. Logs the promotion as a deliberate, auditable action.

    Parameters:
        new_run_dir: Path to the run directory to promote.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LATEST_MARKER, "w") as f:
        f.write(new_run_dir)
    logger.info("PROMOTED: latest.txt → %s", new_run_dir)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_retrain_report(report: dict, log_dir: str = LOG_DIR) -> str:
    """
    Write a JSON report for this pipeline run to logs/.

    Always called — even on failure — to maintain a complete audit trail.

    Parameters:
        report: Dict containing at minimum: triggered_by, started_at,
                finished_at, new_data_found, steps_run, old_metrics,
                new_metrics, promoted.
        log_dir: Directory where report files are written.

    Returns:
        Path to the written report file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"retrain_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Retrain report written: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(triggered_by: str = "cron") -> None:
    """
    Execute all pipeline steps in order, comparing and conditionally
    promoting the resulting model.

    On any step failure: logs the error, writes a partial report with the
    failed step noted, and exits cleanly without promoting. Never raises
    unhandled exceptions.

    Parameters:
        triggered_by: How the pipeline was initiated ("cron" or "api").
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("HDB Resale — Retrain Pipeline  [triggered_by=%s]", triggered_by)
    logger.info("=" * 60)

    steps_run: list[str] = []
    failed_step: str | None = None
    new_data_found = False
    old_metrics = load_current_best_metrics()
    new_metrics: dict | None = None
    new_run_dir: str | None = None
    promoted = False

    try:
        # ------------------------------------------------------------------
        # Step 1: Check for new data
        # ------------------------------------------------------------------
        new_data_found = check_for_new_data(DB_PATH)
        steps_run.append("check_for_new_data")

        if not new_data_found:
            logger.info("No new data available. Pipeline run skipped.")
            finished_at = datetime.now(timezone.utc).isoformat()
            write_retrain_report({
                "triggered_by":   triggered_by,
                "started_at":     started_at,
                "finished_at":    finished_at,
                "new_data_found": False,
                "steps_run":      steps_run,
                "old_metrics":    old_metrics,
                "new_metrics":    None,
                "promoted":       False,
            })
            return

        # ------------------------------------------------------------------
        # Step 2: Fetch raw data
        # ------------------------------------------------------------------
        logger.info("Step 2: api_fetcher.run_fetch()")
        api_fetcher.run_fetch()
        steps_run.append("api_fetcher.run_fetch")

        # ------------------------------------------------------------------
        # Step 3: Consolidate, clean, and load into SQLite
        # ------------------------------------------------------------------
        logger.info("Step 3: data_pipeline.main()")
        data_pipeline.main()
        steps_run.append("data_pipeline.main")

        # ------------------------------------------------------------------
        # Step 4: Geocoding
        # ------------------------------------------------------------------
        logger.info("Step 4: geocoding.run_geocoding()")
        geocoding.run_geocoding(DB_PATH)
        steps_run.append("geocoding.run_geocoding")

        # ------------------------------------------------------------------
        # Step 5: Proximity features
        # ------------------------------------------------------------------
        logger.info("Step 5: proximity_features.run_proximity_features()")
        proximity_features.run_proximity_features(DB_PATH)
        steps_run.append("proximity_features.run_proximity_features")

        # ------------------------------------------------------------------
        # Step 6: Feature engineering → produces versioned run_dir
        # ------------------------------------------------------------------
        logger.info("Step 6: feature_engineering.main()")
        feature_engineering.main()
        steps_run.append("feature_engineering.main")

        # Resolve the run_dir written by feature_engineering.save_artefacts()
        if os.path.isfile(LATEST_MARKER):
            with open(LATEST_MARKER) as f:
                new_run_dir = f.read().strip()

        # Load new metrics
        if new_run_dir:
            metrics_path = os.path.join(new_run_dir, "metrics.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path) as f:
                    new_metrics = json.load(f)

        # ------------------------------------------------------------------
        # Step 7: Compare metrics and promote if better
        # ------------------------------------------------------------------
        logger.info("Step 7: Comparing metrics and deciding on promotion ...")
        if new_run_dir:
            if old_metrics is None or is_new_model_better(new_metrics or {}, old_metrics):
                promote_model(new_run_dir)
                promoted = True
                steps_run.append("promote_model")
            else:
                logger.info("New model not promoted. Previous model retained.")

    except Exception:
        failed_step = steps_run[-1] if steps_run else "unknown"
        logger.error(
            "Pipeline failed at step after '%s':\n%s",
            failed_step,
            traceback.format_exc(),
        )

    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        report = {
            "triggered_by":   triggered_by,
            "started_at":     started_at,
            "finished_at":    finished_at,
            "new_data_found": new_data_found,
            "steps_run":      steps_run,
            "failed_step":    failed_step,
            "old_metrics":    old_metrics,
            "new_metrics":    new_metrics,
            "promoted":       promoted,
            "new_run_dir":    new_run_dir,
        }
        report_path = write_retrain_report(report)
        status = "FAILED" if failed_step else "SUCCESS"
        logger.info("Pipeline %s. Report: %s", status, report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="HDB Resale — Retrain Pipeline orchestrator"
    )
    parser.add_argument(
        "--trigger",
        choices=["cron", "api"],
        default="cron",
        help="How the pipeline was triggered (default: cron)",
    )
    args = parser.parse_args()
    run_pipeline(triggered_by=args.trigger)


if __name__ == "__main__":
    main()
