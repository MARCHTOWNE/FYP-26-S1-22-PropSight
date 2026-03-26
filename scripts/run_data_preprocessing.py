"""
run_data_preprocessing.py
=========================
Run only the data preparation steps:
  1. Fetch latest data from HDB API
  2. Clean and preprocess
  3. Geocode new addresses
  4. Compute proximity features
  5. Sync processed data to Supabase

Usage:
    python scripts/run_data_preprocessing.py
"""

from pipeline_orchestration import run_data_preprocessing_pipeline


if __name__ == "__main__":
    raise SystemExit(run_data_preprocessing_pipeline())
