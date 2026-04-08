"""
retrain_and_deploy.py
=====================
Run the full end-to-end workflow:
  1. Data preprocessing
  2. ML pipeline
  3. Supabase sync

Use this when you want a fresh model training run. For routine weekly
database refreshes from data.gov.sg, prefer scripts/run_data_preprocessing.py.

Focused scripts also exist:
  - python scripts/run_data_preprocessing.py
  - python scripts/run_ml_pipeline.py
  - python scripts/sync_to_supabase.py
  - python scripts/retrain_and_deploy.py
"""

from pipeline_orchestration import run_full_pipeline


if __name__ == "__main__":
    raise SystemExit(run_full_pipeline())
