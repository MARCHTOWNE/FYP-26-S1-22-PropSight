"""
sync_to_supabase.py
===================
Run only the Supabase sync step:
  1. Sync the latest processed data into Supabase
  2. Update model_versions with the latest ML run metrics

Usage:
    python scripts/sync_to_supabase.py
"""

from pipeline_orchestration import run_supabase_sync_pipeline


if __name__ == "__main__":
    raise SystemExit(run_supabase_sync_pipeline())
