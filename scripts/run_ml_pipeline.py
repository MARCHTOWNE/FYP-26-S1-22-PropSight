"""
run_ml_pipeline.py
==================
Run only the ML steps:
  1. Feature engineering from the canonical training source
  2. Model training
  3. Copy latest model artefacts into webapp/model_assets

Usage:
    python scripts/run_ml_pipeline.py
"""

from pipeline_orchestration import run_ml_pipeline


if __name__ == "__main__":
    raise SystemExit(run_ml_pipeline())
