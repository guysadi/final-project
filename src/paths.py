"""Every file location in the project, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"
LOGS_DIR = ROOT / "logs"

CREW1_DIR = ROOT / "artifacts" / "crew1"
CLEAN_DATA = CREW1_DIR / "clean_data.csv"
EDA_REPORT = CREW1_DIR / "eda_report.html"
INSIGHTS = CREW1_DIR / "insights.md"
CONTRACT = CREW1_DIR / "dataset_contract.json"
# Supporting files the agents read; not part of the contract.
CLEANING_SUMMARY = CREW1_DIR / "cleaning_summary.json"
EDA_STATS = CREW1_DIR / "eda_stats.json"

CREW1_ARTIFACTS = [CLEAN_DATA, EDA_REPORT, INSIGHTS, CONTRACT]

CREW2_DIR = ROOT / "artifacts" / "crew2"
FEATURES = CREW2_DIR / "features.csv"
MODEL = CREW2_DIR / "model.joblib"
EVAL_REPORT = CREW2_DIR / "evaluation_report.md"
MODEL_CARD = CREW2_DIR / "model_card.md"
# Supporting files the agents read; not part of the required outputs.
FEATURE_SPEC = CREW2_DIR / "feature_spec.json"
METRICS = CREW2_DIR / "metrics.json"

CREW2_ARTIFACTS = [FEATURES, MODEL, EVAL_REPORT, MODEL_CARD]
