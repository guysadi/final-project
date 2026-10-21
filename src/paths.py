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
