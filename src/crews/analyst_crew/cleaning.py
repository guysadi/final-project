"""Deterministic ingestion and cleaning of the raw E-commerce churn dataset.

Every decision is in code, not in an LLM's head, so running it twice on the same
raw file always produces the same clean_data.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_SHEET = "E Comm"
DICT_SHEET = "Data Dict"
ID_COL = "CustomerID"
TARGET = "Churn"

# Same category spelled two ways in the raw data.
CATEGORY_FIXES = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}

# Columns that are counts / scores and should be whole numbers after imputation.
INTEGER_COLS = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=RAW_SHEET)
    return pd.read_csv(path)


def load_data_dictionary(path: str | Path) -> dict[str, str]:
    """Column descriptions shipped with the dataset (the 'Data Dict' sheet)."""
    d = pd.read_excel(path, sheet_name=DICT_SHEET, header=None).iloc[2:, 2:4].dropna()
    return dict(zip(d[2].str.strip(), d[3].str.strip()))


def profile(df: pd.DataFrame) -> dict:
    """Quick facts about a dataframe, used before and after cleaning."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows_ignoring_id": int(df.drop(columns=[ID_COL]).duplicated().sum())
        if ID_COL in df.columns else int(df.duplicated().sum()),
        "categorical_values": {
            c: sorted(df[c].dropna().unique().tolist())
            for c in df.select_dtypes(include=["object", "string"]).columns
        },
    }


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the cleaned dataframe and a human-readable log of every change made."""
    df = raw.copy()
    log: list[str] = []

    # 1. Duplicates: same customer record under a different CustomerID.
    dupes = df.drop(columns=[ID_COL]).duplicated()
    if dupes.any():
        df = df[~dupes].reset_index(drop=True)
        log.append(f"Dropped {int(dupes.sum())} duplicate rows (identical in every column except {ID_COL}).")

    # 2. Merge category spellings.
    for col, mapping in CATEGORY_FIXES.items():
        changed = int(df[col].isin(mapping.keys()).sum())
        df[col] = df[col].replace(mapping)
        if changed:
            pairs = ", ".join(f"'{a}'→'{b}'" for a, b in mapping.items())
            log.append(f"{col}: merged {changed} values ({pairs}).")

    # 3. Impute missing numeric values with the median.
    for col in df.columns[df.isna().any()]:
        n = int(df[col].isna().sum())
        median = df[col].median()
        df[col] = df[col].fillna(median)
        log.append(f"{col}: filled {n} missing values with the median ({median:g}).")

    # 4. Whole-number columns back to int.
    for col in INTEGER_COLS:
        df[col] = df[col].round().astype("int64")
    log.append(f"Cast {len(INTEGER_COLS)} count columns to integer.")

    # 5. Deterministic row order.
    df = df.sort_values(ID_COL).reset_index(drop=True)
    return df, log


def run_cleaning(raw_path: str | Path, out_path: str | Path) -> dict:
    raw = load_raw(raw_path)
    before = profile(raw)
    df, log = clean(raw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {"before": before, "after": profile(df), "cleaning_log": log, "output": str(out_path)}
