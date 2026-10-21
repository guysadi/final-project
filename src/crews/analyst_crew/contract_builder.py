"""Builds dataset_contract.json from the clean data.

The schema part (columns, dtypes, ranges, allowed values) is derived in code so it
is exact and reproducible. The human part (descriptions, units, assumptions) comes
from the dataset's data dictionary and the Contract Author agent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.validation.contract import validate_contract_structure, validate_data_against_contract

ID_COL = "CustomerID"
TARGET = "Churn"
CATEGORICAL_INT_MAX_LEVELS = 6
# Upper bounds get 50% headroom over the observed max: room for normal growth,
# but a ×100 unit change (dollars→cents) still fails loudly.
MAX_HEADROOM = 1.5
CONTRACT_VERSION = "1.0"


def _numeric_spec(series: pd.Series) -> dict:
    is_int = pd.api.types.is_integer_dtype(series)
    spec: dict = {"dtype": "integer" if is_int else "number", "nullable": bool(series.isna().any())}
    if is_int and series.nunique() <= CATEGORICAL_INT_MAX_LEVELS:
        spec["allowed_values"] = sorted(int(v) for v in series.dropna().unique())
        return spec
    lo, hi = series.min(), series.max()
    spec["min"] = 0 if lo >= 0 else float(lo)
    upper = hi * MAX_HEADROOM
    spec["max"] = int(math.ceil(upper)) if is_int else round(float(upper), 2)
    spec["observed_range"] = [lo.item(), hi.item()]
    return spec


def derive_schema(df: pd.DataFrame) -> dict[str, dict]:
    columns = {}
    for col in df.columns:
        s = df[col]
        if col == ID_COL:
            columns[col] = {"dtype": "integer", "nullable": False}
        elif pd.api.types.is_numeric_dtype(s):
            columns[col] = _numeric_spec(s)
        else:
            columns[col] = {"dtype": "string", "nullable": bool(s.isna().any()),
                            "allowed_values": sorted(s.dropna().unique().tolist())}
    return columns


def build_contract(df: pd.DataFrame, descriptions: dict[str, str] | None = None,
                   units: dict[str, str] | None = None,
                   assumptions: list[str] | None = None) -> dict:
    descriptions, units = descriptions or {}, units or {}
    columns = derive_schema(df)
    for col, spec in columns.items():
        if col in descriptions:
            spec["description"] = descriptions[col]
        if col in units:
            spec["unit"] = units[col]
    return {
        "dataset": "clean_data.csv",
        "version": CONTRACT_VERSION,
        "primary_key": ID_COL,
        "target": TARGET,
        "target_description": "1 = customer churned, 0 = customer stayed",
        # Allow the row count to shrink by 20% on a refresh before the gate objects.
        "min_rows": int(len(df) * 0.8),
        "columns": columns,
        "assumptions": assumptions or [],
    }


def write_contract(clean_path: str | Path, contract_path: str | Path,
                   descriptions: dict[str, str] | None = None,
                   units: dict[str, str] | None = None,
                   assumptions: list[str] | None = None) -> dict:
    """Build the contract, prove it holds on the data it describes, then save it."""
    df = pd.read_csv(clean_path)
    contract = build_contract(df, descriptions, units, assumptions)
    for report in (validate_contract_structure(contract),
                   validate_data_against_contract(df, contract)):
        report.raise_if_failed()
    Path(contract_path).write_text(json.dumps(contract, indent=2))
    return contract
