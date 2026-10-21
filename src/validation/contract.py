"""Validation gate between the Data Analyst crew and the Data Scientist crew.

The dataset contract (artifacts/crew1/dataset_contract.json) is the only thing the
modeling crew may assume about the clean data. Everything here is plain,
deterministic pandas code — no LLM decides whether a handoff is valid.

Contract format:

{
  "dataset": "clean_data.csv",
  "version": "1.0",
  "primary_key": "CustomerID",
  "target": "Churn",
  "min_rows": 1000,
  "columns": {
    "CashbackAmount": {"dtype": "number", "nullable": false,
                       "min": 0, "max": 400, "unit": "USD",
                       "description": "Average cashback last month"},
    "Gender": {"dtype": "string", "nullable": false,
               "allowed_values": ["Female", "Male"]}
  },
  "assumptions": ["One row per customer", "..."]
}

Supported dtypes: "integer", "number", "string".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

VALID_DTYPES = {"integer", "number", "string"}
REQUIRED_CONTRACT_KEYS = {"dataset", "version", "target", "columns"}


class ContractViolation(Exception):
    """Raised when data does not honour its contract. The message is meant for humans."""

    def __init__(self, stage: str, errors: list[str]):
        self.stage = stage
        self.errors = errors
        lines = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"{stage} failed with {len(errors)} problem(s):\n{lines}")


@dataclass
class ValidationReport:
    stage: str
    errors: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ContractViolation(self.stage, self.errors)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "passed": self.passed,
                "checks_run": self.checks_run, "errors": self.errors}


def load_contract(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ContractViolation("Contract loading", [f"Contract file not found: {path}"])
    try:
        contract = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ContractViolation("Contract loading", [f"Contract is not valid JSON: {e}"])
    report = validate_contract_structure(contract)
    report.raise_if_failed()
    return contract


def validate_contract_structure(contract: dict) -> ValidationReport:
    """Check the contract itself is well-formed before using it to judge data."""
    report = ValidationReport("Contract structure check")
    missing = REQUIRED_CONTRACT_KEYS - contract.keys()
    report.checks_run += 1
    if missing:
        report.errors.append(f"Contract is missing required keys: {sorted(missing)}")
        return report

    columns = contract["columns"]
    report.checks_run += 1
    if not isinstance(columns, dict) or not columns:
        report.errors.append("Contract 'columns' must be a non-empty object")
        return report

    for key in ("target", "primary_key"):
        name = contract.get(key)
        report.checks_run += 1
        if name is not None and name not in columns:
            report.errors.append(f"Contract {key} '{name}' is not described in 'columns'")

    for name, spec in columns.items():
        report.checks_run += 1
        dtype = spec.get("dtype")
        if dtype not in VALID_DTYPES:
            report.errors.append(
                f"Column '{name}': dtype '{dtype}' is not one of {sorted(VALID_DTYPES)}")
        if "min" in spec and "max" in spec and spec["min"] > spec["max"]:
            report.errors.append(f"Column '{name}': min {spec['min']} > max {spec['max']}")
    return report


def _dtype_error(name: str, series: pd.Series, dtype: str) -> str | None:
    values = series.dropna()
    if dtype == "string":
        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            return f"Column '{name}': expected string, got {series.dtype}"
        return None
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return f"Column '{name}': expected {dtype}, got {series.dtype}"
    if dtype == "integer" and not (values == values.round()).all():
        return f"Column '{name}': expected integer, found non-whole numbers"
    return None


def validate_data_against_contract(df: pd.DataFrame, contract: dict) -> ValidationReport:
    """The gate: does the clean data look exactly like the contract says it does?"""
    report = ValidationReport("Contract vs. clean data check")
    columns = contract["columns"]

    expected, actual = set(columns), set(df.columns)
    report.checks_run += 1
    if missing := sorted(expected - actual):
        report.errors.append(f"Columns in contract but missing from data: {missing}")
    report.checks_run += 1
    if extra := sorted(actual - expected):
        report.errors.append(
            f"Columns in data but not in contract (the modeling crew may not assume them): {extra}")

    min_rows = contract.get("min_rows")
    report.checks_run += 1
    if min_rows is not None and len(df) < min_rows:
        report.errors.append(f"Data has {len(df)} rows, contract requires at least {min_rows}")

    pk = contract.get("primary_key")
    if pk and pk in df.columns:
        report.checks_run += 1
        if dupes := int(df[pk].duplicated().sum()):
            report.errors.append(f"Primary key '{pk}' has {dupes} duplicate value(s)")

    for name, spec in columns.items():
        if name not in df.columns:
            continue
        series = df[name]

        report.checks_run += 1
        if err := _dtype_error(name, series, spec["dtype"]):
            report.errors.append(err)
            continue

        report.checks_run += 1
        nulls = int(series.isna().sum())
        if not spec.get("nullable", False) and nulls:
            report.errors.append(f"Column '{name}': {nulls} null value(s) but contract says not nullable")

        values = series.dropna()
        if values.empty:
            continue
        unit = f" {spec['unit']}" if spec.get("unit") else ""
        if "min" in spec:
            report.checks_run += 1
            if (below := values < spec["min"]).any():
                report.errors.append(
                    f"Column '{name}': {int(below.sum())} value(s) below contract min "
                    f"{spec['min']}{unit} (lowest: {values.min()})")
        if "max" in spec:
            report.checks_run += 1
            if (above := values > spec["max"]).any():
                report.errors.append(
                    f"Column '{name}': {int(above.sum())} value(s) above contract max "
                    f"{spec['max']}{unit} (highest: {values.max()}) — possible unit change?")
        if "allowed_values" in spec:
            report.checks_run += 1
            allowed = set(spec["allowed_values"])
            if bad := sorted(set(values.unique()) - allowed, key=str):
                report.errors.append(
                    f"Column '{name}': values not allowed by contract: {bad[:10]} "
                    f"(allowed: {sorted(allowed, key=str)})")
    return report


def validate_required_features(df: pd.DataFrame, required: list[str],
                               target: str | None = None) -> ValidationReport:
    """Second gate, before training: every feature the model needs exists and is usable."""
    report = ValidationReport("Required features check")
    report.checks_run += 1
    if missing := [c for c in required if c not in df.columns]:
        report.errors.append(f"Features required for modeling are missing: {missing}")
    for col in [c for c in required if c in df.columns]:
        report.checks_run += 1
        if df[col].isna().any():
            report.errors.append(f"Feature '{col}' contains null values")
        elif not pd.api.types.is_numeric_dtype(df[col]):
            report.errors.append(f"Feature '{col}' is not numeric ({df[col].dtype}) — not encoded?")
    if target is not None:
        report.checks_run += 1
        if target not in df.columns:
            report.errors.append(f"Target column '{target}' is missing from features")
        elif df[target].nunique() < 2:
            report.errors.append(f"Target '{target}' has a single class — nothing to learn")
    return report


def run_gate(clean_data_path: str | Path, contract_path: str | Path) -> ValidationReport:
    """Load both Crew 1 artifacts and validate them. Raises ContractViolation on failure."""
    contract = load_contract(contract_path)
    clean_data_path = Path(clean_data_path)
    if not clean_data_path.exists():
        raise ContractViolation("Contract vs. clean data check",
                                [f"Clean data file not found: {clean_data_path}"])
    df = pd.read_csv(clean_data_path)
    report = validate_data_against_contract(df, contract)
    report.raise_if_failed()
    return report
