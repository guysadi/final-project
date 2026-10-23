"""Deterministic feature engineering, driven by the dataset contract.

Crew 2 never looks past the contract: column types, categories and the target all
come from dataset_contract.json. A column the contract doesn't describe is never used,
and an engineered feature whose source columns are missing from the contract fails
loudly instead of being silently skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.validation.contract import ValidationReport

# Kept out of the model on purpose; used only to audit the model for fairness.
SENSITIVE_COLUMNS = ["Gender", "MaritalStatus"]


@dataclass(frozen=True)
class EngineeredFeature:
    name: str
    sources: tuple[str, ...]
    description: str


ENGINEERED = [
    EngineeredFeature("is_new_customer", ("Tenure",),
                      "1 if Tenure <= 1 month — churn is ~51% in this group vs <10% after"),
    EngineeredFeature("cashback_per_order", ("CashbackAmount", "OrderCount"),
                      "CashbackAmount / OrderCount (OrderCount floored at 1)"),
    EngineeredFeature("coupons_per_order", ("CouponUsed", "OrderCount"),
                      "CouponUsed / OrderCount (OrderCount floored at 1)"),
    EngineeredFeature("days_since_order_per_tenure", ("DaySinceLastOrder", "Tenure"),
                      "DaySinceLastOrder / (Tenure + 1): recency relative to customer age"),
]


def required_source_columns() -> list[str]:
    return sorted({s for f in ENGINEERED for s in f.sources})


def validate_feature_requirements(contract: dict) -> ValidationReport:
    """Gate before modeling: does the contract promise every column the model needs?
    Run by the Flow before Crew 2 starts, so a gap is caught before any tokens are spent."""
    report = ValidationReport("Required features check (before modeling)")
    columns = contract.get("columns", {})
    for key in ("target", "primary_key"):
        report.checks_run += 1
        if contract.get(key) not in columns:
            report.errors.append(f"Contract {key} '{contract.get(key)}' is not a contract column")
    for f in ENGINEERED:
        report.checks_run += 1
        if missing := [s for s in f.sources if s not in columns]:
            report.errors.append(f"Feature '{f.name}' needs {missing}, which the contract does not provide")
    for col in required_source_columns():
        if col in columns:
            report.checks_run += 1
            if columns[col]["dtype"] == "string":
                report.errors.append(f"Feature source '{col}' must be numeric, contract says string")
    report.checks_run += 1
    if not any(c not in {contract.get("primary_key"), contract.get("target"), *SENSITIVE_COLUMNS}
               for c in columns):
        report.errors.append("Contract leaves no usable feature columns")
    return report


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    orders = df["OrderCount"].clip(lower=1)
    return pd.DataFrame({
        "is_new_customer": (df["Tenure"] <= 1).astype("int64"),
        "cashback_per_order": (df["CashbackAmount"] / orders).round(4),
        "coupons_per_order": (df["CouponUsed"] / orders).round(4),
        "days_since_order_per_tenure": (df["DaySinceLastOrder"] / (df["Tenure"] + 1)).round(4),
    }, index=df.index)


def build_features(df: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, dict]:
    """Return (features dataframe, feature spec). The dataframe keeps the primary key and
    target alongside the model features so every row stays traceable."""
    columns, pk, target = contract["columns"], contract.get("primary_key"), contract["target"]

    validate_feature_requirements(contract).raise_if_failed()

    excluded = {pk, target, *SENSITIVE_COLUMNS}
    numeric = [c for c, s in columns.items() if c not in excluded and s["dtype"] != "string"]
    categorical = [c for c, s in columns.items() if c not in excluded and s["dtype"] == "string"]

    parts = [df[numeric]]
    for col in categorical:
        # Categories come from the contract, not the data, so the feature set is fixed
        # even if one category happens to be absent from a given file.
        cats = pd.Categorical(df[col], categories=columns[col]["allowed_values"])
        dummies = pd.get_dummies(cats, prefix=col, drop_first=True, dtype="int64")
        dummies.index = df.index
        parts.append(dummies)
    parts.append(_engineer(df))

    X = pd.concat(parts, axis=1)
    X.columns = [c.replace(" ", "_").replace("&", "and") for c in X.columns]
    out = pd.concat([df[[pk]], X, df[[target]]], axis=1)

    spec = {
        "contract_version": contract["version"],
        "primary_key": pk,
        "target": target,
        "features": list(X.columns),
        "numeric_from_contract": numeric,
        "one_hot_encoded": categorical,
        "engineered": [{"name": f.name, "sources": list(f.sources), "description": f.description}
                       for f in ENGINEERED],
        "excluded_sensitive": SENSITIVE_COLUMNS,
    }
    return out, spec


def run_feature_engineering(clean_path: str | Path, contract_path: str | Path,
                            features_path: str | Path, spec_path: str | Path) -> dict:
    contract = json.loads(Path(contract_path).read_text())
    df = pd.read_csv(clean_path)
    features, spec = build_features(df, contract)
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    Path(spec_path).write_text(json.dumps(spec, indent=2))
    return spec
