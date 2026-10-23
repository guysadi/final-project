"""Logic behind the Streamlit app, kept free of Streamlit so it can be tested."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import pandas as pd

from src import paths
from src.crews.scientist_crew.features import SENSITIVE_COLUMNS, build_features
from src.validation.contract import ValidationReport, validate_data_against_contract


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_model_bundle() -> dict | None:
    return joblib.load(paths.MODEL) if paths.MODEL.exists() else None


def contract_table(contract: dict) -> pd.DataFrame:
    """One row per contract column, readable by a person."""
    rows = []
    for name, spec in contract["columns"].items():
        if "allowed_values" in spec:
            rule = ", ".join(map(str, spec["allowed_values"]))
        elif "min" in spec or "max" in spec:
            rule = f"{spec.get('min', '−∞')} – {spec.get('max', '∞')}"
        else:
            rule = ""
        role = ("primary key" if name == contract.get("primary_key")
                else "target" if name == contract["target"]
                else "sensitive — audit only" if name in SENSITIVE_COLUMNS else "")
        rows.append({"column": name, "dtype": spec["dtype"], "unit": spec.get("unit", ""),
                     "allowed": rule, "nullable": spec.get("nullable", False),
                     "role": role, "description": spec.get("description", "")})
    return pd.DataFrame(rows)


def input_fields(contract: dict, clean: pd.DataFrame) -> list[dict]:
    """Form fields for scoring one customer, derived from the contract. Numeric fields
    are deliberately NOT clamped to the contract's range — entering an out-of-range value
    shows the contract rejecting it, just like Gate 1 would."""
    fields = []
    for name, spec in contract["columns"].items():
        if name in (contract.get("primary_key"), contract["target"]):
            continue
        field = {"name": name, "dtype": spec["dtype"], "unit": spec.get("unit", ""),
                 "description": spec.get("description", ""),
                 "used_by_model": name not in SENSITIVE_COLUMNS}
        if "allowed_values" in spec:
            field["kind"] = "choice"
            field["options"] = spec["allowed_values"]
            field["default"] = clean[name].mode().iloc[0]
        else:
            field["kind"] = "number"
            field["default"] = float(clean[name].median())
            field["integer"] = spec["dtype"] == "integer"
            field["contract_range"] = (spec.get("min"), spec.get("max"))
        fields.append(field)
    return fields


def _single_row_contract(contract: dict) -> dict:
    c = copy.deepcopy(contract)
    c.pop("min_rows", None)  # a single customer is not a dataset
    return c


def score_customer(values: dict, contract: dict, bundle: dict) -> tuple[ValidationReport, float | None]:
    """Validate one customer against the contract, then score it. Returns the validation
    report and the churn probability (None if the customer breaks the contract)."""
    row = {contract["primary_key"]: 0, contract["target"]: 0, **values}
    df = pd.DataFrame([row])[list(contract["columns"])]
    report = validate_data_against_contract(df, _single_row_contract(contract))
    if not report.passed:
        return report, None
    features, _ = build_features(df, contract)
    proba = float(bundle["model"].predict_proba(features[bundle["features"]])[:, 1][0])
    return report, proba
