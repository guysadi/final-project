"""Ways a handoff goes wrong, as pure functions: (clean data, contract) → broken copies.

Shared by scripts/break_handoff.py (which writes the broken files for a Flow run) and
the Streamlit app (which breaks copies in memory and runs the gates on them live).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.crews.scientist_crew.features import validate_feature_requirements
from src.validation.contract import (
    ValidationReport,
    validate_contract_structure,
    validate_data_against_contract,
)

Transform = Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    transform: Transform

    def apply(self, df: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, dict]:
        return self.transform(df.copy(), copy.deepcopy(contract))


def _unit_change(df, c):
    df["CashbackAmount"] = (df["CashbackAmount"] * 100).round(2)
    return df, c


def _rename_column(df, c):
    return df.rename(columns={"Tenure": "tenure_months"}), c


def _drop_contract_column(df, c):
    c["columns"].pop("CouponUsed")
    return df, c


def _bad_category(df, c):
    df["Gender"] = df["Gender"].replace({"Female": "F"})
    return df, c


def _remove_feature_source(df, c):
    c["columns"].pop("OrderCount")
    return df.drop(columns="OrderCount"), c


SCENARIOS = {s.name: s for s in [
    Scenario("unit_change", "CashbackAmount silently converted from dollars to cents.", _unit_change),
    Scenario("rename_column", "Tenure renamed to match a 'new naming convention'.", _rename_column),
    Scenario("drop_contract_column",
             "A row deleted from the contract: data has a column the contract no longer describes.",
             _drop_contract_column),
    Scenario("bad_category", "Someone abbreviates a category value (Female → F).", _bad_category),
    Scenario("remove_feature_source",
             "OrderCount dropped from both data and contract. They still agree, so Gate 1 passes — "
             "but the model needs OrderCount, so Gate 2 stops the Flow before modeling.",
             _remove_feature_source),
]}


def run_gates(df: pd.DataFrame, contract: dict) -> list[ValidationReport]:
    """The Flow's two gates, in order, on in-memory data. Stops at the first failure,
    exactly like the Flow does."""
    structure = validate_contract_structure(contract)
    if not structure.passed:
        return [structure]
    gate1 = validate_data_against_contract(df, contract)
    if not gate1.passed:
        return [gate1]
    return [gate1, validate_feature_requirements(contract)]
