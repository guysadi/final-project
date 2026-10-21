"""Tests for Crew 1's deterministic parts, its tools, and the crew wiring (no LLM calls)."""
import json

import pandas as pd
import pytest

from src import paths
from src.crews.analyst_crew import tools
from src.crews.analyst_crew.cleaning import clean
from src.crews.analyst_crew.contract_builder import build_contract
from src.validation.contract import validate_data_against_contract


@pytest.fixture
def raw():
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Churn": [0, 1, 0, 1, 1],
        "Tenure": [4.0, None, 12.0, 30.0, None],
        "PreferredLoginDevice": ["Phone", "Mobile Phone", "Computer", "Phone", "Mobile Phone"],
        "CityTier": [1, 3, 1, 2, 3],
        "WarehouseToHome": [6.0, 8.0, 30.0, 15.0, 8.0],
        "PreferredPaymentMode": ["CC", "Debit Card", "COD", "UPI", "Debit Card"],
        "HourSpendOnApp": [3.0, 3.0, 2.0, None, 3.0],
        "PreferedOrderCat": ["Mobile", "Fashion", "Grocery", "Mobile Phone", "Fashion"],
        "OrderAmountHikeFromlastYear": [11.0, 15.0, 14.0, 23.0, 15.0],
        "CouponUsed": [1.0, 0.0, 1.0, 1.0, 0.0],
        "OrderCount": [1.0, 1.0, 1.0, 1.0, 1.0],
        "DaySinceLastOrder": [5.0, 0.0, 3.0, 3.0, 0.0],
        "CashbackAmount": [159.93, 120.9, 120.28, 134.07, 120.9],
    })


def test_clean_drops_duplicates_ignoring_id(raw):
    df, log = clean(raw)
    # Customers 2 and 5 are identical apart from CustomerID.
    assert len(df) == 4
    assert any("duplicate" in line for line in log)


def test_clean_merges_category_spellings(raw):
    df, _ = clean(raw)
    assert set(df["PreferredLoginDevice"]) == {"Mobile Phone", "Computer"}
    assert set(df["PreferredPaymentMode"]) == {"Credit Card", "Debit Card", "Cash on Delivery", "UPI"}
    assert "Mobile" not in set(df["PreferedOrderCat"])


def test_clean_leaves_no_nulls_and_integer_counts(raw):
    df, _ = clean(raw)
    assert not df.isna().any().any()
    assert pd.api.types.is_integer_dtype(df["Tenure"])


def test_clean_is_deterministic(raw):
    a, _ = clean(raw)
    b, _ = clean(raw.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(a, b)


def test_built_contract_holds_on_its_own_data(raw):
    df, _ = clean(raw)
    contract = build_contract(df, units={"CashbackAmount": "USD"})
    assert validate_data_against_contract(df, contract).passed
    assert contract["columns"]["CityTier"]["allowed_values"] == [1, 2, 3]
    assert contract["columns"]["CashbackAmount"]["unit"] == "USD"


def test_built_contract_catches_dollars_to_cents(raw):
    df, _ = clean(raw)
    contract = build_contract(df)
    df["CashbackAmount"] *= 100
    errors = validate_data_against_contract(df, contract).errors
    assert any("CashbackAmount" in e and "unit change" in e for e in errors)


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch, raw):
    """Point the tools at a temp folder holding a tiny raw dataset."""
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    monkeypatch.setattr(paths, "RAW_DATA", raw_path)
    for name in ("CLEAN_DATA", "EDA_REPORT", "CONTRACT", "CLEANING_SUMMARY", "EDA_STATS"):
        monkeypatch.setattr(paths, name, tmp_path / getattr(paths, name).name)
    monkeypatch.setattr(tools, "load_data_dictionary", lambda _: {"Churn": "Churn flag"})
    return tmp_path


def test_tools_produce_artifacts_in_order(tmp_paths):
    assert "ERROR" in tools.RunEdaTool()._run()  # nothing cleaned yet

    tools.CleanDataTool()._run()
    assert paths.CLEAN_DATA.exists() and paths.CLEANING_SUMMARY.exists()

    stats = json.loads(tools.RunEdaTool()._run())
    assert paths.EDA_REPORT.exists()
    assert 0 < stats["overall_churn_rate"] < 1

    msg = tools.WriteContractTool()._run(units={"Tenure": "months"}, assumptions=["One row per customer"])
    assert "Validated" in msg
    contract = json.loads(paths.CONTRACT.read_text())
    assert contract["columns"]["Tenure"]["unit"] == "months"
    assert contract["columns"]["Churn"]["description"] == "Churn flag"


def test_crew_wiring(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    from src.crews.analyst_crew.crew import AnalystCrew

    crew = AnalystCrew().crew()
    assert len(crew.agents) == 4
    assert [t.agent.role.strip() for t in crew.tasks] == [
        "Data Engineer", "Exploratory Data Analyst", "Business Insights Analyst", "Dataset Contract Author"]
