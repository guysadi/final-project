"""Tests for the validation gate. Most of them break the data on purpose,
because a validation you've never seen fail is untested code."""
import json

import pandas as pd
import pytest

from src.validation.contract import (
    ContractViolation,
    load_contract,
    run_gate,
    validate_contract_structure,
    validate_data_against_contract,
    validate_required_features,
)

CONTRACT = {
    "dataset": "clean_data.csv",
    "version": "1.0",
    "primary_key": "CustomerID",
    "target": "Churn",
    "min_rows": 3,
    "columns": {
        "CustomerID": {"dtype": "integer", "nullable": False},
        "Churn": {"dtype": "integer", "nullable": False, "allowed_values": [0, 1]},
        "Tenure": {"dtype": "number", "nullable": False, "min": 0, "max": 61},
        "CashbackAmount": {"dtype": "number", "nullable": False,
                           "min": 0, "max": 400, "unit": "USD"},
        "Gender": {"dtype": "string", "nullable": False,
                   "allowed_values": ["Female", "Male"]},
    },
}


@pytest.fixture
def df():
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "Churn": [0, 1, 0, 1],
        "Tenure": [4.0, 0.0, 12.0, 30.0],
        "CashbackAmount": [159.9, 120.9, 324.99, 0.0],
        "Gender": ["Female", "Male", "Male", "Female"],
    })


def errors_for(df):
    return validate_data_against_contract(df, CONTRACT).errors


def test_clean_data_passes(df):
    report = validate_data_against_contract(df, CONTRACT)
    assert report.passed, report.errors
    assert report.checks_run > 10


# --- Break it on purpose -----------------------------------------------------

def test_unit_change_dollars_to_cents_is_caught(df):
    df["CashbackAmount"] = df["CashbackAmount"] * 100
    [err] = errors_for(df)
    assert "CashbackAmount" in err and "possible unit change" in err


def test_renamed_column_is_caught(df):
    df = df.rename(columns={"Tenure": "tenure_months"})
    errs = errors_for(df)
    assert any("missing from data: ['Tenure']" in e for e in errs)
    assert any("not in contract" in e and "tenure_months" in e for e in errs)


def test_column_deleted_from_contract_is_caught(df):
    contract = json.loads(json.dumps(CONTRACT))
    del contract["columns"]["Gender"]
    errs = validate_data_against_contract(df, contract).errors
    assert any("not in contract" in e and "Gender" in e for e in errs)


def test_nulls_in_non_nullable_column_are_caught(df):
    df.loc[0, "Tenure"] = None
    assert any("Tenure" in e and "null" in e for e in errors_for(df))


def test_unknown_category_is_caught(df):
    df.loc[0, "Gender"] = "F"
    assert any("Gender" in e and "'F'" in e for e in errors_for(df))


def test_value_below_min_is_caught(df):
    df.loc[0, "Tenure"] = -1
    assert any("Tenure" in e and "below contract min" in e for e in errors_for(df))


def test_duplicate_primary_key_is_caught(df):
    df.loc[1, "CustomerID"] = 1
    assert any("duplicate" in e for e in errors_for(df))


def test_wrong_dtype_is_caught(df):
    df["Tenure"] = df["Tenure"].astype(str)
    assert any("Tenure" in e and "expected number" in e for e in errors_for(df))


def test_too_few_rows_is_caught(df):
    assert any("rows" in e for e in errors_for(df.head(2)))


# --- Contract file itself ----------------------------------------------------

def test_malformed_contract_is_rejected():
    bad = {"dataset": "x.csv", "version": "1", "target": "Churn",
           "columns": {"Churn": {"dtype": "boolean"}, "A": {"dtype": "number", "min": 5, "max": 1}}}
    errs = validate_contract_structure(bad).errors
    assert any("boolean" in e for e in errs)
    assert any("min 5 > max 1" in e for e in errs)


def test_missing_contract_keys_are_rejected():
    assert not validate_contract_structure({"columns": {}}).passed


def test_missing_contract_file_raises(tmp_path):
    with pytest.raises(ContractViolation, match="not found"):
        load_contract(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    with pytest.raises(ContractViolation, match="not valid JSON"):
        load_contract(p)


# --- End-to-end gate on files --------------------------------------------------

def test_run_gate_passes_and_fails_on_files(tmp_path, df):
    contract_path = tmp_path / "dataset_contract.json"
    data_path = tmp_path / "clean_data.csv"
    contract_path.write_text(json.dumps(CONTRACT))
    df.to_csv(data_path, index=False)
    assert run_gate(data_path, contract_path).passed

    df["CashbackAmount"] *= 100
    df.to_csv(data_path, index=False)
    with pytest.raises(ContractViolation) as exc:
        run_gate(data_path, contract_path)
    assert "CashbackAmount" in str(exc.value)


# --- Required features gate ---------------------------------------------------

def test_required_features_pass():
    feats = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.1], "Churn": [0, 1]})
    assert validate_required_features(feats, ["a", "b"], target="Churn").passed


def test_required_features_catch_missing_unencoded_and_single_class():
    feats = pd.DataFrame({"a": [1, None], "cat": ["x", "y"], "Churn": [0, 0]})
    errs = validate_required_features(feats, ["a", "cat", "b"], target="Churn").errors
    assert any("missing: ['b']" in e for e in errs)
    assert any("'a' contains null" in e for e in errs)
    assert any("'cat' is not numeric" in e for e in errs)
    assert any("single class" in e for e in errs)
