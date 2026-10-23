"""Tests for the Streamlit app: its helpers, the in-memory break scenarios, and a headless
run of the whole app against the committed artifacts."""
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.helpers import contract_table, input_fields, load_model_bundle, score_customer
from src import paths
from src.break_scenarios import SCENARIOS, run_gates


@pytest.fixture(scope="module")
def contract():
    return json.loads(paths.CONTRACT.read_text())


@pytest.fixture(scope="module")
def clean():
    return pd.read_csv(paths.CLEAN_DATA)


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle()


def default_customer(contract, clean):
    return {f["name"]: f["default"] for f in input_fields(contract, clean)}


def test_contract_table_marks_roles(contract):
    table = contract_table(contract).set_index("column")
    assert table.loc["CustomerID", "role"] == "primary key"
    assert table.loc["Churn", "role"] == "target"
    assert table.loc["Gender", "role"].startswith("sensitive")
    assert table.loc["CashbackAmount", "unit"] == "USD"


def test_form_fields_come_from_the_contract(contract, clean):
    fields = {f["name"]: f for f in input_fields(contract, clean)}
    assert "CustomerID" not in fields and "Churn" not in fields
    assert fields["Gender"]["used_by_model"] is False
    assert fields["PreferredPaymentMode"]["options"] == contract["columns"]["PreferredPaymentMode"]["allowed_values"]


def test_valid_customer_is_scored(contract, clean, bundle):
    report, proba = score_customer(default_customer(contract, clean), contract, bundle)
    assert report.passed and 0 <= proba <= 1


def test_new_customer_with_complaint_scores_higher(contract, clean, bundle):
    base = default_customer(contract, clean)
    _, p_base = score_customer(base, contract, bundle)
    _, p_risky = score_customer(base | {"Tenure": 0, "Complain": 1}, contract, bundle)
    assert p_risky > p_base


def test_customer_breaking_the_contract_is_refused(contract, clean, bundle):
    values = default_customer(contract, clean) | {"CashbackAmount": 30000}
    report, proba = score_customer(values, contract, bundle)
    assert proba is None
    assert any("CashbackAmount" in e and "unit change" in e for e in report.errors)


@pytest.mark.parametrize("name, gates_run, stopped", [
    ("unit_change", 1, True),
    ("rename_column", 1, True),
    ("drop_contract_column", 1, True),
    ("bad_category", 1, True),
    ("remove_feature_source", 2, True),
])
def test_in_memory_scenarios_match_the_flow(contract, clean, name, gates_run, stopped):
    df, c = SCENARIOS[name].apply(clean, contract)
    reports = run_gates(df, c)
    assert len(reports) == gates_run
    assert (not reports[-1].passed) is stopped


def test_scenarios_never_touch_the_originals(contract, clean):
    before_df, before_c = clean.copy(), json.dumps(contract)
    for s in SCENARIOS.values():
        s.apply(clean, contract)
    pd.testing.assert_frame_equal(clean, before_df)
    assert json.dumps(contract) == before_c


def test_unbroken_artifacts_pass_both_gates(contract, clean):
    reports = run_gates(clean, contract)
    assert len(reports) == 2 and all(r.passed for r in reports)


def test_app_runs_without_errors():
    at = AppTest.from_file(str(paths.ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]  # includes nested tabs
    assert {"Pipeline", "Crew 1 · Data", "Crew 2 · Model", "Score a customer", "Break it",
            "Dataset contract", "Model card"} <= set(labels)
    # The committed artifacts are healthy: the Pipeline tab reports success, and the only
    # error boxes are the Break it tab's default (deliberately broken) scenario.
    assert any("succeeded" in s.value for s in at.success)
    assert all("Gate 1" in e.value or "Flow stops here" in e.value for e in at.error)
