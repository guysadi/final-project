"""Tests for Crew 2: features from the contract, training, the boundary with Crew 1,
and refusing to train when the handoff is broken. No LLM calls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import paths
from src.crews.analyst_crew.contract_builder import build_contract
from src.crews.scientist_crew import tools
from src.crews.scientist_crew.features import SENSITIVE_COLUMNS, build_features
from src.crews.scientist_crew.training import train_and_evaluate
from src.validation.contract import ContractViolation


def make_clean(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 40, n)
    complain = rng.integers(0, 2, n)
    churn_p = 1 / (1 + np.exp(0.25 * tenure - 1.5 * complain))
    return pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Churn": (rng.random(n) < churn_p).astype("int64"),
        "Tenure": tenure,
        "Complain": complain,
        "OrderCount": rng.integers(1, 15, n),
        "CouponUsed": rng.integers(0, 10, n),
        "DaySinceLastOrder": rng.integers(0, 30, n),
        "CashbackAmount": rng.uniform(0, 300, n).round(2),
        "PreferredPaymentMode": rng.choice(["Credit Card", "Debit Card", "UPI"], n),
        "Gender": rng.choice(["Female", "Male"], n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
    })


@pytest.fixture
def clean():
    return make_clean()


@pytest.fixture
def contract(clean):
    c = build_contract(clean)
    # A category the contract allows but this file happens not to contain.
    c["columns"]["PreferredPaymentMode"]["allowed_values"].append("E wallet")
    return c


# --- Features ------------------------------------------------------------------

def test_features_come_from_the_contract(clean, contract):
    features, spec = build_features(clean, contract)
    assert features.columns[0] == "CustomerID" and features.columns[-1] == "Churn"
    for col in SENSITIVE_COLUMNS:
        assert not any(f.startswith(col) for f in spec["features"])
    # One-hot columns follow the contract's categories, even one absent from the data.
    assert "PreferredPaymentMode_E_wallet" in spec["features"]
    assert {"is_new_customer", "cashback_per_order", "coupons_per_order"} <= set(spec["features"])


def test_columns_outside_the_contract_are_never_used(clean, contract):
    clean["SecretLeakyColumn"] = clean["Churn"]
    _, spec = build_features(clean, contract)
    assert "SecretLeakyColumn" not in spec["features"]


def test_engineered_feature_without_its_source_in_contract_fails(clean, contract):
    del contract["columns"]["CouponUsed"]
    with pytest.raises(ContractViolation, match="CouponUsed"):
        build_features(clean, contract)


# --- Training --------------------------------------------------------------------

def test_training_is_deterministic(clean, contract):
    features, spec = build_features(clean, contract)
    _, m1 = train_and_evaluate(features, spec)
    _, m2 = train_and_evaluate(features, spec)
    assert m1 == m2


def test_best_model_is_chosen_by_cross_validation(clean, contract):
    features, spec = build_features(clean, contract)
    _, m = train_and_evaluate(features, spec)
    assert set(m["models"]) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert m["best_model"] == max(m["models"], key=lambda n: m["models"][n]["cv_roc_auc_mean"])


def test_training_refuses_when_a_required_feature_is_missing(clean, contract):
    features, spec = build_features(clean, contract)
    with pytest.raises(ContractViolation, match="is_new_customer"):
        train_and_evaluate(features.drop(columns="is_new_customer"), spec)


def test_fairness_is_reported_per_sensitive_group(clean, contract):
    features, spec = build_features(clean, contract)
    _, m = train_and_evaluate(features, spec, sensitive=clean[SENSITIVE_COLUMNS])
    assert set(m["fairness_by_group"]["Gender"]) == {"Female", "Male"}
    assert {"recall", "flagged_rate"} <= set(m["fairness_by_group"]["Gender"]["Male"])


# --- Tools and the handoff ------------------------------------------------------------

@pytest.fixture
def handoff(tmp_path, monkeypatch, clean, contract):
    for name in ("CLEAN_DATA", "CONTRACT", "FEATURES", "FEATURE_SPEC", "MODEL", "METRICS",
                 "EVAL_REPORT", "MODEL_CARD"):
        monkeypatch.setattr(paths, name, tmp_path / getattr(paths, name).name)
    monkeypatch.setattr(paths, "CREW2_ARTIFACTS",
                        [paths.FEATURES, paths.MODEL, paths.EVAL_REPORT, paths.MODEL_CARD])
    clean.to_csv(paths.CLEAN_DATA, index=False)
    paths.CONTRACT.write_text(json.dumps(contract))
    return clean


def test_tools_run_the_whole_chain(handoff):
    assert tools.ValidateHandoffTool()._run().startswith("PASSED")
    assert "n_features" in tools.EngineerFeaturesTool()._run()
    metrics = json.loads(tools.TrainModelsTool()._run())
    assert paths.MODEL.exists() and metrics["best_model"]
    summary = json.loads(tools.ReadModelSummaryTool()._run())
    assert summary["contract"]["assumptions"]


def test_unit_change_stops_crew2_tools(handoff):
    """The Harbor & Vale incident: dollars became cents. Nothing downstream may run."""
    handoff["CashbackAmount"] *= 100
    handoff.to_csv(paths.CLEAN_DATA, index=False)
    assert tools.ValidateHandoffTool()._run().startswith("FAILED")
    assert tools.EngineerFeaturesTool()._run().startswith("ERROR")
    assert not paths.FEATURES.exists()
    assert tools.TrainModelsTool()._run().startswith("ERROR")
    assert not paths.MODEL.exists()


def test_broken_handoff_stops_crew2_before_any_agent_runs(handoff, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    from src.crews.scientist_crew import crew as crew_module

    def must_not_run(*a, **k):
        raise AssertionError("agents started despite a broken contract")
    monkeypatch.setattr(crew_module, "ScientistCrew", must_not_run)

    handoff.rename(columns={"Tenure": "tenure_months"}).to_csv(paths.CLEAN_DATA, index=False)
    with pytest.raises(ContractViolation, match="Tenure"):
        crew_module.run_scientist_crew()


def test_crew2_never_touches_raw_data():
    """Crew 2 may only build on Crew 1's published artifacts."""
    crew2_dir = Path(__file__).resolve().parent.parent / "src" / "crews" / "scientist_crew"
    for f in crew2_dir.rglob("*.py"):
        src = f.read_text()
        assert "RAW_DATA" not in src and "data/raw" not in src, f"{f.name} reaches into raw data"


def test_crew_wiring(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    from src.crews.scientist_crew.crew import ScientistCrew

    crew = ScientistCrew().crew()
    assert [t.agent.role.strip() for t in crew.tasks] == [
        "Feature Engineer", "Machine Learning Engineer", "Model Card Author"]
