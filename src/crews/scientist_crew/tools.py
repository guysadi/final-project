"""Tools the Data Scientist agents can call.

The boundary: these tools read only Crew 1's published artifacts (clean_data.csv and
dataset_contract.json) and write only Crew 2's artifacts. Nothing here can reach the raw
data, and every tool that consumes data re-checks the contract first, so even an agent
that skips the validation step cannot train on data that breaks it.
"""
from __future__ import annotations

import json

from crewai.tools import BaseTool

from src import paths
from src.crews.scientist_crew.features import run_feature_engineering
from src.crews.scientist_crew.training import run_training
from src.validation.contract import ContractViolation, run_gate


class ValidateHandoffTool(BaseTool):
    name: str = "validate_handoff"
    description: str = (
        "Validate Crew 1's clean_data.csv against dataset_contract.json: columns, dtypes, "
        "nulls, ranges, allowed values, primary key. Returns PASSED with the number of "
        "checks, or FAILED with every violation. Takes no input.")

    def _run(self) -> str:
        try:
            report = run_gate(paths.CLEAN_DATA, paths.CONTRACT)
        except ContractViolation as e:
            return f"FAILED — do not continue to feature engineering.\n{e}"
        return f"PASSED: {report.checks_run} contract checks, 0 violations."


class EngineerFeaturesTool(BaseTool):
    name: str = "engineer_features"
    description: str = (
        "Build features.csv from the clean data using only what the contract describes: "
        "numeric columns as-is, text columns one-hot encoded with the contract's allowed "
        "values, plus engineered features. Sensitive columns (Gender, MaritalStatus) are "
        "excluded from the model. Re-validates the contract first. Takes no input.")

    def _run(self) -> str:
        try:
            run_gate(paths.CLEAN_DATA, paths.CONTRACT)
            spec = run_feature_engineering(paths.CLEAN_DATA, paths.CONTRACT,
                                           paths.FEATURES, paths.FEATURE_SPEC)
        except ContractViolation as e:
            return f"ERROR: features not built.\n{e}"
        return json.dumps({k: spec[k] for k in (
            "contract_version", "target", "numeric_from_contract", "one_hot_encoded",
            "engineered", "excluded_sensitive")} | {"n_features": len(spec["features"])}, indent=2)


class TrainModelsTool(BaseTool):
    name: str = "train_and_compare_models"
    description: str = (
        "Train and compare three model variations (logistic regression, random forest, "
        "gradient boosting) on features.csv with a fixed seed. Checks every required feature "
        "exists first. Selects the best by 5-fold cross-validated ROC-AUC on the training set, "
        "saves model.joblib, and returns all metrics as JSON: per-model CV and test metrics, "
        "top features, and recall/flag rate by sensitive group. Takes no input.")

    def _run(self) -> str:
        if not paths.FEATURES.exists() or not paths.FEATURE_SPEC.exists():
            return "ERROR: features.csv does not exist yet. engineer_features must run first."
        try:
            metrics = run_training(paths.FEATURES, paths.FEATURE_SPEC, paths.CLEAN_DATA,
                                   paths.MODEL, paths.METRICS)
        except ContractViolation as e:
            return f"ERROR: training refused.\n{e}"
        return json.dumps(metrics, indent=2)


class ReadModelSummaryTool(BaseTool):
    name: str = "read_model_summary"
    description: str = (
        "Read the ground truth for the model card: metrics.json (all model metrics, selected "
        "model, top features, fairness by group), the feature spec, and the dataset "
        "contract's assumptions. Read-only. Takes no input.")

    def _run(self) -> str:
        missing = [p.name for p in (paths.METRICS, paths.FEATURE_SPEC, paths.CONTRACT) if not p.exists()]
        if missing:
            return f"ERROR: missing {missing}. train_and_compare_models must run first."
        contract = json.loads(paths.CONTRACT.read_text())
        return json.dumps({
            "metrics": json.loads(paths.METRICS.read_text()),
            "feature_spec": json.loads(paths.FEATURE_SPEC.read_text()),
            "contract": {"version": contract["version"], "target": contract["target"],
                         "target_description": contract.get("target_description"),
                         "min_rows": contract.get("min_rows"),
                         "assumptions": contract["assumptions"]},
        }, indent=2)
