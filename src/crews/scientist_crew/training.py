"""Deterministic training, comparison and evaluation of churn models.

- One stratified train/test split, fixed seed.
- Every variation is compared by 5-fold cross-validated ROC-AUC on the training set;
  the winner is chosen on that, never on the test set.
- The test set is used once, to report the chosen model's (and the others') metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.crews.scientist_crew.features import SENSITIVE_COLUMNS
from src.validation.contract import validate_required_features

SEED = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5


def model_variations() -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
    }


def _test_metrics(y_true, proba) -> dict:
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _fairness(y_true: pd.Series, proba: np.ndarray, groups: pd.DataFrame) -> dict:
    """Recall and flag rate per sensitive group: does the model miss churners in one group
    more than another, or flag one group far more often?"""
    pred = (proba >= THRESHOLD).astype(int)
    out = {}
    for col in groups.columns:
        out[col] = {}
        for value in sorted(groups[col].unique()):
            mask = (groups[col] == value).to_numpy()
            yt, yp = y_true.to_numpy()[mask], pred[mask]
            out[col][str(value)] = {
                "customers": int(mask.sum()),
                "actual_churn_rate": round(float(yt.mean()), 4),
                "flagged_rate": round(float(yp.mean()), 4),
                "recall": round(recall_score(yt, yp, zero_division=0), 4),
            }
    return out


def train_and_evaluate(features: pd.DataFrame, spec: dict,
                       sensitive: pd.DataFrame | None = None) -> tuple[object, dict]:
    target, feats = spec["target"], spec["features"]
    validate_required_features(features, feats, target=target).raise_if_failed()

    X, y = features[feats], features[target]
    idx_train, idx_test = train_test_split(
        features.index, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train, X_test, y_train, y_test = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in model_variations().items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        fitted[name] = model
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "test": _test_metrics(y_test, model.predict_proba(X_test)[:, 1]),
        }

    best = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    model = fitted[best]
    proba = model.predict_proba(X_test)[:, 1]

    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = (pd.Series(imp.importances_mean, index=feats)
                  .sort_values(ascending=False).round(4).head(10).to_dict())

    metrics = {
        "contract_version": spec["contract_version"],
        "selection_rule": "highest 5-fold cross-validated ROC-AUC on the training set",
        "decision_threshold": THRESHOLD,
        "seed": SEED,
        "rows": {"train": len(idx_train), "test": len(idx_test)},
        "churn_rate": {"train": round(float(y_train.mean()), 4), "test": round(float(y_test.mean()), 4)},
        "n_features": len(feats),
        "models": results,
        "best_model": best,
        "top_features_permutation_importance": importance,
    }
    if sensitive is not None:
        groups = sensitive.loc[idx_test, [c for c in SENSITIVE_COLUMNS if c in sensitive.columns]]
        metrics["fairness_by_group"] = _fairness(y_test, proba, groups)
    return model, metrics


def run_training(features_path: str | Path, spec_path: str | Path, clean_path: str | Path,
                 model_path: str | Path, metrics_path: str | Path) -> dict:
    features = pd.read_csv(features_path)
    spec = json.loads(Path(spec_path).read_text())
    # Sensitive columns come from Crew 1's clean data (a contract artifact), joined by key,
    # only for the fairness audit — they never reach the model.
    clean = pd.read_csv(clean_path)
    pk = spec["primary_key"]
    sensitive = features[[pk]].merge(clean[[pk, *SENSITIVE_COLUMNS]], on=pk, how="left")
    sensitive.index = features.index

    model, metrics = train_and_evaluate(features, spec, sensitive)
    joblib.dump({"model": model, "features": spec["features"], "target": spec["target"],
                 "contract_version": spec["contract_version"],
                 "model_name": metrics["best_model"], "threshold": THRESHOLD},
                model_path, compress=3)  # ~15MB uncompressed: too heavy for the repo
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    return metrics


def metrics_table(metrics: dict) -> str:
    """Markdown comparison table built straight from metrics.json — no LLM in the loop."""
    rows = ["| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|---|"]
    for name, r in metrics["models"].items():
        t = r["test"]
        star = " **(selected)**" if name == metrics["best_model"] else ""
        rows.append(f"| {name}{star} | {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} | "
                    f"{t['roc_auc']:.4f} | {t['pr_auc']:.4f} | {t['precision']:.4f} | "
                    f"{t['recall']:.4f} | {t['f1']:.4f} |")
    return "\n".join(rows)
