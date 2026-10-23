"""Harbor & Vale churn pipeline — demo app.

    streamlit run app/streamlit_app.py

Reads the artifacts the Flow committed to the repo; it never calls an LLM.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.helpers import (  # noqa: E402
    contract_table,
    input_fields,
    load_json,
    load_model_bundle,
    score_customer,
)
from src import paths  # noqa: E402
from src.break_scenarios import SCENARIOS, run_gates  # noqa: E402
from src.crews.scientist_crew.features import validate_feature_requirements  # noqa: E402
from src.validation.contract import ContractViolation, load_contract, run_gate  # noqa: E402

SERIES = "#2a78d6"  # single-series blue, validated in light and dark

st.set_page_config(page_title="Harbor & Vale — Churn Pipeline", page_icon="🛡️", layout="wide")


@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(paths.CLEAN_DATA)


@st.cache_resource
def load_bundle():
    return load_model_bundle()


def read_text(path: Path) -> str | None:
    return path.read_text() if path.exists() else None


def read_doc(path: Path) -> str:
    """A markdown artifact for display inside a tab: headings two levels smaller."""
    text = read_text(path)
    if text is None:
        return f"_{path.name} not found_"
    return re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", text, flags=re.M)


def hbar(data: pd.DataFrame, x: str, y: str, x_title: str, fmt: str = ".3f") -> alt.Chart:
    """Single-series horizontal bars: thin, rounded data end, value tooltip."""
    return (alt.Chart(data)
            .mark_bar(color=SERIES, cornerRadiusEnd=4, height={"band": 0.6})
            .encode(x=alt.X(f"{x}:Q", title=x_title),
                    y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                    tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)])
            .properties(height=alt.Step(26)))


def show_report(report, label: str) -> None:
    if report.passed:
        st.success(f"**{label}** — passed ({report.checks_run} checks)", icon="✅")
    else:
        st.error(f"**{label}** — failed", icon="⛔")
        for e in report.errors:
            st.markdown(f"- `{e}`")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Harbor & Vale — contract-enforced churn pipeline")
st.caption("Two CrewAI crews, one dataset contract between them, and a Flow that refuses to "
           "train when the contract is broken.")

summary = load_json(paths.RUN_SUMMARY)
contract = load_json(paths.CONTRACT)
metrics = load_json(paths.METRICS)

if contract is None or not paths.CLEAN_DATA.exists():
    st.error("Crew 1's artifacts are missing. Run the Flow first: `python -m src.flow`", icon="⛔")
    st.stop()

tab_flow, tab_data, tab_model, tab_score, tab_break = st.tabs(
    ["Pipeline", "Crew 1 · Data", "Crew 2 · Model", "Score a customer", "Break it"])

# ── Pipeline ──────────────────────────────────────────────────────────────────
with tab_flow:
    if summary is None:
        st.info("No Flow run recorded yet. Run `python -m src.flow`.")
    else:
        ok = summary["status"] == "succeeded"
        (st.success if ok else st.error)(
            f"Last run **{summary['run_id']}** — "
            + ("succeeded: all 8 artifacts written." if ok
               else f"stopped at **{summary['failed_stage']}**. No model was trained on that data."),
            icon="✅" if ok else "⛔")

        cols = st.columns(len(summary["stages"]))
        for col, stage in zip(cols, summary["stages"]):
            icon = {"passed": "✅", "failed": "⛔", "skipped": "⏭️"}[stage["status"]]
            name, _, detail = stage["name"].partition(" — ")
            with col.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(detail)
                checks = f" · {stage['checks_run']} checks" if stage.get("checks_run") else ""
                st.markdown(f"{stage['status']} · {stage['seconds']:.1f}s{checks}")
                for e in stage["errors"]:
                    st.markdown(f"- `{e}`")

        st.subheader("Artifacts")
        st.caption("Hashes from the last run. Deterministic artifacts (clean data, features, "
                   "model) hash identically on every run; LLM-written documents do not.")
        st.dataframe(pd.DataFrame([{"artifact": k, "sha256 (first 16)": v}
                                   for k, v in summary["artifacts"].items()]),
                     hide_index=True, width="stretch")

    st.subheader("Check the handoff now")
    st.caption("Runs Gate 1 and Gate 2 on the files currently in the repo — no agents, no tokens.")
    if st.button("Run both gates", type="primary"):
        try:
            show_report(run_gate(paths.CLEAN_DATA, paths.CONTRACT), "Gate 1 — clean data matches contract")
            show_report(validate_feature_requirements(load_contract(paths.CONTRACT)),
                        "Gate 2 — contract provides every required feature")
        except ContractViolation as e:
            st.error(f"**{e.stage}** — failed", icon="⛔")
            for err in e.errors:
                st.markdown(f"- `{err}`")

# ── Crew 1 · Data ─────────────────────────────────────────────────────────────
with tab_data:
    clean = load_clean()
    cleaning = load_json(paths.CLEANING_SUMMARY) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers (clean)", f"{len(clean):,}")
    c2.metric("Churn rate", f"{clean[contract['target']].mean():.1%}")
    c3.metric("Churned customers", f"{int(clean[contract['target']].sum()):,}")
    if cleaning:
        c4.metric("Raw rows removed", f"{cleaning['before']['rows'] - cleaning['after']['rows']:,}",
                  help="Duplicate customers (identical except CustomerID)")

    insights_tab, contract_tab, eda_tab = st.tabs(["Business insights", "Dataset contract", "EDA report"])
    with insights_tab:
        st.markdown(read_doc(paths.INSIGHTS))
    with contract_tab:
        st.markdown(f"**Version {contract['version']}** · target `{contract['target']}` "
                    f"({contract.get('target_description', '')}) · primary key "
                    f"`{contract.get('primary_key')}` · at least {contract.get('min_rows', 0):,} rows")
        st.dataframe(contract_table(contract), hide_index=True, width="stretch")
        st.markdown("**Assumptions the modeling crew may rely on**")
        for a in contract["assumptions"]:
            st.markdown(f"- {a}")
    with eda_tab:
        # Rendered by Crew 1's deterministic EDA code (not LLM text), so safe to embed.
        if paths.EDA_REPORT.exists():
            st.iframe(paths.EDA_REPORT, height=900)

# ── Crew 2 · Model ────────────────────────────────────────────────────────────
with tab_model:
    if metrics is None:
        st.info("Crew 2 hasn't produced a model yet.")
    else:
        best = metrics["best_model"]
        t = metrics["models"][best]["test"]
        st.markdown(f"Selected model: **{best.replace('_', ' ')}** — {metrics['selection_rule']}.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test ROC-AUC", f"{t['roc_auc']:.3f}")
        m2.metric("Recall", f"{t['recall']:.1%}", help="Share of real churners the model flags")
        m3.metric("Precision", f"{t['precision']:.1%}", help="Share of flagged customers who really churn")
        m4.metric("Test customers", f"{metrics['rows']['test']:,}")

        st.subheader("Model comparison")
        st.caption("Chosen by cross-validation on the training set — never by the test set, "
                   "even when another model scores higher on it.")
        comp = pd.DataFrame([{
            "model": n.replace("_", " ") + (" ✓" if n == best else ""),
            "CV ROC-AUC (train)": r["cv_roc_auc_mean"], "± std": r["cv_roc_auc_std"],
            "test ROC-AUC": r["test"]["roc_auc"], "test PR-AUC": r["test"]["pr_auc"],
            "precision": r["test"]["precision"], "recall": r["test"]["recall"], "F1": r["test"]["f1"],
        } for n, r in metrics["models"].items()])
        st.dataframe(comp, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in comp.columns if c != "model"})

        left, right = st.columns(2)
        with left:
            st.subheader("What drives the predictions")
            st.caption("Permutation importance: drop in test ROC-AUC when the feature is shuffled. "
                       "Shows how much a feature matters, not in which direction.")
            imp = pd.DataFrame(list(metrics["top_features_permutation_importance"].items()),
                               columns=["feature", "importance"])
            st.altair_chart(hbar(imp, "importance", "feature", "Drop in ROC-AUC", ".4f"),
                            width="stretch")
        with right:
            st.subheader("Fairness by group")
            st.caption("Gender and MaritalStatus are not model features — yet recall still "
                       "differs between groups. Excluding a column does not guarantee fairness.")
            fair = metrics.get("fairness_by_group", {})
            rows = [{"attribute": attr, "group": g, "customers": v["customers"],
                     "actual churn": v["actual_churn_rate"], "flagged": v["flagged_rate"],
                     "recall": v["recall"]}
                    for attr, groups in fair.items() for g, v in groups.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent")
                                        for c in ("actual churn", "flagged", "recall")})

        eval_tab, card_tab = st.tabs(["Evaluation report", "Model card"])
        with eval_tab:
            st.markdown(read_doc(paths.EVAL_REPORT))
        with card_tab:
            st.markdown(read_doc(paths.MODEL_CARD))

# ── Score a customer ──────────────────────────────────────────────────────────
with tab_score:
    bundle = load_bundle()
    if bundle is None:
        st.info("No model yet — run the Flow first.")
    else:
        st.caption("Every field comes from the dataset contract. Values are checked against it "
                   "before the model sees them — try a cashback of 30,000 and watch it refuse.")
        fields = input_fields(contract, load_clean())
        with st.form("score"):
            values = {}
            grid = st.columns(3)
            for i, f in enumerate(fields):
                label = f["name"] + (f" ({f['unit']})" if f["unit"] else "")
                help_text = f["description"] + ("" if f["used_by_model"]
                                                 else " — not used by the model (fairness audit only)")
                with grid[i % 3]:
                    if f["kind"] == "choice":
                        opts = f["options"]
                        values[f["name"]] = st.selectbox(label, opts, index=opts.index(f["default"]),
                                                         help=help_text)
                    else:
                        default = int(round(f["default"])) if f["integer"] else f["default"]
                        step = 1 if f["integer"] else 0.01
                        values[f["name"]] = st.number_input(label, value=default, step=step,
                                                            help=help_text)
            submitted = st.form_submit_button("Score customer", type="primary")
        if submitted:
            report, proba = score_customer(values, contract, bundle)
            if proba is None:
                st.error("**Refused** — this customer breaks the dataset contract, so it is not scored.",
                         icon="⛔")
                for e in report.errors:
                    st.markdown(f"- `{e}`")
            else:
                at_risk = proba >= bundle["threshold"]
                (st.warning if at_risk else st.success)(
                    f"Churn probability **{proba:.1%}** — "
                    + ("**at risk**: prioritise for retention outreach." if at_risk
                       else "below the outreach threshold."),
                    icon="⚠️" if at_risk else "✅")
                st.caption(f"Model: {bundle['model_name'].replace('_', ' ')} · threshold "
                           f"{bundle['threshold']:.0%} · contract v{bundle['contract_version']}")

# ── Break it ──────────────────────────────────────────────────────────────────
with tab_break:
    st.caption("Break Crew 1's handoff in memory and run the Flow's gates on it. Nothing on "
               "disk changes. To break the real files and run the full Flow, use "
               "`python scripts/break_handoff.py <scenario>` then `python -m src.flow --skip-crew1`.")
    name = st.radio("Scenario", list(SCENARIOS), horizontal=True,
                    format_func=lambda n: n.replace("_", " "))
    scenario = SCENARIOS[name]
    st.markdown(f"_{scenario.description}_")

    df_broken, contract_broken = scenario.apply(load_clean(), contract)
    reports = run_gates(df_broken, contract_broken)
    labels = ["Gate 1 — clean data matches contract", "Gate 2 — contract provides every required feature"]
    for report, label in zip(reports, labels):
        show_report(report, label)
    if all(r.passed for r in reports) and len(reports) == 2:
        st.success("Both gates passed — Crew 2 would start.", icon="✅")
    else:
        st.error("**Flow stops here.** Crew 2 never starts and no model is trained on this data.",
                 icon="⛔")
