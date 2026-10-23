# Harbor & Vale — Contract-Enforced AI Data Workflow

A CrewAI Flow that connects two multi-agent crews — a **Data Analyst Crew** and a
**Data Scientist Crew** — through a machine-enforced **dataset contract**. If the
cleaned data and the contract disagree (missing column, out-of-range value,
changed unit), the Flow stops with a readable error instead of training a model
on bad data.

Dataset: [E-commerce Customer Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle).

## How it works

```
raw data ─▶ Crew 1: Data Analyst ─▶ Gate 1 ─▶ Gate 2 ─▶ Crew 2: Data Scientist ─▶ model
            (5 agents)               │          │        (3 agents)
                                     ▼          ▼
                           clean data matches   contract provides every
                           the contract?        feature the model needs?
                                     └── no ──▶ STOP: readable error, run_summary.json, exit 1
```

| Stage | What it produces |
|---|---|
| **Crew 1** — Data Engineer, EDA Analyst, Business Analyst, Insights Reviewer, Contract Author | `clean_data.csv`, `eda_report.html`, `insights.md`, `dataset_contract.json` |
| **Gate 1** — 71 checks: columns, dtypes, nulls, ranges, allowed values, primary key | pass / stop |
| **Gate 2** — the contract provides every column feature engineering needs | pass / stop |
| **Crew 2** — Feature Engineer, ML Engineer, Model Card Author | `features.csv`, `model.joblib`, `evaluation_report.md`, `model_card.md` |

**Design rule: code decides, agents explain.** Cleaning, the contract schema, the gates,
features, training and metrics are deterministic Python exposed to the agents as tools.
The agents call the tools and write the human-facing parts (insights, units and
assumptions, evaluation report, model card). Rerunning the Flow reproduces every
deterministic artifact byte for byte — `artifacts/run_summary.json` records their hashes.

**The boundary.** Crew 2 reads only `clean_data.csv` and `dataset_contract.json` — never
the raw data (enforced by a test). Its features come only from what the contract describes.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
python scripts/download_data.py
```

## Run

```bash
python -m src.flow
```

A full run takes about 80 seconds and ~40k tokens. Logs go to the console and to
`logs/flow_<run id>.log`; the outcome of every stage is saved to `artifacts/run_summary.json`.

## Break it on purpose

```bash
python scripts/break_handoff.py unit_change   # CashbackAmount: dollars → cents
python -m src.flow --skip-crew1               # reuse Crew 1's artifacts
```

```
━━ Gate 1 — clean data matches dataset contract
   ✘ FAILED after 0.0s
     - Column 'CashbackAmount': 5070 value(s) above contract max 487.49 USD (highest: 32499.0) — possible unit change?
━━ Flow STOPPED at: Gate 1 — clean data matches dataset contract
   Nothing downstream of that stage was run. No model was trained on this data.
```

Other scenarios: `rename_column`, `drop_contract_column`, `bad_category`,
`remove_feature_source` (passes Gate 1, stopped by Gate 2). Undo with
`python scripts/break_handoff.py restore`.

## Tests

```bash
python -m pytest
```

55 tests, no API key needed: the contract gate, both crews' tools and wiring, every
break scenario, and the Flow's routing.

## Project layout

```
data/raw/            raw dataset (downloaded, not committed)
artifacts/crew1/     Crew 1 outputs + supporting stats
artifacts/crew2/     Crew 2 outputs + metrics.json, feature_spec.json
artifacts/run_summary.json   outcome of the latest Flow run
src/validation/      the contract gate
src/crews/           the two crews (agents.yaml, tasks.yaml, tools, deterministic code)
src/flow.py          the Flow
scripts/             data download, break-it scenarios
tests/               pytest suite
```
