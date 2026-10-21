# Harbor & Vale — Contract-Enforced AI Data Workflow

A CrewAI Flow that connects two multi-agent crews — a **Data Analyst Crew** and a
**Data Scientist Crew** — through a machine-enforced **dataset contract**. If the
cleaned data and the contract disagree (missing column, out-of-range value,
changed unit), the Flow stops with a readable error instead of training a model
on bad data.

Dataset: [E-commerce Customer Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle).

## Project layout

```
data/raw/            raw dataset (downloaded, not committed)
artifacts/crew1/     clean_data.csv, eda_report.html, insights.md, dataset_contract.json
artifacts/crew2/     features.csv, model.joblib, evaluation_report.md, model_card.md
src/crews/           the two CrewAI crews
src/validation/      contract validation gate
src/flow.py          the Flow that runs everything
app/                 Streamlit demo app
logs/                run logs
```

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
streamlit run app/streamlit_app.py
```
