"""Tools the Data Analyst agents can call.

Paths are fixed inside each tool: an agent can only read the raw data and write
Crew 1's artifacts — it cannot be talked into touching anything else.
"""
from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src import paths
from src.crews.analyst_crew.cleaning import load_data_dictionary, load_raw, profile, run_cleaning
from src.crews.analyst_crew.contract_builder import write_contract
from src.crews.analyst_crew.eda import run_eda
from src.validation.contract import ContractViolation


class ProfileRawDataTool(BaseTool):
    name: str = "profile_raw_data"
    description: str = (
        "Profile the raw dataset: row/column counts, missing values per column, duplicate "
        "rows and the distinct values of every text column. Takes no input.")

    def _run(self) -> str:
        raw = load_raw(paths.RAW_DATA)
        result = profile(raw)
        result["dtypes"] = {c: str(t) for c, t in raw.dtypes.items()}
        result["data_dictionary"] = load_data_dictionary(paths.RAW_DATA)
        return json.dumps(result, indent=2, default=str)


class CleanDataTool(BaseTool):
    name: str = "clean_data"
    description: str = (
        "Run the deterministic cleaning pipeline on the raw dataset and save clean_data.csv: "
        "drops duplicate customers, merges duplicate category spellings, fills missing numeric "
        "values with the median and casts count columns to integers. Returns the cleaning log "
        "and a before/after profile. Takes no input.")

    def _run(self) -> str:
        summary = run_cleaning(paths.RAW_DATA, paths.CLEAN_DATA)
        paths.CLEANING_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
        return json.dumps(summary, indent=2, default=str)


class RunEdaTool(BaseTool):
    name: str = "run_eda"
    description: str = (
        "Run exploratory data analysis on clean_data.csv. Renders eda_report.html with charts "
        "and returns the computed statistics as JSON: churn rate, correlations with churn, "
        "means for churned vs. retained customers, churn rate by segment and by tenure. "
        "Takes no input. clean_data must have been run first.")

    def _run(self) -> str:
        if not paths.CLEAN_DATA.exists():
            return "ERROR: clean_data.csv does not exist yet. Run clean_data first."
        cleaning = (json.loads(paths.CLEANING_SUMMARY.read_text())
                    if paths.CLEANING_SUMMARY.exists() else None)
        stats = run_eda(paths.CLEAN_DATA, paths.EDA_REPORT, paths.EDA_STATS, cleaning)
        return json.dumps(stats, indent=2, default=str)


class ReadEdaStatsTool(BaseTool):
    name: str = "read_eda_stats"
    description: str = (
        "Read the ground-truth statistics saved by run_eda: overall churn rate, the direction "
        "of every churned-vs-stayed difference, churn rate by segment and by tenure. "
        "Read-only. Takes no input.")

    def _run(self) -> str:
        if not paths.EDA_STATS.exists():
            return "ERROR: eda_stats.json does not exist yet. run_eda must run first."
        stats = json.loads(paths.EDA_STATS.read_text())
        keys = ("rows", "overall_churn_rate", "churned_customers", "churned_vs_stayed",
                "churn_rate_by_segment", "tenure_bucket_churn")
        return json.dumps({k: stats[k] for k in keys}, indent=2)


class ColumnUnit(BaseModel):
    column: str = Field(description="Column name, e.g. CashbackAmount")
    unit: str = Field(description="Unit of measure, e.g. USD, months, days, percent, count")


class WriteContractInput(BaseModel):
    # A list of pairs, not dict[str, str]: OpenAI strict function calling turns a free-form
    # dict into {"properties": {}, "additionalProperties": false}, so the model can only
    # ever send {} — which is exactly what it did in the first real runs.
    units: list[ColumnUnit] = Field(
        description="One entry per numeric column: its unit of measure.")
    assumptions: list[str] = Field(
        description="Plain-English statements the modeling crew may rely on about this data, "
                    "e.g. how missing values were handled or what one row represents.")


class WriteContractTool(BaseTool):
    name: str = "write_dataset_contract"
    description: str = (
        "Write dataset_contract.json. The schema (columns, dtypes, ranges, allowed values) is "
        "derived automatically from clean_data.csv and column descriptions come from the data "
        "dictionary; you supply the units and the assumptions. A unit is REQUIRED for every "
        "numeric column with a range — if any is missing the tool returns an ERROR listing "
        "them. The contract is validated against the clean data before it is saved.")
    args_schema: type[BaseModel] = WriteContractInput

    def _run(self, units: list, assumptions: list[str]) -> str:
        if not paths.CLEAN_DATA.exists():
            return "ERROR: clean_data.csv does not exist yet."
        # CrewAI may hand over pydantic models or plain dicts depending on the call path.
        unit_map = {}
        for u in units:
            u = u.model_dump() if isinstance(u, BaseModel) else u
            unit_map[u["column"]] = u["unit"]
        try:
            contract = write_contract(paths.CLEAN_DATA, paths.CONTRACT,
                                      descriptions=load_data_dictionary(paths.RAW_DATA),
                                      units=unit_map, assumptions=assumptions)
        except ContractViolation as e:
            return f"ERROR: contract not saved. Fix your input and call this tool again.\n{e}"
        return (f"Contract v{contract['version']} saved to {paths.CONTRACT.name} with "
                f"{len(contract['columns'])} columns, target '{contract['target']}', "
                f"{len(contract['assumptions'])} assumptions. Validated against clean_data.csv: OK.")
