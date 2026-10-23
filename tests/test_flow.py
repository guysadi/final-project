"""Tests for the Flow's routing: which stages run, where it stops, and what it reports.
The crews are replaced with stand-ins, so no LLM is called."""
import importlib
import json

import pytest

from src import flow as flow_module
from src import paths
from src.crews.analyst_crew.contract_builder import build_contract
from tests.test_scientist_crew import make_clean


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Valid Crew 1 artifacts in a temp folder, and crews that record whether they ran."""
    for name in ("CLEAN_DATA", "EDA_REPORT", "INSIGHTS", "CONTRACT", "FEATURES", "MODEL",
                 "EVAL_REPORT", "MODEL_CARD", "RUN_SUMMARY"):
        monkeypatch.setattr(paths, name, tmp_path / getattr(paths, name).name)
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "CREW1_ARTIFACTS",
                        [paths.CLEAN_DATA, paths.EDA_REPORT, paths.INSIGHTS, paths.CONTRACT])
    monkeypatch.setattr(paths, "CREW2_ARTIFACTS",
                        [paths.FEATURES, paths.MODEL, paths.EVAL_REPORT, paths.MODEL_CARD])

    clean = make_clean()
    clean.to_csv(paths.CLEAN_DATA, index=False)
    paths.CONTRACT.write_text(json.dumps(build_contract(clean)))
    paths.EDA_REPORT.write_text("<html></html>")
    paths.INSIGHTS.write_text("# insights")

    calls = []

    def fake_crew1():
        calls.append("crew1")
        return paths.CREW1_ARTIFACTS

    def fake_crew2():
        calls.append("crew2")
        for p in paths.CREW2_ARTIFACTS:
            p.write_text("x")
        return paths.CREW2_ARTIFACTS

    monkeypatch.setattr(flow_module, "run_analyst_crew", fake_crew1)
    monkeypatch.setattr(flow_module, "run_scientist_crew", fake_crew2)
    return {"clean": clean, "calls": calls}


def summary():
    return json.loads(paths.RUN_SUMMARY.read_text())


def stage_statuses(state):
    return [(s.name.split(" —")[0], s.status) for s in state.stages]


def test_happy_path_runs_every_stage(env):
    state = flow_module.run_flow()
    assert state.status == "succeeded"
    assert env["calls"] == ["crew1", "crew2"]
    assert stage_statuses(state) == [("Crew 1", "passed"), ("Gate 1", "passed"),
                                     ("Gate 2", "passed"), ("Crew 2", "passed")]
    s = summary()
    assert s["status"] == "succeeded" and len(s["artifacts"]) == 8


def test_unit_change_stops_before_crew2(env):
    """The Harbor & Vale incident, end to end."""
    broken = env["clean"].assign(CashbackAmount=env["clean"]["CashbackAmount"] * 100)
    broken.to_csv(paths.CLEAN_DATA, index=False)

    state = flow_module.run_flow(skip_crew1=True)
    assert state.status == "failed"
    assert state.failed_stage.startswith("Gate 1")
    assert "crew2" not in env["calls"]
    assert any("CashbackAmount" in e and "unit change" in e for e in state.stages[-1].errors)
    s = summary()
    assert s["status"] == "failed" and s["failed_stage"].startswith("Gate 1")


def test_missing_feature_source_stops_at_gate_2(env):
    contract = json.loads(paths.CONTRACT.read_text())
    contract["columns"].pop("OrderCount")
    paths.CONTRACT.write_text(json.dumps(contract))
    env["clean"].drop(columns="OrderCount").to_csv(paths.CLEAN_DATA, index=False)

    state = flow_module.run_flow(skip_crew1=True)
    assert stage_statuses(state)[1:] == [("Gate 1", "passed"), ("Gate 2", "failed")]
    assert "crew2" not in env["calls"]


def test_crew1_crash_stops_the_flow_cleanly(env, monkeypatch):
    def broken_crew1():
        raise RuntimeError("Data Analyst crew finished without producing: ['insights.md']")
    monkeypatch.setattr(flow_module, "run_analyst_crew", broken_crew1)

    state = flow_module.run_flow()
    assert state.failed_stage.startswith("Crew 1")
    assert [s.name.split(" —")[0] for s in state.stages] == ["Crew 1"]  # no gate, no Crew 2
    assert "insights.md" in state.stages[0].errors[0]


def test_crew2_failure_is_reported_not_raised(env, monkeypatch):
    def broken_crew2():
        raise RuntimeError("OpenAI quota exceeded")
    monkeypatch.setattr(flow_module, "run_scientist_crew", broken_crew2)

    state = flow_module.run_flow(skip_crew1=True)
    assert state.failed_stage.startswith("Crew 2")
    assert summary()["status"] == "failed"


def test_skip_crew1_requires_its_artifacts(env):
    paths.INSIGHTS.unlink()
    state = flow_module.run_flow(skip_crew1=True)
    assert state.failed_stage.startswith("Crew 1")
    assert "insights.md" in state.stages[0].errors[0]


def test_flow_logs_to_a_file(env):
    flow_module.run_flow(skip_crew1=True)
    [logfile] = list((paths.LOGS_DIR).glob("flow_*.log"))
    text = logfile.read_text()
    assert "Gate 1" in text and "SUCCEEDED" in text


@pytest.fixture
def break_handoff(env, monkeypatch):
    module = importlib.import_module("scripts.break_handoff")
    monkeypatch.setattr(module, "BACKUP_DIR", paths.ROOT / ".break_backup")
    monkeypatch.setattr(module, "TARGETS", [paths.CLEAN_DATA, paths.CONTRACT])
    return module


@pytest.mark.parametrize("scenario, stopped_at", [
    ("unit_change", "Gate 1"),
    ("rename_column", "Gate 1"),
    ("drop_contract_column", "Gate 1"),
    ("bad_category", "Gate 1"),
    ("remove_feature_source", "Gate 2"),
])
def test_every_break_scenario_stops_the_flow(env, break_handoff, scenario, stopped_at):
    break_handoff._backup()
    break_handoff.SCENARIOS[scenario]()
    state = flow_module.run_flow(skip_crew1=True)
    assert state.failed_stage.startswith(stopped_at)
    assert "crew2" not in env["calls"]


def test_restore_undoes_a_break(env, break_handoff):
    original = paths.CLEAN_DATA.read_text(), paths.CONTRACT.read_text()
    break_handoff._backup()
    break_handoff.SCENARIOS["unit_change"]()
    break_handoff.restore()
    assert (paths.CLEAN_DATA.read_text(), paths.CONTRACT.read_text()) == original
    assert flow_module.run_flow(skip_crew1=True).status == "succeeded"
