"""The Flow: Crew 1 → contract gate → required-features gate → Crew 2.

This is the piece Harbor & Vale never had. Each gate is plain code. If one fails the
Flow stops, says exactly why, writes artifacts/run_summary.json and exits non-zero —
Crew 2 never starts, so no model is trained on data that broke the contract.

    python -m src.flow                # full run: Crew 1, gates, Crew 2
    python -m src.flow --skip-crew1   # reuse Crew 1's committed artifacts (for break-it demos)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Literal

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from src import paths
from src.crews.analyst_crew.crew import run_analyst_crew
from src.crews.scientist_crew.crew import run_scientist_crew
from src.crews.scientist_crew.features import validate_feature_requirements
from src.validation.contract import ContractViolation, load_contract, run_gate

log = logging.getLogger("flow")


class StageResult(BaseModel):
    name: str
    status: Literal["passed", "failed", "skipped"]
    seconds: float = 0.0
    checks_run: int | None = None
    errors: list[str] = Field(default_factory=list)


class PipelineState(BaseModel):
    run_id: str = ""
    skip_crew1: bool = False
    stages: list[StageResult] = Field(default_factory=list)
    status: Literal["running", "succeeded", "failed"] = "running"
    failed_stage: str | None = None

    def record(self, result: StageResult) -> None:
        self.stages.append(result)
        if result.status == "failed":
            self.status, self.failed_stage = "failed", result.name


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _errors_of(exc: Exception) -> list[str]:
    if isinstance(exc, ContractViolation):
        return exc.errors
    return [f"{type(exc).__name__}: {exc}"]


class ChurnPipelineFlow(Flow[PipelineState]):

    def _stage(self, name: str, fn) -> StageResult:
        """Run one stage, time it, log it, and turn any exception into a recorded failure."""
        log.info("━━ %s", name)
        t0 = time.perf_counter()
        try:
            checks = fn()
            result = StageResult(name=name, status="passed", checks_run=checks)
        except Exception as exc:  # noqa: BLE001 — the Flow must stop cleanly, whatever broke
            result = StageResult(name=name, status="failed", errors=_errors_of(exc))
        result.seconds = round(time.perf_counter() - t0, 2)
        if result.status == "passed":
            extra = f", {result.checks_run} checks" if result.checks_run else ""
            log.info("   ✔ passed (%.1fs%s)", result.seconds, extra)
        else:
            log.error("   ✘ FAILED after %.1fs", result.seconds)
            for e in result.errors:
                log.error("     - %s", e)
        self.state.record(result)
        return result

    @start()
    def analyst_crew(self) -> None:
        if self.state.skip_crew1:
            def reuse():
                if missing := [p.name for p in paths.CREW1_ARTIFACTS if not p.exists()]:
                    raise RuntimeError(f"--skip-crew1 but Crew 1 artifacts are missing: {missing}")
            self._stage("Crew 1 — Data Analyst (reusing committed artifacts)", reuse)
        else:
            self._stage("Crew 1 — Data Analyst", lambda: run_analyst_crew() and None)

    @router(analyst_crew)
    def contract_gate(self) -> Literal["contract_ok", "failed"]:
        if self.state.status == "failed":
            return "failed"
        r = self._stage("Gate 1 — clean data matches dataset contract",
                        lambda: run_gate(paths.CLEAN_DATA, paths.CONTRACT).checks_run)
        return "contract_ok" if r.status == "passed" else "failed"

    @router("contract_ok")
    def features_gate(self) -> Literal["features_ok", "failed"]:
        def check():
            report = validate_feature_requirements(load_contract(paths.CONTRACT))
            report.raise_if_failed()
            return report.checks_run
        r = self._stage("Gate 2 — contract provides every required feature", check)
        return "features_ok" if r.status == "passed" else "failed"

    @router("features_ok")
    def scientist_crew(self) -> Literal["succeeded", "failed"]:
        r = self._stage("Crew 2 — Data Scientist", lambda: run_scientist_crew() and None)
        return "succeeded" if r.status == "passed" else "failed"

    @listen("succeeded")
    def on_success(self) -> None:
        self.state.status = "succeeded"
        self._write_summary()
        log.info("━━ Flow SUCCEEDED — all 8 artifacts written. Summary: %s",
                 paths.RUN_SUMMARY.relative_to(paths.ROOT))

    @listen("failed")
    def on_failure(self) -> None:
        self._write_summary()
        log.error("━━ Flow STOPPED at: %s", self.state.failed_stage)
        log.error("   Nothing downstream of that stage was run. No model was trained on this data.")
        log.error("   Summary: %s", paths.RUN_SUMMARY.relative_to(paths.ROOT))

    def _write_summary(self) -> None:
        done = [p for p in paths.CREW1_ARTIFACTS + paths.CREW2_ARTIFACTS if p.exists()]
        summary = {
            "run_id": self.state.run_id,
            "status": self.state.status,
            "failed_stage": self.state.failed_stage,
            "skip_crew1": self.state.skip_crew1,
            "stages": [s.model_dump() for s in self.state.stages],
            # Hashes make reproducibility checkable: deterministic artifacts (clean data,
            # features, metrics) hash identically across runs; LLM-written prose does not.
            "artifacts": {str(p.relative_to(paths.ROOT)): _sha256(p) for p in done},
        }
        paths.RUN_SUMMARY.write_text(json.dumps(summary, indent=2))


def _setup_logging(run_id: str) -> None:
    paths.LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(paths.LOGS_DIR / f"flow_{run_id}.log", encoding="utf-8")
    console = logging.StreamHandler(sys.stderr)
    for h in (file_handler, console):
        h.setFormatter(fmt)
    log.handlers[:] = [file_handler, console]
    log.setLevel(logging.INFO)
    log.propagate = False


def run_flow(skip_crew1: bool = False) -> PipelineState:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _setup_logging(run_id)
    log.info("Harbor & Vale churn pipeline — run %s%s", run_id, " (skip Crew 1)" if skip_crew1 else "")
    flow = ChurnPipelineFlow()
    flow.kickoff(inputs={"run_id": run_id, "skip_crew1": skip_crew1})
    return flow.state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-crew1", action="store_true",
                        help="reuse Crew 1's existing artifacts instead of re-running it")
    args = parser.parse_args()
    state = run_flow(skip_crew1=args.skip_crew1)
    return 0 if state.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
