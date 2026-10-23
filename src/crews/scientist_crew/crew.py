"""Crew 2 — Data Scientist Crew: from the dataset contract to a model."""
import json

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from src import paths
from src.crews.analyst_crew.crew import _strip_code_fence
from src.crews.scientist_crew.tools import (
    EngineerFeaturesTool,
    ReadModelSummaryTool,
    TrainModelsTool,
    ValidateHandoffTool,
)
from src.crews.scientist_crew.training import metrics_table
from src.llm import get_llm
from src.validation.contract import run_gate


@CrewBase
class ScientistCrew:
    """Validates the handoff, engineers features, trains and compares models, writes the model card."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def feature_engineer(self) -> Agent:
        return Agent(config=self.agents_config["feature_engineer"],  # type: ignore[index]
                     tools=[ValidateHandoffTool(), EngineerFeaturesTool()], llm=get_llm(), verbose=True)

    @agent
    def ml_engineer(self) -> Agent:
        return Agent(config=self.agents_config["ml_engineer"],  # type: ignore[index]
                     tools=[TrainModelsTool()], llm=get_llm(reasoning=True), verbose=True)

    @agent
    def model_card_author(self) -> Agent:
        return Agent(config=self.agents_config["model_card_author"],  # type: ignore[index]
                     tools=[ReadModelSummaryTool()], llm=get_llm(reasoning=True), verbose=True)

    @task
    def validate_and_engineer(self) -> Task:
        return Task(config=self.tasks_config["validate_and_engineer"])  # type: ignore[index]

    @task
    def train_models(self) -> Task:
        return Task(config=self.tasks_config["train_models"])  # type: ignore[index]

    @task
    def write_model_card(self) -> Task:
        return Task(config=self.tasks_config["write_model_card"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks,
                    process=Process.sequential, verbose=True)


def _with_generated_metrics(doc: str, metrics: dict) -> str:
    """Append the comparison table straight from metrics.json, so the numbers in the
    document never depend on an LLM copying them correctly."""
    return (_strip_code_fence(doc).strip()
            + "\n\n## Appendix — metrics (generated from metrics.json)\n\n"
            + metrics_table(metrics) + "\n")


def run_scientist_crew() -> list:
    """Run Crew 2 from scratch and return the paths of its four artifacts.

    The contract gate runs before any agent starts: if Crew 1's handoff is broken,
    this raises ContractViolation and no tokens are spent.
    """
    for p in paths.CREW2_ARTIFACTS + [paths.FEATURE_SPEC, paths.METRICS]:
        p.unlink(missing_ok=True)
    run_gate(paths.CLEAN_DATA, paths.CONTRACT)

    result = ScientistCrew().crew().kickoff()
    outputs = {t.name: t.raw for t in result.tasks_output}
    if not paths.METRICS.exists():
        raise RuntimeError("Data Scientist crew finished without training a model "
                           f"(feature engineer reported: {outputs.get('validate_and_engineer', '')[:500]})")
    metrics = json.loads(paths.METRICS.read_text())
    paths.EVAL_REPORT.write_text(_with_generated_metrics(outputs["train_models"], metrics))
    paths.MODEL_CARD.write_text(_with_generated_metrics(outputs["write_model_card"], metrics))
    print(f"Token usage: {result.token_usage}")

    missing = [p.name for p in paths.CREW2_ARTIFACTS if not p.exists()]
    if missing:
        raise RuntimeError(f"Data Scientist crew finished without producing: {missing}")
    return paths.CREW2_ARTIFACTS


if __name__ == "__main__":
    for p in run_scientist_crew():
        print(f"✔ {p.relative_to(paths.ROOT)}")
