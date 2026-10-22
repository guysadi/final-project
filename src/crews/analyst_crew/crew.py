"""Crew 1 — Data Analyst Crew: from raw data to a dataset contract."""
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from src import paths
from src.crews.analyst_crew.tools import (
    CleanDataTool,
    ProfileRawDataTool,
    ReadEdaStatsTool,
    RunEdaTool,
    WriteContractTool,
)
from src.llm import get_llm


@CrewBase
class AnalystCrew:
    """Ingests, cleans and explores the data, then publishes the dataset contract."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def data_engineer(self) -> Agent:
        return Agent(config=self.agents_config["data_engineer"],  # type: ignore[index]
                     tools=[ProfileRawDataTool(), CleanDataTool()], llm=get_llm(), verbose=True)

    @agent
    def eda_analyst(self) -> Agent:
        return Agent(config=self.agents_config["eda_analyst"],  # type: ignore[index]
                     tools=[RunEdaTool()], llm=get_llm(reasoning=True), verbose=True)

    @agent
    def business_analyst(self) -> Agent:
        return Agent(config=self.agents_config["business_analyst"],  # type: ignore[index]
                     llm=get_llm(reasoning=True), verbose=True)

    @agent
    def insights_reviewer(self) -> Agent:
        return Agent(config=self.agents_config["insights_reviewer"],  # type: ignore[index]
                     tools=[ReadEdaStatsTool()], llm=get_llm(reasoning=True), verbose=True)

    @agent
    def contract_author(self) -> Agent:
        return Agent(config=self.agents_config["contract_author"],  # type: ignore[index]
                     tools=[WriteContractTool()], llm=get_llm(), verbose=True)

    @task
    def ingest_and_clean(self) -> Task:
        return Task(config=self.tasks_config["ingest_and_clean"])  # type: ignore[index]

    @task
    def explore_data(self) -> Task:
        return Task(config=self.tasks_config["explore_data"])  # type: ignore[index]

    @task
    def write_insights(self) -> Task:
        return Task(config=self.tasks_config["write_insights"])  # type: ignore[index]

    @task
    def review_insights(self) -> Task:
        # Saved by run_analyst_crew, not output_file: CrewAI strips the leading "/" from
        # absolute output_file paths and writes them relative to the current directory.
        return Task(config=self.tasks_config["review_insights"])  # type: ignore[index]

    @task
    def write_contract(self) -> Task:
        return Task(config=self.tasks_config["write_contract"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks,
                    process=Process.sequential, verbose=True)


def _strip_code_fence(text: str) -> str:
    """LLMs sometimes wrap markdown in ```markdown ... ``` despite being told not to."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines)


def run_analyst_crew() -> list:
    """Run Crew 1 from scratch and return the paths of its four artifacts.

    Old artifacts are deleted first, so a file left over from a previous run can never
    be mistaken for this run's output.
    """
    for p in paths.CREW1_ARTIFACTS + [paths.CLEANING_SUMMARY, paths.EDA_STATS]:
        p.unlink(missing_ok=True)
    result = AnalystCrew().crew().kickoff()
    insights = next(t.raw for t in result.tasks_output if t.name == "review_insights")
    paths.INSIGHTS.write_text(_strip_code_fence(insights).strip() + "\n")
    print(f"Token usage: {result.token_usage}")
    missing = [p.name for p in paths.CREW1_ARTIFACTS if not p.exists()]
    if missing:
        raise RuntimeError(f"Data Analyst crew finished without producing: {missing}")
    return paths.CREW1_ARTIFACTS


if __name__ == "__main__":
    for p in run_analyst_crew():
        print(f"✔ {p.relative_to(paths.ROOT)}")
