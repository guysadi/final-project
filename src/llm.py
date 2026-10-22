"""Shared LLM configuration for every agent.

Two tiers:
- fast: agents whose job is to call tools and summarise their output.
- reasoning: agents that interpret statistics. In real runs gpt-4o-mini reversed the
  direction of findings (e.g. read "fewer days since last order" as less engaged) and a
  gpt-4o-mini reviewer made it worse, so interpretation gets a reasoning model.
"""
import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def get_llm(reasoning: bool = False) -> LLM:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    if reasoning:
        # Reasoning models don't accept temperature/seed.
        return LLM(model=os.getenv("OPENAI_REASONING_MODEL", "gpt-5.4-mini"), reasoning_effort="low")
    # temperature=0 keeps the agents' wording as stable as an LLM allows; everything
    # that must be exactly reproducible (cleaning, schema, metrics) is done in code.
    return LLM(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0, seed=42)
