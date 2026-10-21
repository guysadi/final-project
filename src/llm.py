"""Shared LLM configuration for every agent."""
import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> LLM:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    # temperature=0 keeps the agents' wording as stable as an LLM allows; everything
    # that must be exactly reproducible (cleaning, schema, metrics) is done in code.
    return LLM(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0, seed=42)
