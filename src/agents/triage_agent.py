# src/agents/triage_agent.py

from .base_agent import BaseAgent
from typing import List
from langchain_core.tools import BaseTool


class TriageAgent(BaseAgent):
    """
    Classifies intent and loan type on every single message.
    Uses the fast 8B model. No tools needed — pure LLM comprehension.
    """

    def get_tools(self) -> List[BaseTool]:
        return []

    def get_prompt_key(self) -> str:
        return "triage_agent"
