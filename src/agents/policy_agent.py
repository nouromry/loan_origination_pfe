# src/agents/policy_agent.py

from .base_agent import BaseAgent
from typing import List
from langchain_core.tools import BaseTool
from src.tools.rag_tools import query_policy


class PolicyAgent(BaseAgent):
    """
    Answers policy questions using ChromaDB RAG.
    Never invents policy values. Always cites retrieved text.
    Uses the fast 8B model.
    """

    def get_tools(self) -> List[BaseTool]:
        return [
            query_policy,
        ]

    def get_prompt_key(self) -> str:
        return "policy_agent"
