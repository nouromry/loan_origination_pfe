# src/agents/collect_agent.py

from .base_agent import BaseAgent
from typing import List
from langchain_core.tools import BaseTool
from src.tools.extraction_tools import (
    extract_national_id,
    extract_email,
    extract_phone,
    extract_amount,
    extract_loan_term,
)


class CollectAgent(BaseAgent):
    """
    Extracts exactly ONE requested field from user messages.
    Uses the fast 8B model with extraction tools for validation.
    """

    def get_tools(self) -> List[BaseTool]:
        return [
            extract_national_id,
            extract_email,
            extract_phone,
            extract_amount,
            extract_loan_term,
        ]

    def get_prompt_key(self) -> str:
        return "collect_agent"
