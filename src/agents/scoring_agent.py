# src/agents/scoring_agent.py

from .base_agent import BaseAgent
from typing import List
from langchain_core.tools import BaseTool
from src.tools.financial_tools import calculate_dti
from src.tools.scoring_tools import (
    build_scoring_payload,
    call_scoring_api,
    parse_scoring_response,
)


class ScoringAgent(BaseAgent):
    """
    Calculates DTI and interacts with the ML scoring API.
    Runs for BOTH personal and business loans.
    Uses the fast 8B model.
    """

    def get_tools(self) -> List[BaseTool]:
        return [
            calculate_dti,
            build_scoring_payload,
            call_scoring_api,
            parse_scoring_response,
        ]

    def get_prompt_key(self) -> str:
        return "scoring_agent"
