# src/agents/collect_agent.py

"""
Collect Agent — Extracts ONE field per message.

HYBRID APPROACH (same pattern as RiskAssessment and Decision):
  - Regex tools handle obvious fields (national_id, email, phone, amounts)
  - LLM handles ambiguous fields (marital_status, housing_status, industry)
  - No LLM call needed for ~70% of fields

Tool-first extraction order:
  1. Keyword match (loan_type only)
  2. Regex tool match (national_id, email, phone, amounts, counts)
  3. LLM one-shot extraction (date_of_birth, marital_status, housing, industry)
"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent
from src.tools.extraction_tools import (
    extract_national_id,
    extract_email,
    extract_phone,
    extract_amount,
    extract_loan_term,
)


# Maps field name → which tool to try first
REGEX_TOOL_MAP = {
    "national_id": extract_national_id,
    "email": extract_email,
    "phone": extract_phone,
    "loan_amount": extract_amount,
    "loan_term_months": extract_loan_term,
}

# Fields with simple value matching (no tool, no LLM)
KEYWORD_FIELDS = {
    "loan_type": {
        "personal": ["personal", "personnel", "individuel", "شخصي"],
        "business": ["business", "entreprise", "professionnel", "commercial",
                     "restaurant", "shop", "company", "société", "magasin", "تجاري"],
    },
    "number_of_dependents": {
        0: ["0", "none", "no", "aucun", "zero", "zéro", "لا", "صفر"],
    },
    "marital_status": {
        "single": ["single", "célibataire", "celibataire", "أعزب"],
        "married": ["married", "marié", "mariée", "متزوج"],
        "divorced": ["divorced", "divorcé", "divorcée", "مطلق"],
        "widowed": ["widowed", "veuf", "veuve", "أرمل"],
    },
    "housing_status": {
        "owner": ["owner", "propriétaire", "proprietaire", "مالك"],
        "renter": ["renter", "locataire", "مستأجر"],
        "family": ["family", "famille", "familial", "عائلي", "parents"],
    },
}

# Fields that can be extracted with a simple digit regex
DIGIT_FIELDS = {
    "number_of_dependents": 20,       # max reasonable value
    "number_of_employees": 100000,
    "applicant_ownership_percentage": 100,
}


class CollectAgent(BaseAgent):
    """
    Extracts exactly ONE requested field from user messages.
    Tries regex tools first, falls back to LLM for ambiguous fields.
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

    # ------------------------------------------------------------------
    # Override BaseAgent.run() with regex-first pipeline
    # ------------------------------------------------------------------
    def run(self, goal: str, **context) -> Dict[str, Any]:
        """
        Extract one field from the user message.
        
        Pipeline:
          1. Keyword match (loan_type, marital_status, housing_status)
          2. Regex tool (national_id, email, phone, amounts)
          3. Digit regex (dependents, employees, ownership %)
          4. LLM one-shot (date_of_birth, industry, anything else)
        """
        target_field = context.get("target_field", "")
        message = context.get("message", "")
        last_question = context.get("last_question", "")

        if not message.strip():
            return {"field": target_field, "value": None, "method": "empty_message"}

        # Step 1: Keyword match
        value = self._try_keyword_match(target_field, message)
        if value is not None:
            return {"field": target_field, "value": value, "method": "keyword"}

        # Step 2: Regex tool
        value = self._try_regex_tool(target_field, message)
        if value is not None:
            return {"field": target_field, "value": value, "method": "regex_tool"}

        # Step 3: Simple digit extraction
        value = self._try_digit_extract(target_field, message)
        if value is not None:
            return {"field": target_field, "value": value, "method": "digit_regex"}

        # Step 4: LLM fallback
        value = self._try_llm_extract(target_field, message, last_question)
        if value is not None:
            return {"field": target_field, "value": value, "method": "llm"}

        return {"field": target_field, "value": None, "method": "failed"}

    # ------------------------------------------------------------------
    # Step 1: Keyword match
    # ------------------------------------------------------------------
    def _try_keyword_match(self, field: str, message: str) -> Optional[Any]:
        """Match field value from keywords. Returns value or None."""
        if field not in KEYWORD_FIELDS:
            return None

        msg_lower = message.lower().strip()
        for value, keywords in KEYWORD_FIELDS[field].items():
            if any(kw in msg_lower for kw in keywords):
                return value

        return None

    # ------------------------------------------------------------------
    # Step 2: Regex tool
    # ------------------------------------------------------------------
    def _try_regex_tool(self, field: str, message: str) -> Optional[Any]:
        """Use the matching extraction tool. Returns value or None."""
        tool_fn = REGEX_TOOL_MAP.get(field)
        if tool_fn is None:
            return None

        try:
            result = tool_fn.invoke({"message": message})
            return result if result is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Step 3: Simple digit extraction
    # ------------------------------------------------------------------
    def _try_digit_extract(self, field: str, message: str) -> Optional[Any]:
        """Extract a number for simple numeric fields."""
        max_val = DIGIT_FIELDS.get(field)
        if max_val is None:
            return None

        # Check for zero/none keywords first (for dependents)
        msg_lower = message.lower().strip()
        if field == "number_of_dependents":
            if msg_lower in ("0", "none", "no", "aucun", "zero", "zéro"):
                return 0

        # Extract first number
        m = re.search(r'(\d+(?:\.\d+)?)', message)
        if m:
            val = float(m.group(1))
            if field == "applicant_ownership_percentage":
                if 0 <= val <= 100:
                    return val
            else:
                val = int(val)
                if 0 <= val <= max_val:
                    return val

        return None

    # ------------------------------------------------------------------
    # Step 4: LLM one-shot extraction (last resort)
    # ------------------------------------------------------------------
    def _try_llm_extract(self, field: str, message: str, last_question: str) -> Optional[Any]:
        """Single LLM call to extract the field value."""
        system_directive = (
            f"You are a strict data extraction assistant.\n"
            f"The user was asked: '{last_question}'.\n"
            f"Extract ONLY the value for this field: '{field}'.\n\n"
            f"Respond ONLY with valid JSON.\n"
            f"Format: {{\"{field}\": value}}\n"
            f"If missing, return: {{\"{field}\": null}}"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_directive),
                HumanMessage(content=message),
            ])

            raw = response.content.strip()

            # Clean markdown
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            clean = match.group(1) if match else raw

            parsed = json.loads(clean)

            # Extract value
            if field in parsed:
                return parsed[field]
            if "value" in parsed:
                return parsed["value"]
            for k, v in parsed.items():
                if k not in ("raw_response", "error") and v is not None:
                    return v

        except Exception as e:
            print(f"[CollectAgent LLM] {e}")

        return None