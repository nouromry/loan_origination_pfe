# src/agents/collect_agent.py

"""
Collect Agent — Multi-field extraction.

Tries to extract ALL missing fields from a single user message.
The user can dump everything at once ("my id is X, email Y, phone Z...")
and the agent will capture whatever it can. Fields that couldn't be
extracted will be asked about one by one in subsequent messages.

4-step pipeline per field:
  1. Keyword match (loan_type, marital_status, housing_status, etc.)
  2. Regex tool (national_id, email, phone, amounts)
  3. Digit regex (dependents, employees, ownership %)
  4. LLM one-shot (last resort, only for the target field)
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


# Maps field name → regex extraction tool
REGEX_TOOL_MAP = {
    "national_id": extract_national_id,
    "email": extract_email,
    "phone": extract_phone,
    "loan_amount": extract_amount,
    "loan_term_months": extract_loan_term,
}

# Fields extracted by keyword matching (no tool, no LLM)
KEYWORD_FIELDS = {
    "loan_type": {
        "personal": ["personal", "personnel", "individuel", "شخصي"],
        "business": ["business", "entreprise", "professionnel", "commercial",
                     "restaurant", "shop", "company", "société", "magasin", "تجاري"],
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

# Simple numeric fields — contextual extraction needed
DIGIT_FIELDS = {
    "number_of_dependents": 20,
    "number_of_employees": 100000,
    "applicant_ownership_percentage": 100,
}


class CollectAgent(BaseAgent):
    """
    Multi-field extraction agent.
    Given a user message + list of missing fields, extracts as many
    as possible in a single pass.
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
    # Multi-field extraction (new)
    # ------------------------------------------------------------------
    def run_multi(self, missing_fields: List[str], message: str,
                  last_question: str = "") -> Dict[str, Any]:
        """
        Extract as many fields as possible from a single message.
        
        Runs extractions in a careful order to avoid collisions:
        national_id and phone are both 8 digits, so we extract national_id
        FIRST, strip it from the message, then extract phone.
        
        Returns:
            {
                "extracted": {field_name: value, ...},
                "methods": {field_name: "keyword|regex|digit|llm", ...},
                "failed": [field1, field2, ...],
            }
        """
        if not message.strip() or not missing_fields:
            return {"extracted": {}, "methods": {}, "failed": missing_fields}

        extracted = {}
        methods = {}
        failed = []

        # The CURRENT target (what we just asked) deserves LLM fallback.
        target_field = missing_fields[0] if missing_fields else None

        # Work on a mutable copy of the message — we'll strip matched
        # values so later extractions don't re-match them
        working_message = message

        # Smart extraction order: extract fields that could collide FIRST,
        # then strip the matched value from the working message
        ordered_fields = self._order_fields_for_extraction(missing_fields)

        for field in ordered_fields:
            # Step 1: Keyword match (uses original message for keyword context)
            value = self._try_keyword_match(field, message)
            if value is not None:
                extracted[field] = value
                methods[field] = "keyword"
                continue

            # Step 2: Regex tool (uses working_message to avoid collisions)
            value = self._try_regex_tool(field, working_message)
            if value is not None:
                extracted[field] = value
                methods[field] = "regex_tool"
                # Strip the matched value so later fields don't re-match
                working_message = self._strip_value(working_message, field, value)
                continue

            # Step 3: Digit regex (with context checks)
            value = self._try_digit_extract(field, message, target_field)
            if value is not None:
                extracted[field] = value
                methods[field] = "digit_regex"
                continue

            # Step 4: LLM fallback — only for the TARGET field
            if field == target_field:
                value = self._try_llm_extract(field, message, last_question)
                if value is not None:
                    extracted[field] = value
                    methods[field] = "llm"
                    continue

            failed.append(field)

        return {
            "extracted": extracted,
            "methods": methods,
            "failed": failed,
        }

    def _order_fields_for_extraction(self, fields: List[str]) -> List[str]:
        """
        Order fields so that colliding regex patterns don't interfere.
        Priority:
          1. national_id (8 digits, easy to confuse with phone)
          2. email (unique pattern, no collision)
          3. phone (8 digits, must come AFTER national_id)
          4. loan_amount (could be confused with phone if 8-digit)
          5. loan_term_months
          6. everything else
        """
        priority = {
            "national_id": 1,
            "email": 2,
            "phone": 3,
            "loan_amount": 4,
            "loan_term_months": 5,
        }
        return sorted(fields, key=lambda f: priority.get(f, 99))

    def _strip_value(self, message: str, field: str, value: Any) -> str:
        """
        Remove a matched value from the message so subsequent extractions
        don't re-match it. Critical for avoiding phone ↔ national_id collisions.
        """
        if value is None:
            return message

        val_str = str(value)

        if field == "national_id":
            # Remove the exact 8-digit ID
            return re.sub(rf'\b{re.escape(val_str)}\b', '', message)

        if field == "email":
            return message.replace(val_str, '')

        if field == "phone":
            # Phone might have been normalized (8 digits) but the original
            # could have +216 prefix — try both
            message = message.replace(val_str, '')
            message = re.sub(rf'\+?216\s*{re.escape(val_str)}', '', message)
            return message

        if field == "loan_amount":
            # Value might be 50000.0 but message has "50000" or "50K"
            val_int = int(value) if value == int(value) else value
            message = re.sub(rf'\b{re.escape(str(val_int))}\s*(?:K|TND|DT|\$|€)?', '', message, flags=re.IGNORECASE)
            return message

        if field == "loan_term_months":
            val_int = int(value)
            message = re.sub(rf'\b{val_int}\s*months?', '', message, flags=re.IGNORECASE)
            message = re.sub(rf'\b{val_int // 12}\s*years?', '', message, flags=re.IGNORECASE)
            return message

        return message

    # ------------------------------------------------------------------
    # Single-field extraction (kept for backward compatibility)
    # ------------------------------------------------------------------
    def run(self, goal: str, **context) -> Dict[str, Any]:
        target_field = context.get("target_field", "")
        message = context.get("message", "")
        last_question = context.get("last_question", "")

        result = self.run_multi(
            missing_fields=[target_field],
            message=message,
            last_question=last_question,
        )
        value = result["extracted"].get(target_field)
        method = result["methods"].get(target_field, "failed")
        return {"field": target_field, "value": value, "method": method}

    # ------------------------------------------------------------------
    # Step 1: Keyword match
    # ------------------------------------------------------------------
    def _try_keyword_match(self, field: str, message: str) -> Optional[Any]:
        if field not in KEYWORD_FIELDS:
            # Special case: number_of_dependents with "none" / "aucun" / "0"
            if field == "number_of_dependents":
                msg_lower = message.lower().strip()
                # Only match if message is SHORT (otherwise "no" matches everywhere)
                if msg_lower in ("0", "none", "no", "aucun", "zero", "zéro"):
                    return 0
            return None

        msg_lower = message.lower()
        for value, keywords in KEYWORD_FIELDS[field].items():
            # Use word boundaries for keyword matching to avoid false positives
            for kw in keywords:
                # Match whole word or exact phrase
                if re.search(rf'\b{re.escape(kw)}\b', msg_lower):
                    return value
        return None

    # ------------------------------------------------------------------
    # Step 2: Regex tool
    # ------------------------------------------------------------------
    def _try_regex_tool(self, field: str, message: str) -> Optional[Any]:
        tool_fn = REGEX_TOOL_MAP.get(field)
        if tool_fn is None:
            return None
        try:
            result = tool_fn.invoke({"message": message})
            return result if result is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Step 3: Digit regex
    # ------------------------------------------------------------------
    def _try_digit_extract(self, field: str, message: str,
                           target_field: Optional[str]) -> Optional[Any]:
        """
        Extract a number for simple numeric fields.
        IMPORTANT: Only extract a digit for a field when it is the TARGET field
        OR when the field name is explicitly mentioned in the message.
        Otherwise we risk grabbing unrelated numbers (e.g., the phone number
        being interpreted as number_of_employees).
        """
        max_val = DIGIT_FIELDS.get(field)
        if max_val is None:
            return None

        msg_lower = message.lower().strip()

        # Field keyword hints — if user explicitly mentions the field, we can
        # safely extract a nearby number
        field_hints = {
            "number_of_dependents": ["dependent", "dependant", "children", "enfant",
                                     "à charge", "معال"],
            "number_of_employees": ["employee", "employé", "worker", "staff", "عامل"],
            "applicant_ownership_percentage": ["ownership", "own", "possède",
                                                "propriétaire de", "share", "%"],
        }
        hints = field_hints.get(field, [])
        has_field_hint = any(h in msg_lower for h in hints)

        # Only process this field if:
        # 1. it's the current target, OR
        # 2. the message explicitly mentions a hint word
        if field != target_field and not has_field_hint:
            return None

        # Zero-keyword case for dependents
        if field == "number_of_dependents" and msg_lower in ("0", "none", "no", "aucun", "zero", "zéro"):
            return 0

        # Extract first number (near the hint if possible)
        if has_field_hint and hints:
            # Try to find a number near the hint word
            for hint in hints:
                pattern = rf'(\d+(?:\.\d+)?)[^\d]*{re.escape(hint)}|{re.escape(hint)}[^\d]*(\d+(?:\.\d+)?)'
                m = re.search(pattern, msg_lower)
                if m:
                    raw = m.group(1) or m.group(2)
                    val = float(raw)
                    if field == "applicant_ownership_percentage":
                        if 0 <= val <= 100:
                            return val
                    else:
                        val = int(val)
                        if 0 <= val <= max_val:
                            return val
        else:
            # Target field: just grab first number
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
    def _try_llm_extract(self, field: str, message: str,
                         last_question: str) -> Optional[Any]:
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

            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            clean = match.group(1) if match else raw

            parsed = json.loads(clean)

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