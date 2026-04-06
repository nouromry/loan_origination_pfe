# src/agents/document_agent.py

"""
Document Agent — Processes financial documents.

ARCHITECTURE:
  The document_node handles: file discovery → text extraction → classification → CIN parsing
  This agent handles ONLY: financial data extraction from already-extracted text.

  It overrides BaseAgent.run() with a SINGLE direct LLM call that takes
  the extracted text + doc_type and returns structured JSON.
  No ReAct loop — the LLM gets one shot with a very strict prompt.
"""

import json
import re
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent
from src.tools.document_tools import (
    extract_text,
    ocr_extract_text,
    classify_document_type,
    parse_cin_card,
    cross_check_cin,
)


# Extraction prompts per document type — tells the LLM exactly what fields to find
EXTRACTION_PROMPTS = {
    "salary_slip": {
        "fields": ["gross_salary", "net_salary", "employer_name", "employment_status", "employment_duration_months"],
        "instructions": (
            "Extract these fields from the salary slip:\n"
            "- gross_salary: total salary before deductions (number)\n"
            "- net_salary: take-home pay after deductions (number)\n"
            "- employer_name: company/employer name (string)\n"
            "- employment_status: 'permanent', 'contract', or 'temporary' (string)\n"
            "- employment_duration_months: how long employed, in months (number, estimate from dates if needed)\n"
        ),
    },
    "bank_statement": {
        "fields": ["monthly_cash_flow", "total_monthly_credits", "existing_debt_payments", "monthly_rent_or_mortgage"],
        "instructions": (
            "Extract these fields from the bank statement:\n"
            "- monthly_cash_flow: average net monthly cash flow = credits - debits (number)\n"
            "- total_monthly_credits: average total monthly deposits/credits (number)\n"
            "- existing_debt_payments: total monthly loan/debt payments you can identify (number, 0 if none)\n"
            "- monthly_rent_or_mortgage: monthly rent or mortgage payment (number, 0 if none found)\n"
        ),
    },
    "income_statement": {
        "fields": ["total_revenue", "gross_profit", "total_expenses", "net_income"],
        "instructions": (
            "Extract these fields from the income statement:\n"
            "- total_revenue: total sales/revenue (number)\n"
            "- gross_profit: revenue minus cost of goods sold (number)\n"
            "- total_expenses: total operating expenses (number)\n"
            "- net_income: final net profit/loss (number, negative if loss)\n"
        ),
    },
    "balance_sheet": {
        "fields": ["total_assets", "current_assets", "total_liabilities", "current_liabilities", "equity"],
        "instructions": (
            "Extract these fields from the balance sheet:\n"
            "- total_assets: total assets (number)\n"
            "- current_assets: current/short-term assets (number)\n"
            "- total_liabilities: total liabilities (number)\n"
            "- current_liabilities: current/short-term liabilities (number)\n"
            "- equity: shareholders equity / capitaux propres (number)\n"
        ),
    },
    "business_registration": {
        "fields": ["business_name", "business_registration_number", "legal_structure", "business_address", "years_in_operation"],
        "instructions": (
            "Extract these fields from the business registration document:\n"
            "- business_name: official business name (string)\n"
            "- business_registration_number: registration/matricule number (string)\n"
            "- legal_structure: SARL, SA, SUARL, auto-entrepreneur, etc. (string)\n"
            "- business_address: registered address (string)\n"
            "- years_in_operation: years since registration (number, calculate from date if needed)\n"
        ),
    },
}


class DocumentAgent(BaseAgent):
    """
    Processes financial documents via direct LLM parsing.
    No ReAct loop — one-shot extraction per document.
    """

    def get_tools(self) -> List[BaseTool]:
        return [
            extract_text,
            ocr_extract_text,
            classify_document_type,
            parse_cin_card,
            cross_check_cin,
        ]

    def get_prompt_key(self) -> str:
        return "document_agent"

    # ------------------------------------------------------------------
    # Override BaseAgent.run() — direct LLM call, no ReAct
    # ------------------------------------------------------------------
    def run(self, goal: str, **context) -> Dict[str, Any]:
        """
        Single-shot LLM extraction.
        
        Receives: doc_type, extracted_text, loan_type
        Returns: dict with extracted financial fields
        """
        doc_type = context.get("doc_type", "unknown")
        extracted_text = context.get("extracted_text", "")
        loan_type = context.get("loan_type", "unknown")

        if not extracted_text.strip():
            return {"error": "No text to process", "doc_type": doc_type}

        # Get the extraction config for this doc type
        config = EXTRACTION_PROMPTS.get(doc_type)

        if config is None:
            # Unknown document type — try generic extraction
            return self._generic_extract(extracted_text, doc_type)

        # Build the strict JSON prompt
        fields = config["fields"]
        instructions = config["instructions"]

        system_prompt = (
            "You are a financial document parser. Extract structured data from document text.\n"
            "STRICT RULES:\n"
            "- Return ONLY a valid JSON OBJECT starting with { and ending with }\n"
            "- Do NOT return a JSON array [...]\n"
            "- Do NOT do math or calculations inside the JSON values\n"
            "- Every value must be a final number, string, or null\n"
            "- WRONG: \"monthly_cash_flow\": 3169 - 850 - 320\n"
            "- RIGHT: \"monthly_cash_flow\": 1999\n"
            "- Numbers must be plain numbers with no currency symbols or commas\n"
            "- If you cannot find a value, use null\n"
        )

        user_prompt = (
            f"Document type: {doc_type}\n"
            f"Loan type: {loan_type}\n\n"
            f"{instructions}\n"
            f"Return JSON with these exact keys: {json.dumps(fields)}\n\n"
            f"--- DOCUMENT TEXT ---\n{extracted_text[:3500]}\n--- END ---\n\n"
            f"JSON output:"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            return self._parse_json_response(response.content, fields)

        except Exception as e:
            print(f"[DocumentAgent] LLM extraction failed for {doc_type}: {e}")
            return {"error": str(e), "doc_type": doc_type}

    # ------------------------------------------------------------------
    # JSON parsing with fallbacks
    # ------------------------------------------------------------------
    def _parse_json_response(self, content: str, expected_fields: list) -> Dict[str, Any]:
        """Parse JSON from LLM response with multiple fallback strategies."""
        if not content:
            return {f: None for f in expected_fields}

        cleaned = content.strip()

        # Fix common LLM mistakes before parsing:
        # 1. Replace [...] with {...} if it looks like a dict inside brackets
        if cleaned.startswith("[") and ":" in cleaned:
            cleaned = "{" + cleaned[1:]
            if cleaned.endswith("]"):
                cleaned = cleaned[:-1] + "}"

        # 2. Remove math expressions in values (e.g., "value": 3169 - 850 - 320)
        #    Replace with null since we can't trust the LLM's arithmetic
        cleaned = re.sub(
            r':\s*(\d[\d\s\+\-\*\/\.]+\d)\s*([,}])',
            lambda m: ': ' + str(self._try_eval_math(m.group(1))) + m.group(2),
            cleaned
        )

        # Try direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
            # If it parsed as a list, try to convert
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                return result[0]
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Try finding the first { ... } block (greedy)
        brace_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Last resort: regex extract key-value pairs
        result = {}
        for field in expected_fields:
            # Try to find "field": value patterns
            pattern = rf'"{field}"\s*:\s*(?:"([^"]*)"|([\d.]+)|null)'
            m = re.search(pattern, content)
            if m:
                if m.group(1) is not None:
                    result[field] = m.group(1)
                elif m.group(2) is not None:
                    try:
                        result[field] = float(m.group(2))
                    except ValueError:
                        result[field] = m.group(2)
                else:
                    result[field] = None
            else:
                result[field] = None

        if any(v is not None for v in result.values()):
            return result

        print(f"[DocumentAgent] Could not parse JSON from: {content[:200]}")
        return {f: None for f in expected_fields}

    @staticmethod
    def _try_eval_math(expr: str) -> float:
        """Safely evaluate simple arithmetic expressions from LLM output."""
        try:
            # Only allow digits, +, -, *, /, ., and spaces
            if re.match(r'^[\d\s\+\-\*\/\.]+$', expr.strip()):
                result = eval(expr.strip())  # noqa: S307
                return round(float(result), 2)
        except Exception:
            pass
        return None

    def _generic_extract(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Fallback for unknown document types."""
        try:
            response = self.llm.invoke([
                SystemMessage(content="Extract any financial data you can find. Return valid JSON only."),
                HumanMessage(content=f"Document type: {doc_type}\n\nText:\n{text[:2000]}\n\nJSON:"),
            ])
            try:
                return json.loads(response.content.strip())
            except json.JSONDecodeError:
                return {"raw_response": response.content[:500], "doc_type": doc_type}
        except Exception as e:
            return {"error": str(e), "doc_type": doc_type}