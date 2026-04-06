# src/tools/scoring_tools.py

import os
import requests
from langchain_core.tools import tool
from typing import Dict, Any


@tool
def build_scoring_payload(
    loan_amount: float,
    term_months: int,
    monthly_income: float,
    credit_score: int,
    dti: float,
    loan_type: str,
) -> Dict[str, Any]:
    """Build the JSON payload for the ML scoring API.
    
    Assembles all required fields into the format the external
    scoring model expects.
    """
    return {
        "loan_amount": loan_amount,
        "term_months": term_months,
        "monthly_income": monthly_income,
        "credit_score": credit_score,
        "dti_ratio": dti,
        "loan_type": loan_type,
    }


@tool
def call_scoring_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the external ML API to get default probability and risk category.
    
    Uses the SCORING_API_URL from environment. Falls back gracefully
    if the API is unavailable so the application never crashes.
    """
    endpoint_url = os.getenv("SCORING_API_URL", "")

    if not endpoint_url:
        return {
            "success": False,
            "risk_category": "unknown",
            "default_probability": None,
            "error": "SCORING_API_URL not configured",
        }

    try:
        response = requests.post(endpoint_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "risk_category": data.get("risk_category", "unknown"),
            "default_probability": data.get("default_probability"),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "risk_category": "unknown",
            "default_probability": None,
            "error": "Scoring API timed out after 30s",
        }
    except Exception as e:
        return {
            "success": False,
            "risk_category": "unknown",
            "default_probability": None,
            "error": str(e),
        }


@tool
def parse_scoring_response(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and normalize the ML scoring API response.
    
    Converts the raw API response into a standardized format
    with risk_category, default_probability, and a human-readable summary.
    """
    risk_category = api_response.get("risk_category", "unknown")
    default_prob = api_response.get("default_probability")
    success = api_response.get("success", False)

    # Normalize risk category
    valid_categories = {"low", "medium", "high", "unknown"}
    if risk_category not in valid_categories:
        risk_category = "unknown"

    # Build summary
    if success and default_prob is not None:
        summary = (
            f"ML Score: {risk_category.upper()} risk "
            f"(default probability: {default_prob:.1%})"
        )
    elif success:
        summary = f"ML Score: {risk_category.upper()} risk (no probability returned)"
    else:
        summary = f"ML API unavailable — defaulting to UNKNOWN risk. Error: {api_response.get('error', 'N/A')}"

    return {
        "risk_category": risk_category,
        "default_probability": default_prob,
        "api_available": success,
        "summary": summary,
    }
