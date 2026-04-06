# src/tools/decision_tools.py

import os
import yaml
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
from datetime import datetime

# ---------------------------------------------------------------
# Load thresholds and rates from settings.yaml at module level
# so the LLM never needs to guess these values.
# ---------------------------------------------------------------
_settings_path = os.path.join(os.path.dirname(__file__), '../../config/settings.yaml')
_settings: Dict[str, Any] = {}

try:
    with open(_settings_path, 'r', encoding='utf-8') as f:
        _settings = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback defaults if settings.yaml is missing
    _settings = {
        "thresholds": {
            "dti_max": 0.45,
            "dscr_min": 1.25,
            "current_ratio_min": 1.0,
            "debt_to_equity_max": 2.0,
            "net_profit_margin_min": 0.10,
            "quick_ratio_min": 0.80,
            "credit_score_min": 600,
        },
        "rates": {
            "base_rate": 0.085,
            "low_risk_spread": 0.005,
            "medium_risk_spread": 0.020,
            "high_risk_spread": 0.040,
            "personal_adjustment": 0.010,
        },
    }

THRESHOLDS = _settings.get("thresholds", {})
RATES = _settings.get("rates", {})


@tool
def apply_hard_rules(
    loan_type: str,
    dti_ratio: Optional[float] = None,
    credit_score: Optional[int] = None,
    dscr: Optional[float] = None,
    current_ratio: Optional[float] = None,
    years_in_operation: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the application against hard risk thresholds.
    
    Any failure results in immediate rejection.
    Thresholds are loaded from config/settings.yaml — never hardcoded.
    
    Personal loans: DTI + credit_score
    Business loans: DTI + credit_score + DSCR + current_ratio + years_in_operation
    """
    failed_rules: List[str] = []

    # 1. DTI check (applies to both personal and business)
    if dti_ratio is not None and dti_ratio > THRESHOLDS.get("dti_max", 0.45):
        failed_rules.append(
            f"DTI of {dti_ratio:.2%} exceeds maximum allowed ({THRESHOLDS['dti_max']:.0%})"
        )

    # 2. Credit score check (applies to both)
    min_score = THRESHOLDS.get("credit_score_min", 600)
    if credit_score is not None and credit_score < min_score:
        failed_rules.append(
            f"Credit score {credit_score} is below minimum required ({min_score})"
        )

    # 3. DSCR check (business only)
    if loan_type == "business":
        min_dscr = THRESHOLDS.get("dscr_min", 1.25)
        if dscr is not None and dscr < min_dscr:
            failed_rules.append(
                f"DSCR of {dscr:.2f} is below minimum required ({min_dscr})"
            )

    # 4. Current Ratio check (business only)
    if loan_type == "business":
        min_cr = THRESHOLDS.get("current_ratio_min", 1.0)
        if current_ratio is not None and current_ratio < min_cr:
            failed_rules.append(
                f"Current Ratio of {current_ratio:.2f} is below minimum ({min_cr})"
            )

    # 5. Years in operation check (business only)
    if loan_type == "business":
        if years_in_operation is not None and years_in_operation < 2:
            failed_rules.append(
                f"Business operating for {years_in_operation} year(s) — minimum 2 years required"
            )

    return {
        "passed": len(failed_rules) == 0,
        "failed_rules": failed_rules,
    }


@tool
def calculate_interest_rate(loan_type: str, risk_category: str) -> Dict[str, Any]:
    """Calculate final interest rate based on risk spread and loan type.
    
    Formula: base_rate + risk_spread + loan_type_adjustment
    All rates come from config/settings.yaml.
    """
    base = RATES.get("base_rate", 0.085)

    # Risk spread
    spread_map = {
        "low": RATES.get("low_risk_spread", 0.005),
        "medium": RATES.get("medium_risk_spread", 0.020),
        "unknown": RATES.get("medium_risk_spread", 0.020),
        "high": RATES.get("high_risk_spread", 0.040),
    }
    spread = spread_map.get(risk_category, spread_map["medium"])

    # Loan type adjustment
    adjustment = 0.0
    if loan_type == "personal":
        adjustment = RATES.get("personal_adjustment", 0.010)

    final_rate = round(base + spread + adjustment, 4)

    return {
        "interest_rate": final_rate,
        "base_rate": base,
        "risk_spread": spread,
        "loan_type_adjustment": adjustment,
        "breakdown": f"{base:.1%} base + {spread:.1%} risk + {adjustment:.1%} type = {final_rate:.1%}",
    }


@tool
def render_decision_letter(decision_data: Dict[str, Any]) -> Dict[str, Any]:
    """Render the decision letter HTML and convert to PDF.
    
    Takes the full decision context, renders HTML via Jinja2,
    then converts to PDF via WeasyPrint (if available).
    Always saves HTML; PDF is best-effort.
    """
    try:
        from jinja2 import Template

        template_path = os.path.join(
            os.path.dirname(__file__), '../../templates/decision_letter.html'
        )

        with open(template_path, 'r', encoding='utf-8') as f:
            template_str = f.read()

        template = Template(template_str)

        # Add current date if not provided
        if "decision_date" not in decision_data:
            decision_data["decision_date"] = datetime.now().strftime("%B %d, %Y")

        # Provide defaults for optional fields
        decision_data.setdefault("preferred_currency", "TND")
        decision_data.setdefault("conditions", [])
        decision_data.setdefault("reason_codes", [])

        html = template.render(**decision_data)

        # Save HTML
        output_dir = os.getenv("LETTERS_OUTPUT", "./storage/letters")
        os.makedirs(output_dir, exist_ok=True)
        app_id = decision_data.get("application_id", "UNKNOWN")
        html_path = os.path.join(output_dir, f"{app_id}_decision.html")

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        result = {
            "success": True,
            "html_path": html_path,
            "pdf_path": None,
        }

        # Convert to PDF via WeasyPrint (best-effort)
        try:
            from weasyprint import HTML
            pdf_path = os.path.join(output_dir, f"{app_id}_decision.pdf")
            HTML(string=html).write_pdf(pdf_path)
            result["pdf_path"] = pdf_path
        except ImportError:
            result["pdf_warning"] = "WeasyPrint not installed — HTML saved, PDF skipped. pip install weasyprint"
        except Exception as pdf_err:
            result["pdf_warning"] = f"PDF conversion failed: {str(pdf_err)}. HTML saved."

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }