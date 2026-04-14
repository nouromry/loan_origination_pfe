# src/models/global_state.py

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def add_thought(state: dict, thought: str) -> None:
    """Helper to append a thought step to the state."""
    if "thought_steps" not in state or state["thought_steps"] is None:
        state["thought_steps"] = []
    state["thought_steps"].append(thought)


def get_status_summary(state: dict) -> str:
    """
    Build a human-readable status summary the user can see at any time.
    Called by the responder when the user asks 'what's my status?'
    """
    app_id = state.get("application_id", "N/A")
    app_status = state.get("application_status", "collecting_data")
    loan_type = state.get("loan_type", "not determined yet")
    loan_amount = state.get("loan_amount")
    stage = state.get("stage", "collecting")

    # Count collected vs required fields
    missing = get_fields_to_ask(state)
    total_fields = len(missing)  # only missing ones
    
    # Build progress checklist
    steps = []
    steps.append(("Loan type identified", loan_type is not None and loan_type != "unknown"))
    steps.append(("Personal details collected", total_fields == 0))
    steps.append(("Credit score fetched", state.get("credit_score_fetched", False)))
    steps.append(("Documents uploaded", state.get("documents_uploaded", False)))
    steps.append(("Documents processed", bool(state.get("document_result"))))
    steps.append(("Financial scoring", bool(state.get("scoring_result"))))

    if loan_type == "business":
        steps.append(("Risk assessment", bool(state.get("risk_result"))))

    steps.append(("Final decision", bool(state.get("decision_result"))))

    progress_lines = []
    for label, done in steps:
        icon = "DONE" if done else "PENDING"
        progress_lines.append(f"  {icon}: {label}")

    # Status message for the LLM
    amount_str = f"{loan_amount:,.0f} TND" if loan_amount else "not specified yet"
    status_label = app_status.replace("_", " ").title()

    summary = (
        f"Application: {app_id}\n"
        f"Status: {status_label}\n"
        f"Loan Type: {loan_type or 'not set'}\n"
        f"Amount: {amount_str}\n"
        f"Stage: {stage}\n"
        f"Missing fields: {', '.join(missing) if missing else 'none'}\n"
        f"\nProgress:\n" + "\n".join(progress_lines)
    )

    return summary


def get_fields_to_ask(state: dict) -> List[str]:
    """
    Returns the list of fields still missing based on loan_type.
    Personal loans need personal fields; business loans need business fields.
    
    FIELD ORDER MATTERS — we ask in a natural conversation flow:
      1. loan_type (personal or business?)
      2. national_id (identity)
      3. email, phone (contact)
      4. loan_amount, loan_term_months (loan details)
      5. Type-specific fields
    """
    # loan_type MUST be asked first — everything else depends on it
    if state.get("loan_type") is None:
        return ["loan_type"]

    # Base fields required for ALL loan types (after we know the type)
    base_fields = [
        "national_id", "email", "phone",
        "loan_amount", "loan_term_months",
    ]

    personal_fields = [
        "date_of_birth", "marital_status", "housing_status",
        "number_of_dependents",
    ]

    business_fields = [
        "industry", "number_of_employees",
        "applicant_ownership_percentage",
    ]

    loan_type = state.get("loan_type")

    if loan_type == "personal":
        required = base_fields + personal_fields
    elif loan_type == "business":
        required = base_fields + business_fields
    else:
        required = base_fields

    missing = []
    for field in required:
        if state.get(field) is None:
            missing.append(field)
    return missing


# ---------------------------------------------------------------
# Document Validation — checks quality of extracted data
# ---------------------------------------------------------------

# Fields that MUST come from documents for scoring/decision to work
CRITICAL_FIELDS_BY_LOAN_TYPE = {
    "personal": {
        "monthly_income": "Net monthly salary from your salary slip",
        "monthly_cash_flow": "Monthly cash flow from your bank statement",
    },
    "business": {
        "total_revenue": "Total revenue from your income statement",
        "net_income": "Net income from your income statement",
        "total_assets": "Total assets from your balance sheet",
        "total_liabilities": "Total liabilities from your balance sheet",
        "years_in_operation": "Years in operation from your business registration",
    },
}


def validate_document_extraction(state: dict) -> dict:
    """
    Validate that document processing produced usable data.
    
    Checks three things:
      1. Failed documents (couldn't be read at all)
      2. Unknown document types (classifier didn't recognize them)
      3. Critical fields missing (required for scoring/decision)
    
    Returns:
        {
            "has_issues": bool,
            "failed_documents": [filename, ...],
            "unknown_documents": [filename, ...],
            "missing_critical_fields": [{"field": name, "description": text}, ...],
            "expected_not_uploaded": [doc_type, ...],
            "user_message": "...",   # human-readable summary
        }
    """
    loan_type = state.get("loan_type", "personal")
    tier = state.get("compliance_tier", "personal")
    doc_result = state.get("document_result", {}) or {}

    report = {
        "has_issues": False,
        "failed_documents": [],
        "unknown_documents": [],
        "missing_critical_fields": [],
        "expected_not_uploaded": [],
        "user_message": "",
    }

    # ---- Check 1: Failed / unknown documents ----
    uploaded_types = set()
    for fname, data in doc_result.items():
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            report["failed_documents"].append(fname)
            continue
        doc_type = data.get("type")
        if doc_type == "unknown" or doc_type is None:
            report["unknown_documents"].append(fname)
            continue
        uploaded_types.add(doc_type)

    # ---- Check 2: Critical fields missing ----
    critical = CRITICAL_FIELDS_BY_LOAN_TYPE.get(loan_type, {})
    for field, description in critical.items():
        if state.get(field) is None:
            report["missing_critical_fields"].append({
                "field": field,
                "description": description,
            })

    # ---- Check 3: Expected tier documents not uploaded ----
    # We need the settings to know the tier requirements. Since we can't
    # import BaseAgent here (circular), we hardcode the tier map.
    TIER_DOCS = {
        "personal": ["cin_card", "salary_slip", "bank_statement"],
        "small": ["cin_card", "bank_statement", "business_registration", "income_statement"],
        "medium": ["cin_card", "bank_statement", "business_registration",
                   "income_statement", "balance_sheet", "tax_return"],
        "large": ["cin_card", "bank_statement", "business_registration",
                  "income_statement", "balance_sheet", "tax_return", "collateral_appraisal"],
        "very_large": ["cin_card", "bank_statement", "business_registration",
                       "income_statement", "balance_sheet", "tax_return",
                       "collateral_appraisal", "business_plan"],
    }
    expected = TIER_DOCS.get(tier, TIER_DOCS["personal"])
    for doc_type in expected:
        if doc_type not in uploaded_types:
            report["expected_not_uploaded"].append(doc_type)

    # ---- Summary ----
    report["has_issues"] = bool(
        report["failed_documents"]
        or report["unknown_documents"]
        or report["missing_critical_fields"]
        or report["expected_not_uploaded"]
    )

    # ---- Build a user-friendly message ----
    lines = []
    if report["failed_documents"]:
        lines.append(
            f"Could not read {len(report['failed_documents'])} document(s): "
            f"{', '.join(report['failed_documents'])}. The file(s) may be corrupted, "
            f"password-protected, or too blurry to process."
        )
    if report["unknown_documents"]:
        lines.append(
            f"Could not identify {len(report['unknown_documents'])} document(s): "
            f"{', '.join(report['unknown_documents'])}. Please make sure you uploaded "
            f"the correct document type."
        )
    if report["expected_not_uploaded"]:
        doc_names = {
            "cin_card": "CIN Card",
            "salary_slip": "Salary Slip",
            "bank_statement": "Bank Statement",
            "business_registration": "Business Registration",
            "income_statement": "Income Statement",
            "balance_sheet": "Balance Sheet",
            "tax_return": "Tax Return",
            "collateral_appraisal": "Collateral Appraisal",
            "business_plan": "Business Plan",
        }
        missing_names = [doc_names.get(d, d) for d in report["expected_not_uploaded"]]
        lines.append(f"Missing required document(s): {', '.join(missing_names)}.")
    if report["missing_critical_fields"]:
        descriptions = [f["description"] for f in report["missing_critical_fields"]]
        lines.append(
            f"Could not find these key values in your documents: "
            f"{'; '.join(descriptions)}."
        )

    report["user_message"] = " ".join(lines) if lines else "All documents processed successfully."
    return report


def compute_tier(state: dict) -> str:
    """
    Determine compliance tier from loan_type and loan_amount.
    Personal loans always return 'personal'.
    Business tiers: small ≤ 50k, medium ≤ 200k, large ≤ 500k, very_large > 500k.
    """
    loan_type = state.get("loan_type")
    loan_amount = state.get("loan_amount")

    if loan_type == "personal":
        return "personal"

    if loan_amount is None:
        return "small"  # default until amount known

    if loan_amount <= 50_000:
        return "small"
    elif loan_amount <= 200_000:
        return "medium"
    elif loan_amount <= 500_000:
        return "large"
    else:
        return "very_large"


class GlobalState(TypedDict, total=False):
    """
    The single source of truth for the entire application.
    Every node reads from it and writes back to it.
    Uses total=False so fields are optional at runtime.
    """

    # ---------------------------------------------------------
    # Shared System & Workflow Fields
    # ---------------------------------------------------------
    application_id: str
    loan_type: Optional[str]              # 'personal' | 'business' | 'unknown'
    compliance_tier: Optional[str]        # 'personal' | 'small' | 'medium' | 'large' | 'very_large'
    preferred_currency: str               # Default 'TND'
    intent: Optional[str]                 # 'credit_workflow' | 'policy_question'
    stage: str                            # 'collecting' | 'processing' | 'scoring' | 'decision' | 'complete'
    application_status: str               # 'collecting_data' | 'pending_review' | 'approved' | 'rejected' | 'blocked'
    rejection_reason: Optional[str]

    # State Booleans
    credit_score_fetched: bool
    documents_uploaded: bool

    # Node Result Containers
    document_result: Dict[str, Any]
    scoring_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    decision_result: Dict[str, Any]

    # Conversation & UI
    thought_steps: List[str]              # Chain of thought exposed to the UI
    messages: Annotated[List[BaseMessage], add_messages]  # LangGraph message history
    last_response: str
    policy_answer: Optional[str]
    status_message: Optional[str]         # Human-readable status summary (updated every node)
    created_at: str
    updated_at: str

    # ---------------------------------------------------------
    # Applicant Identity & Contact (User Typed)
    # ---------------------------------------------------------
    national_id: Optional[str]
    date_of_birth: Optional[str]
    marital_status: Optional[str]
    housing_status: Optional[str]
    email: Optional[str]
    phone: Optional[str]

    # ---------------------------------------------------------
    # Loan Request Details (User Typed)
    # ---------------------------------------------------------
    loan_amount: Optional[float]
    loan_term_months: Optional[int]
    loan_purpose_category: Optional[str]
    loan_purpose_description: Optional[str]

    # Personal specific
    number_of_dependents: Optional[int]

    # Business specific
    applicant_ownership_percentage: Optional[float]
    industry: Optional[str]
    number_of_employees: Optional[int]

    # ---------------------------------------------------------
    # Extracted from CIN Card
    # ---------------------------------------------------------
    name: Optional[str]
    cin_national_id: Optional[str]
    cin_date_of_birth: Optional[str]
    home_address: Optional[str]

    # ---------------------------------------------------------
    # Extracted from Salary Slip (Personal)
    # ---------------------------------------------------------
    employer_name: Optional[str]
    employment_status: Optional[str]
    salary_type: Optional[str]
    gross_salary: Optional[float]
    net_salary: Optional[float]
    employment_duration_months: Optional[int]

    # ---------------------------------------------------------
    # Extracted from Bank Statement
    # ---------------------------------------------------------
    monthly_income: Optional[float]       # Used for cross-check
    monthly_rent_or_mortgage: Optional[float]
    existing_debt_payments: Optional[float]
    monthly_cash_flow: Optional[float]
    total_monthly_credits: Optional[float]

    # ---------------------------------------------------------
    # User Confirmed Income
    # ---------------------------------------------------------
    other_monthly_income: Optional[float]

    # ---------------------------------------------------------
    # Extracted from Business Registration Cert (Business)
    # ---------------------------------------------------------
    business_name: Optional[str]
    business_registration_number: Optional[str]
    legal_structure: Optional[str]
    business_address: Optional[str]
    years_in_operation: Optional[int]

    # ---------------------------------------------------------
    # Extracted from Income Statement (Business)
    # ---------------------------------------------------------
    total_revenue: Optional[float]
    gross_profit: Optional[float]
    total_expenses: Optional[float]
    net_income: Optional[float]

    # ---------------------------------------------------------
    # Extracted from Balance Sheet (Business)
    # ---------------------------------------------------------
    total_assets: Optional[float]
    current_assets: Optional[float]
    total_liabilities: Optional[float]
    current_liabilities: Optional[float]
    equity: Optional[float]

    # ---------------------------------------------------------
    # Auto-Fetched API Data
    # ---------------------------------------------------------
    credit_score: Optional[int]