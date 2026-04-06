# src/nodes/scoring_node.py

from src.models.global_state import GlobalState, add_thought
from src.agents.scoring_agent import ScoringAgent


def scoring_node(state: GlobalState) -> GlobalState:
    """
    Calculates DTI and calls the ML scoring API.
    Runs for BOTH personal and business loans.

    For business loans, monthly_income may not come from a salary slip.
    Fallback order: monthly_income → monthly_cash_flow → net_income/12
    """
    state["application_status"] = "scoring"
    state["stage"] = "scoring"
    add_thought(state, "Running scoring pipeline...")

    # Resolve income
    loan_type = state.get("loan_type", "personal")
    monthly_income = state.get("monthly_income")

    # Personal loan fallbacks
    if monthly_income is None and loan_type == "personal":
        # Try net_salary directly (in case document_node stored it separately)
        monthly_income = state.get("net_salary")
        if monthly_income:
            add_thought(state, f"Using net_salary ({monthly_income}) as monthly income")

    if monthly_income is None and loan_type == "personal":
        # Try gross_salary as last resort
        monthly_income = state.get("gross_salary")
        if monthly_income:
            add_thought(state, f"Using gross_salary ({monthly_income}) as monthly income fallback")

    # Business loan fallbacks
    if monthly_income is None and loan_type == "business":
        monthly_income = state.get("monthly_cash_flow")
        if monthly_income:
            add_thought(state, f"Using monthly_cash_flow ({monthly_income}) as income proxy")

    if monthly_income is None and loan_type == "business":
        net_income = state.get("net_income")
        if net_income and net_income > 0:
            monthly_income = round(net_income / 12, 2)
            add_thought(state, f"Using net_income/12 ({monthly_income}) as income proxy")

    if monthly_income is None:
        add_thought(state, "WARNING: Could not determine monthly income. Scoring with DTI=None.")

    loan_amount = state.get("loan_amount")
    term_months = state.get("loan_term_months")

    # If critical data is missing, create a minimal result and continue
    if monthly_income is None or loan_amount is None or term_months is None:
        add_thought(state, "Insufficient data for full scoring. Setting risk as unknown.")
        state["scoring_result"] = {
            "dti": None,
            "dti_passed": False,
            "risk_category": "unknown",
            "default_probability": None,
            "api_available": False,
            "summary": "Scoring incomplete — missing income or loan data.",
        }
        state["application_status"] = "scored"
        return state

    agent = ScoringAgent()
    result = agent.run(
        goal="Calculate DTI ratio and fetch the ML risk score using your tools.",
        loan_amount=loan_amount,
        term_months=term_months,
        monthly_income=monthly_income,
        credit_score=state.get("credit_score"),
        loan_type=loan_type,
    )

    state["scoring_result"] = result
    risk = result.get("risk_category", "unknown")
    dti = result.get("dti")

    state["application_status"] = "scored"
    add_thought(
        state,
        f"Scoring complete. DTI={dti}, Risk={risk}"
    )

    return state