# src/nodes/decision_node.py

from src.models.global_state import GlobalState, add_thought
from src.agents.decision_agent import DecisionAgent


def decision_node(state: GlobalState) -> GlobalState:
    """
    Makes the final approval/rejection/conditional decision.

    The node is thin — it just:
      1. Pulls context from GlobalState
      2. Calls the DecisionAgent
      3. Writes results back to GlobalState

    All logic (hard rules, interest rate, LLM explanation,
    letter rendering, DB save) lives in DecisionAgent.
    """
    state["application_status"] = "processing"
    state["stage"] = "decision"
    add_thought(state, "Making final credit decision...")

    # Build context from state
    context = {
        "scoring_result": state.get("scoring_result", {}),
        "risk_result": state.get("risk_result", {}),
        "credit_score": state.get("credit_score"),
        "loan_type": state.get("loan_type", "personal"),
        "loan_amount": state.get("loan_amount"),
        "loan_term_months": state.get("loan_term_months"),
        "loan_purpose_category": state.get("loan_purpose_category"),
        "name": state.get("name", "Applicant"),
        "application_id": state.get("application_id", "UNKNOWN"),
        # For DB save
        "national_id": state.get("national_id"),
        "email": state.get("email"),
        "phone": state.get("phone"),
        "monthly_income": state.get("monthly_income"),
        "application_status": state.get("application_status"),
    }

    # Business-specific fields
    if state.get("loan_type") == "business":
        context["years_in_operation"] = state.get("years_in_operation")
        context["industry"] = state.get("industry")
        context["business_name"] = state.get("business_name")
        risk = state.get("risk_result", {})
        ratios = risk.get("ratios", {})
        context["dscr"] = ratios.get("dscr", {}).get("dscr") if isinstance(ratios.get("dscr"), dict) else None
        context["current_ratio"] = ratios.get("current_ratio", {}).get("current_ratio") if isinstance(ratios.get("current_ratio"), dict) else None
        context["risk_level"] = risk.get("risk_level", "unknown")
        context["risk_recommendation"] = risk.get("recommendation", "unknown")

    # Run the agent (handles everything: rules → rate → LLM → letter → DB)
    agent = DecisionAgent()
    result = agent.run(
        goal="Apply hard rules, evaluate all data, and make the final credit decision.",
        **context
    )

    # Write results back to state
    state["decision_result"] = result

    decision = result.get("decision", "rejected").lower()

    if decision == "approved":
        state["application_status"] = "approved"
    elif decision == "conditional":
        state["application_status"] = "approved"
    else:
        state["application_status"] = "rejected"
        state["rejection_reason"] = result.get(
            "reason", "Application did not meet credit criteria."
        )

    state["stage"] = "complete"
    add_thought(state, f"Decision: {decision.upper()}")

    return state
