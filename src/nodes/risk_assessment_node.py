# src/nodes/risk_assessment_node.py

from src.models.global_state import GlobalState, add_thought
from src.agents.risk_assessment_agent import RiskAssessmentAgent


def risk_assessment_node(state: GlobalState) -> GlobalState:
    """
    Runs the business risk assessment.
    Only executes if loan_type is 'business'.

    The node is thin — it just:
      1. Pulls data from GlobalState
      2. Calls the agent
      3. Writes results back to GlobalState

    All logic (ratio calc, benchmarks, comparison, LLM analysis)
    lives in RiskAssessmentAgent.
    """
    # Safety check: only for business loans
    if state.get("loan_type") != "business":
        return state

    state["application_status"] = "risk_assessment"
    state["stage"] = "scoring"
    add_thought(state, "Running business risk assessment...")

    # Build context from state (the agent doesn't read state directly)
    context = {
        "industry":             state.get("industry", "General"),
        "loan_amount":          state.get("loan_amount"),
        "loan_term_months":     state.get("loan_term_months"),
        "monthly_cash_flow":    state.get("monthly_cash_flow"),
        "total_revenue":        state.get("total_revenue"),
        "net_income":           state.get("net_income"),
        "total_assets":         state.get("total_assets"),
        "current_assets":       state.get("current_assets"),
        "total_liabilities":    state.get("total_liabilities"),
        "current_liabilities":  state.get("current_liabilities"),
        "equity":               state.get("equity"),
    }

    agent = RiskAssessmentAgent()
    result = agent.run(
        goal="Calculate all business financial ratios, compare to industry benchmarks, and write qualitative analysis.",
        **context
    )

    state["risk_result"] = result

    risk_level = result.get("risk_level", "unknown")
    passed = result.get("passed_count", 0)
    total = result.get("total_ratios", 0)

    state["application_status"] = "risk_assessed"
    add_thought(state, f"Risk assessment complete. {passed}/{total} ratios passed. Level: {risk_level.upper()}")

    return state