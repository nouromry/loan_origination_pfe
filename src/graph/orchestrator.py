
from langgraph.graph import StateGraph, END
from src.models.global_state import GlobalState
from src.graph.node_factory import node_factory


def route(state: GlobalState) -> str:
    """
    Dynamic routing function.
    Reads the full state and decides which node should run next.
    Called after triage on every single message.
    """

    # 0. If application is blocked/rejected, go straight to responder
    if state.get("application_status") in ("blocked", "rejected", "approved"):
        if state.get("decision_result") or state.get("rejection_reason"):
            return "responder"

    # 0.5. Status inquiry → straight to responder (no processing)
    if state.get("intent") == "ask_status":
        return "responder"

    # 1. Policy question → skip everything → answer and return
    if state.get("intent") == "policy_question":
        return "policy"

    # 2. Documents uploaded but not yet processed → process them now
    if state.get("documents_uploaded") and not state.get("document_result"):
        return "document"

    # 3. Documents processed, DTI not yet calculated → score now
    #    (both personal and business need DTI)
    if state.get("document_result") and not state.get("scoring_result"):
        return "scoring"

    # 4. Business loan → run risk assessment before decision
    if (state.get("loan_type") == "business"
            and state.get("scoring_result")
            and not state.get("risk_result")):
        return "risk_assessment"

    # 5. Ready for final decision
    if state.get("scoring_result") and not state.get("decision_result"):
        # Personal: only needs scoring_result
        if state.get("loan_type") == "personal":
            return "decision"
        # Business: needs both scoring_result AND risk_result
        elif state.get("risk_result"):
            return "decision"

    # 6. Default: still collecting fields or delivering a response
    return "collect"


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph StateGraph.
    
    Returns the compiled graph ready for .invoke().
    """
    graph = StateGraph(GlobalState)

    # Register all nodes via factory
    graph.add_node("triage", node_factory.triage)
    graph.add_node("collect", node_factory.collect)
    graph.add_node("document", node_factory.document)
    graph.add_node("scoring", node_factory.scoring)
    graph.add_node("risk_assessment", node_factory.risk_assessment)
    graph.add_node("decision", node_factory.decision)
    graph.add_node("policy", node_factory.policy)
    graph.add_node("responder", node_factory.responder)

    # Every message enters at triage
    graph.set_entry_point("triage")

    # After triage → route() reads full state → picks the correct node
    graph.add_conditional_edges("triage", route, {
        "collect": "collect",
        "document": "document",
        "scoring": "scoring",
        "risk_assessment": "risk_assessment",
        "decision": "decision",
        "policy": "policy",
        "responder": "responder",
    })

    # Every processing node → responder → END
    for node_name in ["collect", "document", "scoring", "risk_assessment", "decision", "policy"]:
        graph.add_edge(node_name, "responder")

    graph.add_edge("responder", END)

    return graph.compile()


# Module-level compiled graph — import this
app_graph = build_graph()