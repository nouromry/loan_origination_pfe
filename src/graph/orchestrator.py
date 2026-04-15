# src/graph/orchestrator.py

"""
LangGraph Orchestrator — The Brain of AXE Finance.

This is NOT a pipeline. It reads the full GlobalState on every message
and decides dynamically which node to run. The user can do anything
in any order — the system adapts.

Flow:
    User message → triage → route() picks next node → node runs → responder → END
"""

import os
import re
from langgraph.graph import StateGraph, END
from src.models.global_state import GlobalState
from src.graph.node_factory import node_factory


def _missing_mandatory_chat_fields(state: dict) -> bool:
    """
    Check if mandatory chat-collected fields are still missing.
    Documents alone are not enough to make a decision — we need basic
    loan parameters from the chat (amount, term, etc.).
    """
    mandatory = ["loan_type", "loan_amount", "loan_term_months", "national_id"]
    for field in mandatory:
        if state.get(field) is None:
            return True
    return False


def _has_unprocessed_files(state: dict) -> bool:
    """
    Check if there are uploaded files that haven't been processed yet.
    A file is "processed" if its name+size key is in state['processed_files'].
    Enables incremental document upload.
    """
    upload_dir = os.getenv("TEMP_UPLOADS_DIR", "./temp_uploads")
    app_id = state.get("application_id", "UNKNOWN")
    app_dir = os.path.join(upload_dir, app_id)

    if not os.path.exists(app_dir):
        return False

    processed = set(state.get("processed_files") or [])
    valid_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.tiff')

    try:
        for fname in os.listdir(app_dir):
            if not fname.lower().endswith(valid_extensions):
                continue
            try:
                fsize = os.path.getsize(os.path.join(app_dir, fname))
            except OSError:
                continue
            file_key = f"{fname}:{fsize}"
            if file_key not in processed:
                return True
    except OSError:
        return False

    return False


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

    # 0.55. User asks what data we have on file
    if state.get("intent") == "ask_data":
        return "responder"

    # 0.57. User asked to start over
    if state.get("intent") == "reset":
        return "reset"

    # 0.6. Vague policy announcement ("I want to ask about policies" without a specific question)
    #      → skip RAG, let responder ask for clarification
    if state.get("intent") == "vague_policy":
        return "responder"

    # 0.7. Document validation failed — but FIRST try to extract any chat data
    #      from the user's message (loan_amount, email, phone, etc.).
    #      Only block on documents_incomplete if there's nothing to extract.
    if state.get("application_status") == "documents_incomplete":
        # If the user provided any extractable data in chat, route to collect
        # so we can capture it. On the NEXT turn, validation rechecks.
        if _missing_mandatory_chat_fields(state):
            messages = state.get("messages", [])
            if messages:
                latest = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
                # If message contains digits or @, it may have extractable data
                if re.search(r"\d|@", latest):
                    return "collect"

        # Nothing useful in the message — show the "documents needed" message
        return "responder"

    # 1. Policy question → skip everything → answer and return
    if state.get("intent") == "policy_question":
        return "policy"

    # 2. Documents uploaded but some are not yet processed → process them now
    #    Checks both: never-processed (document_result empty) AND incremental
    #    (new files added after prior processing)
    if state.get("documents_uploaded") and _has_unprocessed_files(state):
        return "document"

    # 2.5. Documents processed but mandatory chat fields still missing
    #      → go back to data collection, don't try to score with incomplete data
    if state.get("document_result"):
        if _missing_mandatory_chat_fields(state):
            return "collect"

    # 3. Documents processed, all chat fields present, DTI not yet calculated → score
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
    graph.add_node("reset", node_factory.reset)

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
        "reset": "reset",
        "responder": "responder",
    })

    # Every processing node → responder → END
    for node_name in ["collect", "document", "scoring", "risk_assessment", "decision", "policy", "reset"]:
        graph.add_edge(node_name, "responder")

    graph.add_edge("responder", END)

    return graph.compile()


# Module-level compiled graph — import this
app_graph = build_graph()
