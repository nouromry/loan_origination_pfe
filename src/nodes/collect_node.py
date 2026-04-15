# src/nodes/collect_node.py

from src.models.global_state import GlobalState, get_fields_to_ask, compute_tier
from src.agents.collect_agent import CollectAgent


def collect_node(state: GlobalState) -> dict:
    """
    Thin wrapper around CollectAgent.
    
    On each message, tries to extract ALL missing fields at once —
    so if the user dumps everything in one message, the agent captures
    as much as possible in one pass and only re-asks for what's left.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    fields_to_ask = get_fields_to_ask(state)
    thoughts = list(state.get("thought_steps", []))

    # ---------------------------------------------------------
    # Nothing to collect
    # ---------------------------------------------------------
    if not fields_to_ask:
        update = {}
        if not state.get("documents_uploaded"):
            update["application_status"] = "awaiting_documents"
            update["stage"] = "collecting"
            thoughts.append("All required fields collected. Waiting for document upload.")
        else:
            update["stage"] = "processing"
            thoughts.append("All fields collected and documents uploaded. Ready for processing.")
        update["thought_steps"] = thoughts
        return update

    # ---------------------------------------------------------
    # Multi-field extraction from the latest message
    # ---------------------------------------------------------
    latest_message = messages[-1].content
    last_question = state.get("last_response", "")

    agent = CollectAgent()
    result = agent.run_multi(
        missing_fields=fields_to_ask,
        message=latest_message,
        last_question=last_question,
    )

    extracted = result.get("extracted", {})
    methods = result.get("methods", {})

    # ---------------------------------------------------------
    # Write all extracted fields back to state
    # ---------------------------------------------------------
    update = {}

    for field, value in extracted.items():
        update[field] = value
        method = methods.get(field, "unknown")
        thoughts.append(f"Extracted: {field} = {value} ({method})")

    if not extracted:
        target = fields_to_ask[0]
        thoughts.append(f"Could not extract '{target}' from user message. Will re-ask.")
    elif len(extracted) > 1:
        thoughts.append(f"Multi-field extraction: captured {len(extracted)} fields in one message.")

    if extracted:
        update["in_application_mode"] = True

    # ---------------------------------------------------------
    # Credit score trigger
    # ---------------------------------------------------------
    if (update.get("national_id") or state.get("national_id")) and not state.get("credit_score_fetched"):
        update["credit_score_fetched"] = True
        thoughts.append("Credit score fetch triggered")

    # ---------------------------------------------------------
    # Tier computation
    # ---------------------------------------------------------
    if state.get("loan_type") or update.get("loan_type"):
        if update.get("loan_amount") or state.get("loan_amount"):
            merged = {**state, **update}
            update["compliance_tier"] = compute_tier(merged)

    # ---------------------------------------------------------
    # Check if all fields are now complete
    # ---------------------------------------------------------
    merged = {**state, **update}
    remaining = get_fields_to_ask(merged)
    if not remaining:
        update["application_status"] = "awaiting_documents"
        thoughts.append("All required fields now collected!")

    update["thought_steps"] = thoughts
    return update
