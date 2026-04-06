# src/nodes/collect_node.py

from src.models.global_state import GlobalState, get_fields_to_ask, compute_tier
from src.agents.collect_agent import CollectAgent


def collect_node(state: GlobalState) -> dict:
    """
    Thin wrapper: pulls context from state, calls CollectAgent, writes back.
    
    All extraction logic (regex tools, keyword match, LLM fallback)
    lives in CollectAgent.run().
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    fields_to_ask = get_fields_to_ask(state)
    thoughts = list(state.get("thought_steps", []))

    # ---------------------------------------------------------
    # All fields collected — nothing to extract
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
    # Extract one field via the agent
    # ---------------------------------------------------------
    target_field = fields_to_ask[0]
    latest_message = messages[-1].content
    last_question = state.get("last_response", "")

    agent = CollectAgent()
    result = agent.run(
        goal=f"Extract the value for '{target_field}' from the user message.",
        target_field=target_field,
        message=latest_message,
        last_question=last_question,
    )

    # ---------------------------------------------------------
    # Write result back to state
    # ---------------------------------------------------------
    update = {}
    extracted_val = result.get("value")
    method = result.get("method", "unknown")

    if extracted_val is not None:
        update[target_field] = extracted_val
        thoughts.append(f"Extracted: {target_field} = {extracted_val} ({method})")
    else:
        thoughts.append(f"Could not extract '{target_field}' from user message. Will re-ask.")

    # Credit score trigger
    if (update.get("national_id") or state.get("national_id")) and not state.get("credit_score_fetched"):
        update["credit_score_fetched"] = True
        thoughts.append("Credit score fetch triggered")

    # Tier computation
    if state.get("loan_type") and (update.get("loan_amount") or state.get("loan_amount")):
        merged = {**state, **update}
        update["compliance_tier"] = compute_tier(merged)

    # Check if all fields are now complete
    merged = {**state, **update}
    if not get_fields_to_ask(merged):
        update["application_status"] = "awaiting_documents"
        thoughts.append("All required fields now collected!")

    update["thought_steps"] = thoughts
    return update