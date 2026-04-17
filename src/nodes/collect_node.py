# src/nodes/collect_node.py

from src.models.global_state import (
    GlobalState,
    get_fields_to_ask,
    compute_tier,
    validate_document_extraction,
    CORRECTION_KEYWORDS,
)
from src.agents.collect_agent import CollectAgent
import re

CIN_ID_PATTERN = r'(?<!\d)\d{8}(?!\d)'
CORRECTION_CHOICE_KEYWORDS = ("correct", "typed id", "option 1")


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
    # Correction flow: resolve CIN mismatch / partial extraction
    # ---------------------------------------------------------
    correction_update = _try_resolve_identity_correction(state, messages[-1].content)
    if correction_update:
        thoughts.extend(correction_update.pop("thought_steps", []))
        correction_update["thought_steps"] = thoughts
        return correction_update

    # ---------------------------------------------------------
    # Nothing to collect
    # ---------------------------------------------------------
    if not fields_to_ask:
        update = {}
        if not state.get("documents_uploaded"):
            update["application_status"] = "collecting"
            update["stage"] = "collecting"
            thoughts.append("All required fields collected. Waiting for document upload.")
        else:
            update["application_status"] = "processing"
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
        update["application_status"] = "collecting"
        thoughts.append("All required fields now collected!")

    update["thought_steps"] = thoughts
    return update


def _try_resolve_identity_correction(state: GlobalState, latest_message: str) -> dict:
    """Resolve recoverable CIN mismatch in-chat without blocking the application."""
    if state.get("application_status") != "needs_correction":
        return {}

    mismatch = state.get("identity_mismatch") or {}
    if not mismatch:
        return {}

    update = {}
    thoughts = []
    message_lower = (latest_message or "").lower()
    cin_id = mismatch.get("cin_national_id") or state.get("cin_national_id")

    use_document_choice = any(k in message_lower for k in ["use document", "document id", "option 2"])
    if use_document_choice and cin_id:
        update["national_id"] = cin_id
        thoughts.append("User selected document ID as the authoritative national ID.")
    else:
        extracted_ids = re.findall(CIN_ID_PATTERN, latest_message or "")
        if extracted_ids:
            update["national_id"] = extracted_ids[-1]
            thoughts.append("User provided corrected typed national ID.")
        elif any(k in message_lower for k in CORRECTION_CHOICE_KEYWORDS):
            thoughts.append("User chose to correct typed ID but did not provide the new 8-digit value yet.")
            return {"application_status": "needs_correction", "thought_steps": thoughts}
        else:
            return {}

    merged = {**state, **update}
    merged["identity_mismatch"] = {}
    merged["rejection_reason"] = None
    report = validate_document_extraction(merged)
    update["document_validation"] = report
    update["identity_mismatch"] = {}
    update["rejection_reason"] = None
    update["in_application_mode"] = True

    if report.get("has_issues") or merged.get("cin_missing_fields"):
        update["application_status"] = "needs_correction"
    else:
        update["application_status"] = "ready_for_scoring"
        thoughts.append("Identity correction resolved. Application is ready for scoring.")

    return {"thought_steps": thoughts, **update}
