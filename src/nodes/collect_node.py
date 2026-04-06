# src/nodes/collect_node.py

import json
import re
from src.models.global_state import GlobalState, get_fields_to_ask, compute_tier
from src.agents.collect_agent import CollectAgent
from langchain_core.messages import SystemMessage, HumanMessage


def collect_node(state: GlobalState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}

    fields_to_ask = get_fields_to_ask(state)
    thoughts = list(state.get("thought_steps", []))

    # ---------------------------------------------------------
    # BUG FIX: When all fields are collected, DON'T silently skip.
    # Tell the responder what's happening so it doesn't hallucinate.
    # ---------------------------------------------------------
    if not fields_to_ask:
        update = {}

        if not state.get("documents_uploaded"):
            # All fields done, waiting for documents
            update["application_status"] = "awaiting_documents"
            update["stage"] = "collecting"
            thoughts.append("All required fields collected. Waiting for document upload.")
        else:
            # Fields done AND documents uploaded — shouldn't be here
            # (orchestrator should have routed to document/scoring/decision)
            # But just in case, mark it and let responder handle gracefully
            update["stage"] = "processing"
            thoughts.append("All fields collected and documents uploaded. Ready for processing.")

        update["thought_steps"] = thoughts
        return update

    target_field = fields_to_ask[0]
    latest_message = messages[-1].content
    last_question = state.get("last_response", "")

    agent = CollectAgent()

    # ONE-SHOT DIRECTIVE (FAST)
    system_directive = (
        f"You are a strict data extraction assistant.\n"
        f"The user was asked: '{last_question}'.\n"
        f"Extract ONLY the value for this field: '{target_field}'.\n\n"
        f"Respond ONLY with valid JSON.\n"
        f"Format: {{\"{target_field}\": value}}\n"
        f"If missing, return: {{\"{target_field}\": null}}"
    )

    try:
        response = agent.llm.invoke([
            SystemMessage(content=system_directive),
            HumanMessage(content=latest_message)
        ])

        raw_content = response.content

        # Clean markdown if exists
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
        clean_json = match.group(1) if match else raw_content.strip()

        parsed = json.loads(clean_json)

    except Exception as e:
        print(f"[Collect Error] {e}")
        parsed = {target_field: None}

    update = {}

    # --- SAFE EXTRACTION LOGIC ---
    extracted_val = None

    if target_field in parsed:
        extracted_val = parsed[target_field]
    elif "value" in parsed:
        extracted_val = parsed["value"]
    else:
        for k, v in parsed.items():
            if k not in ("raw_response", "error") and v is not None:
                extracted_val = v
                break

    # Save extracted value
    if extracted_val is not None:
        # Special handling for loan_type — normalize to personal/business
        if target_field == "loan_type":
            val_lower = str(extracted_val).lower().strip()
            if "personal" in val_lower or "personnel" in val_lower:
                extracted_val = "personal"
            elif "business" in val_lower or "entreprise" in val_lower or "professionnel" in val_lower:
                extracted_val = "business"
            else:
                # Couldn't parse — don't save garbage
                extracted_val = None
                thoughts.append(f"Could not determine loan_type from '{val_lower}'. Will re-ask.")

        if extracted_val is not None:
            update[target_field] = extracted_val
            thoughts.append(f"Extracted: {target_field} = {extracted_val}")
    else:
        # BUG FIX: Log when extraction fails so responder can re-ask
        thoughts.append(f"Could not extract '{target_field}' from user message. Will re-ask.")

    # Credit score trigger
    if update.get("national_id") or state.get("national_id"):
        if not state.get("credit_score_fetched"):
            update["credit_score_fetched"] = True
            thoughts.append("Credit score fetch triggered")

    # Tier computation
    if state.get("loan_type") and (update.get("loan_amount") or state.get("loan_amount")):
        merged_state = {**state, **update}
        update["compliance_tier"] = compute_tier(merged_state)

    # Check if this extraction completed all fields
    merged_for_check = {**state, **update}
    remaining = get_fields_to_ask(merged_for_check)
    if not remaining:
        update["application_status"] = "awaiting_documents"
        thoughts.append("All required fields now collected!")

    update["thought_steps"] = thoughts

    return update