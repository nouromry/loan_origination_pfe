from datetime import datetime
from src.models.global_state import GlobalState


def reset_node(state: GlobalState) -> dict:
    """
    Reset application data while keeping the same application_id/session.
    This is core logic so any client (current or future) can trigger reset
    through chat without UI-specific behavior.
    """
    keep_keys = {"application_id", "created_at", "preferred_currency", "messages"}
    update = {}

    # Clear every state key except core identifiers/session keys.
    for key in list(state.keys()):
        if key in keep_keys:
            continue
        update[key] = None

    # Re-initialize workflow defaults
    update.update({
        "loan_type": None,
        "compliance_tier": None,
        "intent": "reset",
        "stage": "collecting",
        "application_status": "collecting_data",
        "rejection_reason": None,
        "credit_score_fetched": False,
        "documents_uploaded": False,
        "document_result": {},
        "scoring_result": {},
        "risk_result": {},
        "decision_result": {},
        "thought_steps": ["Application reset requested by user."],
        "last_response": "",
        "status_message": None,
        "updated_at": datetime.now().isoformat(),
    })

    return update
