# src/nodes/triage_node.py

from src.models.global_state import GlobalState, add_thought
from src.agents.triage_agent import TriageAgent


def triage_node(state: GlobalState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}

    latest_message = messages[-1].content

    # Quick keyword check for status inquiries (skip LLM call for speed)
    status_keywords = [
        "status", "where are we", "how far", "progress", "what stage",
        "where is my application", "application status", "how is my",
        "what's happening", "update on my", "check my application",
    ]
    is_status_query = any(kw in latest_message.lower() for kw in status_keywords)

    if is_status_query:
        thoughts = list(state.get("thought_steps", []))
        thoughts.append("Triage: intent=ask_status (keyword match)")
        return {
            "intent": "ask_status",
            "thought_steps": thoughts,
        }

    agent = TriageAgent()
    result = agent.run(
        goal="Classify this message",
        message=latest_message,
        history=[m.content for m in messages[-4:]],
    )

    update = {
        "intent": result.get("intent", "credit_workflow")
    }

    # Only set loan_type if user EXPLICITLY mentions it
    # "I want a personal loan" → yes
    # "I want to apply for a loan" → no (LLM might guess "personal" but user didn't say it)
    detected_type = result.get("loan_type")
    if detected_type in ("personal", "business") and not state.get("loan_type"):
        msg_lower = latest_message.lower()
        explicit_personal = any(kw in msg_lower for kw in ["personal loan", "personal credit", "prêt personnel"])
        explicit_business = any(kw in msg_lower for kw in [
            "business loan", "business credit", "prêt professionnel",
            "for my business", "for my company", "for my restaurant",
            "for my shop", "pour mon entreprise",
        ])
        if explicit_personal:
            update["loan_type"] = "personal"
        elif explicit_business:
            update["loan_type"] = "business"
        # Otherwise: don't set it — let the collect node ask

    # Thought must also be returned, not mutated
    thoughts = list(state.get("thought_steps", []))
    thoughts.append(f"Triage: intent={update['intent']}, loan_type={update.get('loan_type', state.get('loan_type'))}")
    update["thought_steps"] = thoughts

    return update