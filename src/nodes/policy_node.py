# src/nodes/policy_node.py

from src.models.global_state import GlobalState
from src.tools.rag_tools import query_policy

def policy_node(state: GlobalState) -> dict:
    """Answers policy questions using RAG (ChromaDB)."""
    try:
        messages = state.get("messages", [])
        latest_message = messages[-1].content if messages else ""

        try:
            policy_text = query_policy.invoke({"question": latest_message})
        except AttributeError:
            policy_text = query_policy(latest_message)
        
        # We inject strict behavioral constraints directly into the text variable.
        enforced_context = (
            f"DATABASE RESULT: {policy_text}\n\n"
            f"CRITICAL DIRECTIVE: You MUST answer the user's question using ONLY the database result above. "
            f"DO NOT ask for their National ID. DO NOT ask for any application data. "
            f"DO NOT proceed with the loan workflow. Just politely answer the policy question."
        )
        
        return {
            "policy_answer": enforced_context,
            "thought_steps": ["Policy retrieved instantly from ChromaDB."]
        }
    except Exception as e:
        print(f"\n[CRITICAL] Policy Node Error: {e}")
        return {
            "policy_answer": "System error retrieving policy. Apologize to the user and DO NOT ask for their ID.",
            "thought_steps": [f"Policy Node Failed: {e}"]
        }