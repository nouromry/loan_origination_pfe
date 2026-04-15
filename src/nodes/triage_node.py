# src/nodes/triage_node.py

from src.models.global_state import GlobalState
from src.agents.triage_agent import TriageAgent


# Specific policy question indicators — concrete things users ask about
POLICY_SPECIFIC_KEYWORDS = [
    "minimum credit", "maximum", "interest rate", "late fee", "penalty",
    "required document", "how long", "how much", "eligibility",
    "self-employed", "guarantor", "collateral", "early repayment",
    "taux d'intérêt", "crédit minimum", "pénalité", "garant",
    "document requis", "durée maximum", "remboursement anticipé",
]

# Vague "I want to know about policies" patterns — not real questions
VAGUE_POLICY_PATTERNS = [
    "want to ask about",
    "want to know about",
    "have a question about",
    "have questions about",
    "tell me about",
    "ask about policies",
    "ask about bank",
    "veux demander",
    "une question sur",
    "parle-moi des",
    "dis-moi",
]

# Explicit loan type phrases
EXPLICIT_PERSONAL = [
    "personal loan", "personal credit", "prêt personnel", "crédit personnel",
]
EXPLICIT_BUSINESS = [
    "business loan", "business credit", "prêt professionnel",
    "for my business", "for my company", "for my restaurant",
    "for my shop", "for my startup", "pour mon entreprise",
    "pour ma société", "pour mon restaurant", "pour ma boutique",
]


def triage_node(state: GlobalState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}

    latest_message = messages[-1].content
    msg_lower = latest_message.lower().strip()

    thoughts = list(state.get("thought_steps", []))

    # ================================================================
    # Fast path 1: status inquiry
    # ================================================================
    status_keywords = [
        "status", "where are we", "how far", "progress", "what stage",
        "where is my application", "application status", "how is my",
        "what's happening", "update on my", "check my application",
        "où en suis", "ou en suis", "avancement", "état de ma demande",
        "etat de ma demande", "suivi de ma demande",
        "وين وصلت", "حالة طلبي", "شنو حالة طلبي", "فين وصلت",
    ]
    if any(kw in msg_lower for kw in status_keywords):
        thoughts.append("Triage: intent=ask_status (keyword match)")
        return {"intent": "ask_status", "thought_steps": thoughts}

    # ================================================================
    # Fast path 1.5: user data inquiry
    # ================================================================
    data_keywords = [
        "what info do you have on me", "what information do you have on me",
        "what data do you have on me", "what do you know about me",
        "show my data", "show my information", "show what you have",
        "quelles informations avez-vous sur moi", "quelles infos avez-vous sur moi",
        "quelles données avez-vous sur moi", "montre mes informations",
        "ما المعلومات التي لديك عني", "ما هي المعلومات عندك عني",
        "شنو تعرف علي", "شنو المعلومات اللي عندك علي",
    ]
    if any(kw in msg_lower for kw in data_keywords):
        thoughts.append("Triage: intent=ask_data (user requested stored information)")
        return {"intent": "ask_data", "thought_steps": thoughts}

    # ================================================================
    # Fast path 1.6: reset / start over intent
    # ================================================================
    reset_keywords = [
        "start over", "from scratch", "reset", "cancel application",
        "let me cancel", "let me start over", "try again",
        "recommencer", "recommence", "à zéro", "a zero",
        "ابدأ من جديد", "من الصفر", "إعادة البدء", "اعادة البدء",
    ]
    if any(kw in msg_lower for kw in reset_keywords):
        thoughts.append("Triage: intent=reset (user requested fresh start)")
        return {"intent": "reset", "thought_steps": thoughts}

    # ================================================================
    # Fast path 2: explicit specific policy question
    # ================================================================
    has_specific_policy_keyword = any(kw in msg_lower for kw in POLICY_SPECIFIC_KEYWORDS)
    has_question_mark = "?" in msg_lower
    starts_with_question_word = any(
        msg_lower.startswith(qw) for qw in
        ["what", "how", "when", "why", "where", "can you", "is there",
         "do you", "does", "are you", "quoi", "comment", "quand",
         "est-ce que", "quel", "quelle"]
    )

    is_specific_policy = has_specific_policy_keyword and (has_question_mark or starts_with_question_word)

    # ================================================================
    # Fast path 3: vague policy announcement
    # "I want to ask about bank policies" → NOT a policy question yet
    # ================================================================
    is_vague_policy_announcement = any(p in msg_lower for p in VAGUE_POLICY_PATTERNS)

    if is_vague_policy_announcement and not is_specific_policy:
        thoughts.append("Triage: intent=vague_policy (user announced intent but did not ask a specific question)")
        return {
            "intent": "vague_policy",
            "thought_steps": thoughts,
        }

    if is_specific_policy:
        thoughts.append("Triage: intent=policy_question (specific policy keyword + question form)")
        update = {"intent": "policy_question", "thought_steps": thoughts}
        _detect_loan_type(msg_lower, state, update)
        return update

    # ================================================================
    # LLM triage for everything else
    # ================================================================
    agent = TriageAgent()
    result = agent.run(
        goal="Classify this message",
        message=latest_message,
        history=[m.content for m in messages[-4:]],
    )

    update = {"intent": result.get("intent", "credit_workflow")}

    detected_type = result.get("loan_type")
    if detected_type in ("personal", "business") and not state.get("loan_type"):
        _detect_loan_type(msg_lower, state, update)

    thoughts.append(
        f"Triage: intent={update['intent']}, "
        f"loan_type={update.get('loan_type', state.get('loan_type'))}"
    )
    update["thought_steps"] = thoughts
    return update


def _detect_loan_type(msg_lower: str, state: dict, update: dict) -> None:
    """Only set loan_type if the user EXPLICITLY mentions it."""
    if state.get("loan_type"):
        return

    if any(kw in msg_lower for kw in EXPLICIT_PERSONAL):
        update["loan_type"] = "personal"
    elif any(kw in msg_lower for kw in EXPLICIT_BUSINESS):
        update["loan_type"] = "business"
