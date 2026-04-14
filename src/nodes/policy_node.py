# src/nodes/policy_node.py

from src.models.global_state import GlobalState
from src.tools.rag_tools import query_policy


# Question words that signal an actual question
QUESTION_WORDS = [
    "what", "how", "when", "why", "where", "which", "who",
    "can", "could", "is", "are", "do", "does", "will", "would",
    "quoi", "comment", "quand", "pourquoi", "où", "quel", "quelle",
    "peut", "est-ce", "puis-je",
    "ما", "كيف", "متى", "لماذا", "أين",
]

# Specific policy topics that indicate a concrete question
POLICY_TOPICS = [
    "rate", "interest", "minimum", "maximum", "credit score",
    "requirement", "eligibility", "late", "penalty", "fee",
    "document", "term", "duration", "amount", "limit",
    "self-employed", "guarantor", "collateral", "approval",
    "taux", "intérêt", "minimum", "maximum", "exigence", "pénalité",
    "document", "durée", "montant", "approbation", "garant",
]


def _is_vague_query(message: str) -> bool:
    """
    Detect if the query is too vague to answer with RAG.
    Vague: "tell me about policies", "I want to ask something"
    Specific: "what's the minimum credit score?", "do you accept self-employed?"
    """
    msg = message.lower().strip()
    words = msg.split()

    # Very short messages with no specifics
    if len(words) < 3:
        return True

    # No question word AND no policy topic → vague
    has_question_word = any(qw in msg for qw in QUESTION_WORDS)
    has_policy_topic = any(topic in msg for topic in POLICY_TOPICS)
    has_question_mark = "?" in msg

    if not has_question_word and not has_policy_topic and not has_question_mark:
        return True

    # Catches patterns like "I want to ask about policies" where user is
    # announcing intent, not asking a specific question
    announce_patterns = [
        "want to ask", "want to know", "have a question", "have questions",
        "tell me about", "ask about", "learn about",
        "veux demander", "aimerais savoir", "parle-moi de",
    ]
    if any(pattern in msg for pattern in announce_patterns):
        # If there's also a specific topic, it's OK
        if not has_policy_topic:
            return True

    return False


def policy_node(state: GlobalState) -> dict:
    """
    Answers policy questions using RAG (ChromaDB).
    
    If the query is vague, returns a clarification prompt instead of
    calling RAG (which would return irrelevant chunks and hallucinate).
    """
    try:
        messages = state.get("messages", [])
        latest_message = messages[-1].content if messages else ""

        # Guard: vague query → ask for clarification, don't call RAG
        if _is_vague_query(latest_message):
            return {
                "policy_answer": (
                    "CLARIFICATION NEEDED: The user mentioned wanting to know about policies "
                    "but did not ask a specific question. Ask them politely what specific topic "
                    "they want to know about (e.g., interest rates, minimum credit score, "
                    "required documents, eligibility criteria). DO NOT invent any policy content. "
                    "DO NOT mention any specific policy details."
                ),
                "thought_steps": [
                    "Policy Node: query too vague. Asking user to clarify instead of calling RAG."
                ],
            }

        # Query ChromaDB for relevant policy chunks
        try:
            policy_text = query_policy.invoke({"question": latest_message})
        except AttributeError:
            policy_text = query_policy(latest_message)

        # Guard: RAG returned nothing useful
        if not policy_text or len(str(policy_text).strip()) < 20:
            return {
                "policy_answer": (
                    "NO RESULT: The policy database did not return a relevant answer. "
                    "Tell the user you don't have that specific information in the policy "
                    "database and suggest they contact a bank advisor. DO NOT invent policy content."
                ),
                "thought_steps": ["Policy Node: RAG returned empty/irrelevant results."],
            }

        # Inject strict behavioral constraints into the context
        enforced_context = (
            f"DATABASE RESULT:\n{policy_text}\n\n"
            f"CRITICAL DIRECTIVE: Answer the user's EXACT question using ONLY the database result above. "
            f"Do not add information not in the database. Do not invent rates or numbers. "
            f"Do not ask for their National ID. Do not proceed with the loan workflow. "
            f"Just answer the policy question concisely and politely."
        )

        return {
            "policy_answer": enforced_context,
            "thought_steps": ["Policy retrieved from ChromaDB RAG."],
        }

    except Exception as e:
        print(f"\n[CRITICAL] Policy Node Error: {e}")
        return {
            "policy_answer": (
                "SYSTEM ERROR: Could not retrieve policy information. "
                "Apologize briefly and suggest contacting a bank advisor. DO NOT invent any policy details."
            ),
            "thought_steps": [f"Policy Node Failed: {e}"],
        }