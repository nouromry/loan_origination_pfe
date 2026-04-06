# src/nodes/responder_node.py

from src.models.global_state import GlobalState, get_fields_to_ask, get_status_summary
from langchain_core.messages import AIMessage, SystemMessage
from src.agents.base_agent import BaseAgent

FIELD_QUESTIONS = {
    "loan_type": "What type of loan are you looking for? (personal or business)",
    "national_id": "What's your national ID number? (8 digits)",
    "email": "What's your email address?",
    "phone": "What's your phone number? (8 digits)",
    "loan_amount": "How much would you like to borrow? (e.g., 50000 or 50K TND)",
    "loan_term_months": "Over how many months would you like to repay? (e.g., 36 months)",
    "date_of_birth": "What's your date of birth?",
    "marital_status": "What's your marital status? (single, married, divorced, widowed)",
    "housing_status": "What's your housing status? (owner, renter, family)",
    "number_of_dependents": "How many dependents do you have?",
    "industry": "What industry is your business in?",
    "number_of_employees": "How many employees does your business have?",
    "applicant_ownership_percentage": "What percentage of the business do you own?",
}

def responder_node(state: GlobalState) -> dict:
    """The ONLY node that generates user-facing text."""
    try:
        response = _generate_llm_response(state)
        status_msg = get_status_summary(state)

        return {
            "last_response": response,
            "status_message": status_msg,
            "messages": [AIMessage(content=response)]
        }
    except Exception as e:
        print(f"\n[CRITICAL] Responder Node Error: {e}")
        fallback = "I'm having trouble processing that right now."
        return {
            "last_response": fallback,
            "messages": [AIMessage(content=fallback)]
        }


def _generate_llm_response(state: GlobalState) -> str:
    prompts = BaseAgent._load_prompts()
    system_prompt = prompts.get("responder_node", "You are a helpful bank assistant.")

    context = _build_context_directive(state)
    full_prompt = f"{system_prompt}\n\n{context}"

    settings = BaseAgent._load_settings()
    model_config = settings.get("models", {}).get("responder", {})

    if isinstance(model_config, str):
        provider, model = "groq", model_config
    else:
        provider = model_config.get("provider", "ollama")
        model = model_config.get("model", "llama3.1:8b")

    llm = BaseAgent._create_llm(provider, model)
    llm.temperature = 0.4

    messages_history = state.get("messages", [])
    if not isinstance(messages_history, list):
        messages_history = []

    recent_messages = messages_history[-4:]
    messages = [SystemMessage(content=full_prompt)] + recent_messages

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"\n[CRITICAL] LLM Error in Responder: {e}")
        return "I encountered an error trying to formulate a response."


def _build_context_directive(state: GlobalState) -> str:
    app_status = state.get("application_status", "collecting_data")
    intent = state.get("intent", "credit_workflow")
    stage = state.get("stage", "collecting")

    # ---------------------------------------------------------
    # CASE 0: User asking about their application status
    # ---------------------------------------------------------
    if intent == "ask_status":
        summary = get_status_summary(state)
        missing = get_fields_to_ask(state)

        if missing:
            next_field = missing[0]
            question = FIELD_QUESTIONS.get(next_field, f"Please provide your {next_field.replace('_', ' ')}.")
            return (
                f"The user asked about their application status. Briefly summarize:\n{summary}\n\n"
                f"Keep the status update to 2 sentences.\n\n"
                f"THEN, in the SAME response, you MUST continue with this EXACT sentence:\n"
                f"'To continue — {question}'\n\n"
                f"Your response MUST contain BOTH the status AND the follow-up question."
            )
        else:
            return (
                f"The user wants to know their application status:\n\n{summary}\n\n"
                f"Present this clearly and warmly."
            )

    # ---------------------------------------------------------
    # CASE 1: Final states — approved / rejected / blocked
    # ---------------------------------------------------------
    if app_status == "approved":
        decision = state.get("decision_result", {})
        return (
            f"GREAT NEWS: The loan has been APPROVED. Congratulate the user warmly.\n"
            f"Decision details: {decision.get('explanation', 'Approved based on strong financial profile.')}\n"
            f"Interest rate: {decision.get('interest_rate', 'N/A')}\n"
            f"Approved amount: {decision.get('approved_amount', state.get('loan_amount', 'N/A'))}\n"
            f"Keep it celebratory but professional. Mention next steps (signing, disbursement)."
        )

    if app_status in ("rejected", "blocked"):
        reason = state.get("rejection_reason", "Application did not meet credit criteria.")
        return (
            f"The loan application was {'BLOCKED' if app_status == 'blocked' else 'REJECTED'}.\n"
            f"Reason: {reason}\n"
            f"Be empathetic and clear. If blocked, explain it's under manual review. "
            f"If rejected, suggest what the user could improve."
        )

    # ---------------------------------------------------------
    # CASE 2: Policy question
    # ---------------------------------------------------------
    if intent == "policy_question":
        policy = state.get("policy_answer", "Our minimum credit score is 600.")
        missing = get_fields_to_ask(state)

        if missing:
            next_field = missing[0]
            question = FIELD_QUESTIONS.get(next_field, f"Please provide your {next_field.replace('_', ' ')}.")
            return (
                f"The user asked a policy question. First, answer it using ONLY this info: {policy}. "
                f"Keep the answer to 1-2 sentences.\n\n"
                f"THEN, in the SAME response, you MUST continue with this EXACT sentence:\n"
                f"'Now, back to your application — {question}'\n\n"
                f"Your response MUST contain BOTH the policy answer AND the follow-up question. "
                f"Do NOT stop after the policy answer."
            )
        else:
            return (
                f"Answer the user's policy question using this exact info: {policy}. "
                f"Keep it to 2-3 sentences."
            )

    # ---------------------------------------------------------
    # CASE 3: Documents just processed
    # ---------------------------------------------------------
    if state.get("document_result") and not state.get("scoring_result"):
        doc_result = state.get("document_result", {})
        quality = doc_result.get("quality_score", "N/A")
        user_msg = doc_result.get("user_message", "")
        return (
            f"Documents have been processed successfully.\n"
            f"Quality score: {quality}\n"
            f"Document feedback: {user_msg}\n"
            f"Tell the user their documents have been reviewed. "
            f"If there were issues, mention them briefly. "
            f"Let them know the system is now calculating their financial ratios. "
            f"Do NOT invent any numbers or scores not provided above."
        )

    # ---------------------------------------------------------
    # CASE 4: Scoring just completed
    # ---------------------------------------------------------
    if state.get("scoring_result") and not state.get("decision_result"):
        scoring = state.get("scoring_result", {})
        dti = scoring.get("dti", "N/A")
        risk = scoring.get("risk_category", "unknown")

        if state.get("loan_type") == "business" and not state.get("risk_result"):
            return (
                f"Financial scoring complete. DTI: {dti}, Risk category: {risk}.\n"
                f"Tell the user their financial ratios have been calculated. "
                f"The system is now running a detailed business risk assessment. "
                f"Keep it brief and reassuring."
            )

        return (
            f"Financial scoring complete. DTI: {dti}, Risk category: {risk}.\n"
            f"Tell the user their financials look processed and a decision is being made. "
            f"Do NOT tell them the DTI number or risk category directly — just say processing is going well."
        )

    # ---------------------------------------------------------
    # CASE 5: Risk assessment done, decision pending
    # ---------------------------------------------------------
    if state.get("risk_result") and not state.get("decision_result"):
        return (
            f"Business risk assessment is complete. "
            f"The system is now making the final credit decision. "
            f"Tell the user briefly that their application is in its final review stage."
        )

    # ---------------------------------------------------------
    # CASE 6: Awaiting documents (status already set)
    # ---------------------------------------------------------
    if app_status == "awaiting_documents":
        return _build_upload_directive(state)

    # ---------------------------------------------------------
    # CASE 7: Data collection
    # ---------------------------------------------------------
    missing_fields = get_fields_to_ask(state)

    # 7a: All fields collected but status not yet updated to awaiting_documents
    #     This catches the gap between collect_node setting the last field
    #     and the status being updated. Without this, the LLM falls to
    #     CASE 8 and hallucinated random questions.
    if not missing_fields:
        if not state.get("documents_uploaded"):
            return _build_upload_directive(state)
        else:
            return (
                f"All fields collected and documents uploaded. Processing is underway. "
                f"Reassure the user that everything is progressing. "
                f"Do NOT ask any more personal questions."
            )

    # 7b: Still have fields to collect — ask the next one
    next_field = missing_fields[0]
    question_to_ask = FIELD_QUESTIONS.get(
        next_field,
        f"Please provide your {next_field.replace('_', ' ')}."
    )

    # Check if last extraction failed
    thoughts = state.get("thought_steps", [])
    extraction_failed = any("Could not extract" in t for t in thoughts[-3:])

    if extraction_failed:
        return (
            f"The user's last message didn't contain a clear value for '{next_field}'. "
            f"Politely let them know you didn't quite catch that, and re-ask: '{question_to_ask}'. "
            f"Give a quick example of what you expect."
        )

    return (
        f"CRITICAL DIRECTIVE: Briefly acknowledge the user's last message, "
        f"but your PRIMARY GOAL is to ask them this question to continue the application: "
        f"'{question_to_ask}'. "
        f"If they asked to upload documents, tell them they can do that soon, "
        f"but you need this information first."
    )


def _build_upload_directive(state: GlobalState) -> str:
    """Build the directive telling the user to upload documents."""
    loan_type = state.get("loan_type", "personal")
    tier = state.get("compliance_tier", "personal")

    settings = BaseAgent._load_settings()
    tier_config = settings.get("compliance_tiers", {}).get(tier, {})
    required_docs = tier_config.get("documents", [])

    if required_docs:
        doc_names = {
            "cin_card": "CIN Card (National ID)",
            "salary_slip": "Salary Slip (last 3 months)",
            "bank_statement": "Bank Statement (last 6 months)",
            "business_registration": "Business Registration Certificate",
            "income_statement": "Income Statement (last year)",
            "balance_sheet": "Balance Sheet (last year)",
            "tax_return": "Tax Return",
            "collateral_appraisal": "Collateral Appraisal",
            "business_plan": "Business Plan",
        }
        docs = ", ".join(doc_names.get(d, d) for d in required_docs)
    elif loan_type == "personal":
        docs = "CIN Card, Salary Slip, and Bank Statement"
    else:
        docs = "CIN Card, Business Registration, Income Statement, Balance Sheet, and Bank Statement"

    return (
        f"All required information has been collected! "
        f"Ask the user to upload: {docs}. "
        f"Be enthusiastic about their progress. Keep it brief. "
        f"Do NOT ask any more personal questions — only ask for document uploads."
    )