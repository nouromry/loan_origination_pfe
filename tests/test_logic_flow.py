from langchain_core.messages import HumanMessage


def test_get_fields_to_ask_uses_cin_fallbacks():
    from src.models.global_state import get_fields_to_ask

    state = {
        "loan_type": "personal",
        "cin_national_id": "12714074",
        "cin_date_of_birth": "25/09/2001",
        "email": "nour@example.com",
        "phone": "20123456",
        "loan_amount": 50000,
        "loan_term_months": 24,
        "marital_status": "single",
        "housing_status": "renter",
        "number_of_dependents": 0,
    }
    missing = get_fields_to_ask(state)
    assert "national_id" not in missing
    assert "date_of_birth" not in missing


def test_triage_detects_reset_multilingual():
    from src.nodes.triage_node import triage_node

    state = {"messages": [HumanMessage(content="recommencer")], "thought_steps": []}
    out = triage_node(state)
    assert out["intent"] == "reset"


def test_triage_detects_data_inquiry():
    from src.nodes.triage_node import triage_node

    state = {"messages": [HumanMessage(content="what info do you have on me?")], "thought_steps": []}
    out = triage_node(state)
    assert out["intent"] == "ask_data"


def test_reset_node_clears_application_data():
    from src.nodes.reset_node import reset_node

    state = {
        "application_id": "APP_12345678",
        "loan_type": "personal",
        "national_id": "12714074",
        "documents_uploaded": True,
        "document_result": {"cin.pdf": {"type": "cin_card"}},
        "messages": [HumanMessage(content="start over")],
    }
    out = reset_node(state)
    assert out["application_status"] == "collecting_data"
    assert out["loan_type"] is None
    assert out["national_id"] is None
    assert out["documents_uploaded"] is False
    assert out["document_result"] == {}
