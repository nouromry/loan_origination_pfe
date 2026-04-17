from langchain_core.messages import HumanMessage
import os


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
        "document_validation": {"has_issues": True},
        "processed_files": ["cin.pdf:1000"],
        "messages": [HumanMessage(content="start over")],
    }
    out = reset_node(state)
    assert out["application_status"] == "collecting"
    assert out["loan_type"] is None
    assert out["national_id"] is None
    assert out["documents_uploaded"] is False
    assert out["document_result"] == {}
    assert out["document_validation"] == {}
    assert out["processed_files"] == []


def test_document_node_keeps_prior_results(tmp_path, monkeypatch):
    from src.nodes import document_node as dn

    app_id = "APP_INC_1"
    app_dir = tmp_path / app_id
    app_dir.mkdir()
    cin_path = app_dir / "cin.pdf"
    salary_path = app_dir / "salary.pdf"
    cin_path.write_bytes(b"old-cin-content")
    salary_path.write_bytes(b"new-salary-content")

    cin_key = f"cin.pdf:{cin_path.stat().st_size}"
    salary_key = f"salary.pdf:{salary_path.stat().st_size}"

    monkeypatch.setenv("TEMP_UPLOADS_DIR", str(tmp_path))

    class _Tool:
        def __init__(self, fn):
            self._fn = fn

        def invoke(self, payload):
            return self._fn(payload)

    def fake_extract_text(payload):
        file_name = os.path.basename(payload["file_path"])
        return {"success": True, "text": f"content for {file_name} " + ("x" * 80)}

    monkeypatch.setattr(dn, "extract_text", _Tool(fake_extract_text))
    monkeypatch.setattr(dn, "ocr_extract_text", _Tool(lambda payload: {"success": False, "error": "unused"}))
    monkeypatch.setattr(
        dn,
        "classify_document_type",
        _Tool(lambda payload: "salary_slip" if payload["file_name"] == "salary.pdf" else "cin_card"),
    )
    monkeypatch.setattr(dn, "parse_cin_card", _Tool(lambda payload: {"cin_national_id": "12714074"}))
    monkeypatch.setattr(dn, "extract_tables_as_markdown", _Tool(lambda payload: {"success": False, "has_tables": False}))
    monkeypatch.setattr(dn, "cross_check_cin", _Tool(lambda payload: {"fraud_flag": False, "dob_match": True}))

    class _FakeDocumentAgent:
        def run(self, **kwargs):
            return {"net_salary": 3169}

    monkeypatch.setattr(dn, "DocumentAgent", _FakeDocumentAgent)

    import src.models.global_state as gs
    monkeypatch.setattr(
        gs,
        "validate_document_extraction",
        lambda state: {
            "has_issues": False,
            "failed_documents": [],
            "unknown_documents": [],
            "missing_critical_fields": [],
            "missing_required_documents": [],
            "user_message": "ok",
        },
    )

    state = {
        "application_id": app_id,
        "loan_type": "personal",
        "thought_steps": [],
        "processed_files": [cin_key],
        "document_result": {"cin.pdf": {"type": "cin_card", "result": {"cin_national_id": "12714074"}}},
    }

    out = dn.document_node(state)

    assert "cin.pdf" in out["document_result"]
    assert "salary.pdf" in out["document_result"]
    assert out["document_result"]["salary.pdf"]["type"] == "salary_slip"
    assert cin_key in out["processed_files"]
    assert salary_key in out["processed_files"]


def test_document_node_reupload_reprocesses(tmp_path, monkeypatch):
    from src.nodes import document_node as dn

    app_id = "APP_REUP_1"
    app_dir = tmp_path / app_id
    app_dir.mkdir()
    salary_path = app_dir / "salary.pdf"
    salary_path.write_bytes(b"new-content-with-different-size")

    new_key = f"salary.pdf:{salary_path.stat().st_size}"
    old_key = "salary.pdf:3"

    monkeypatch.setenv("TEMP_UPLOADS_DIR", str(tmp_path))

    class _Tool:
        def __init__(self, fn):
            self._fn = fn

        def invoke(self, payload):
            return self._fn(payload)

    monkeypatch.setattr(dn, "extract_text", _Tool(lambda payload: {"success": True, "text": "x" * 120}))
    monkeypatch.setattr(dn, "ocr_extract_text", _Tool(lambda payload: {"success": False, "error": "unused"}))
    monkeypatch.setattr(dn, "classify_document_type", _Tool(lambda payload: "salary_slip"))
    monkeypatch.setattr(dn, "extract_tables_as_markdown", _Tool(lambda payload: {"success": False, "has_tables": False}))
    monkeypatch.setattr(dn, "cross_check_cin", _Tool(lambda payload: {"fraud_flag": False, "dob_match": True}))

    class _FakeDocumentAgent:
        def run(self, **kwargs):
            return {"net_salary": 4500}

    monkeypatch.setattr(dn, "DocumentAgent", _FakeDocumentAgent)

    import src.models.global_state as gs
    monkeypatch.setattr(
        gs,
        "validate_document_extraction",
        lambda state: {
            "has_issues": False,
            "failed_documents": [],
            "unknown_documents": [],
            "missing_critical_fields": [],
            "missing_required_documents": [],
            "user_message": "ok",
        },
    )

    state = {
        "application_id": app_id,
        "loan_type": "personal",
        "thought_steps": [],
        "processed_files": [old_key],
        "document_result": {"salary.pdf": {"type": "salary_slip", "result": {"net_salary": 1200}}},
    }

    out = dn.document_node(state)

    # Replacement keeps only the latest filename entry and latest processed key.
    assert out["document_result"]["salary.pdf"]["result"]["net_salary"] == 4500
    assert set(out["document_result"].keys()) == {"salary.pdf"}
    assert old_key not in out["processed_files"]
    assert new_key in out["processed_files"]


def test_orchestrator_routes_to_document_for_unprocessed_incremental_upload(tmp_path, monkeypatch):
    from src.graph.orchestrator import route

    app_id = "APP_ROUTE_DOC_1"
    app_dir = tmp_path / app_id
    app_dir.mkdir()
    cin_path = app_dir / "cin.pdf"
    cin_path.write_bytes(b"cin-file-content")
    file_key = f"cin.pdf:{cin_path.stat().st_size}"

    monkeypatch.setenv("TEMP_UPLOADS_DIR", str(tmp_path))

    state = {
        "application_id": app_id,
        "intent": "credit_workflow",
        "documents_uploaded": True,
        "processed_files": [],
        "document_result": {},
        "loan_type": None,
        "loan_amount": None,
        "loan_term_months": None,
        "national_id": None,
    }
    assert route(state) == "document"

    # Once marked processed, route should no longer force document node
    state["processed_files"] = [file_key]
    assert route(state) == "collect"


def test_orchestrator_needs_correction_can_route_to_collect_for_chat_recovery():
    from src.graph.orchestrator import route

    state = {
        "application_status": "needs_correction",
        "intent": "credit_workflow",
        "loan_type": "personal",
        "loan_amount": None,
        "loan_term_months": 24,
        "national_id": "12714074",
        "messages": [HumanMessage(content="my monthly income is 3169")],
    }
    assert route(state) == "collect"


def test_orchestrator_needs_correction_routes_to_responder_without_extractable_chat():
    from src.graph.orchestrator import route

    state = {
        "application_status": "needs_correction",
        "intent": "credit_workflow",
        "loan_type": "personal",
        "loan_amount": None,
        "loan_term_months": 24,
        "national_id": "12714074",
        "messages": [HumanMessage(content="okay thanks")],
    }
    assert route(state) == "responder"


def test_document_node_cin_mismatch_sets_needs_correction_not_blocked(tmp_path, monkeypatch):
    from src.nodes import document_node as dn

    app_id = "APP_MISMATCH_1"
    app_dir = tmp_path / app_id
    app_dir.mkdir()
    cin_path = app_dir / "cin.pdf"
    cin_path.write_bytes(b"cin-content")

    monkeypatch.setenv("TEMP_UPLOADS_DIR", str(tmp_path))

    class _Tool:
        def __init__(self, fn):
            self._fn = fn

        def invoke(self, payload):
            return self._fn(payload)

    monkeypatch.setattr(dn, "extract_text", _Tool(lambda payload: {"success": True, "text": "x" * 120}))
    monkeypatch.setattr(dn, "ocr_extract_text", _Tool(lambda payload: {"success": False, "error": "unused"}))
    monkeypatch.setattr(dn, "classify_document_type", _Tool(lambda payload: "cin_card"))
    monkeypatch.setattr(
        dn, "parse_cin_card", _Tool(lambda payload: {"cin_national_id": "12714074", "fields_extracted": 4, "extraction_quality": "good"})
    )
    monkeypatch.setattr(
        dn,
        "cross_check_cin",
        _Tool(lambda payload: {"id_match": False, "fraud_flag": True, "mismatches": ["National ID mismatch"]}),
    )

    import src.models.global_state as gs
    monkeypatch.setattr(
        gs,
        "validate_document_extraction",
        lambda state: {
            "has_issues": False,
            "failed_documents": [],
            "unknown_documents": [],
            "missing_critical_fields": [],
            "missing_required_documents": [],
            "user_message": "ok",
        },
    )

    state = {
        "application_id": app_id,
        "loan_type": "personal",
        "national_id": "12567898",
        "thought_steps": [],
        "processed_files": [],
        "document_result": {},
    }
    out = dn.document_node(state)
    assert out["application_status"] == "needs_correction"
    assert out.get("identity_mismatch")
    assert out.get("rejection_reason") is None


def test_collect_node_resolves_identity_mismatch_using_document_id():
    from src.nodes.collect_node import collect_node

    state = {
        "application_status": "needs_correction",
        "identity_mismatch": {"typed_national_id": "12567898", "cin_national_id": "12714074"},
        "cin_national_id": "12714074",
        "loan_type": "personal",
        "monthly_income": 3169,
        "monthly_cash_flow": 1999,
        "messages": [HumanMessage(content="Use document ID please")],
        "thought_steps": [],
    }

    out = collect_node(state)
    assert out["national_id"] == "12714074"
    assert out["application_status"] == "ready_for_scoring"
    assert out["identity_mismatch"] == {}
