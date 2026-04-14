#!/usr/bin/env python3
# api.py
#
# AXE Finance — FastAPI Backend
#
# Run:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
#
# This is a thin REST API wrapper around the LangGraph multi-agent system.
# State is stored in-memory by application_id. Each pipeline step is a
# synchronous endpoint that returns when done. The chat endpoint always
# works in parallel with pipeline endpoints (FastAPI handles concurrent
# requests natively).

import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from src.graph.orchestrator import app_graph
from src.models.global_state import GlobalState, get_fields_to_ask, get_status_summary

# Direct node imports for pipeline endpoints
from src.nodes.document_node import document_node
from src.nodes.scoring_node import scoring_node
from src.nodes.risk_assessment_node import risk_assessment_node
from src.nodes.decision_node import decision_node
from src.nodes.triage_node import triage_node
from src.nodes.policy_node import policy_node
from src.nodes.responder_node import responder_node


# ===============================================================
# FastAPI App
# ===============================================================
app = FastAPI(
    title="AXE Finance API",
    description="AI-powered loan origination system — REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================================================
# In-Memory State Store
# ===============================================================
# Maps application_id → GlobalState dict
# Lost on server restart. For production, swap with Redis.
APPLICATIONS: Dict[str, dict] = {}

# Maps application_id → list of progress messages from pipeline steps
PROGRESS_LOGS: Dict[str, List[Dict[str, Any]]] = {}


def get_application(app_id: str) -> dict:
    """Fetch an application or raise 404."""
    if app_id not in APPLICATIONS:
        raise HTTPException(status_code=404, detail=f"Application {app_id} not found")
    return APPLICATIONS[app_id]


def add_progress(app_id: str, message: str, level: str = "info"):
    """Add a progress message for an application."""
    if app_id not in PROGRESS_LOGS:
        PROGRESS_LOGS[app_id] = []
    PROGRESS_LOGS[app_id].append({
        "message": message,
        "level": level,
        "timestamp": datetime.now().isoformat(),
    })


# ===============================================================
# Pydantic Models (Request / Response)
# ===============================================================
class CreateApplicationResponse(BaseModel):
    application_id: str
    greeting: str
    created_at: str


class ChatMessageRequest(BaseModel):
    message: str


class ChatMessageResponse(BaseModel):
    response: str
    application_status: str
    intent: Optional[str] = None
    missing_fields: List[str] = []
    pipeline_ready: bool = False  # True if pipeline can be triggered


class DocumentUploadResponse(BaseModel):
    uploaded_files: List[str]
    application_status: str


class PipelineStepResponse(BaseModel):
    step: str
    completed: bool
    progress_messages: List[Dict[str, Any]] = []
    application_status: str
    next_step: Optional[str] = None


class StatusResponse(BaseModel):
    application_id: str
    application_status: str
    loan_type: Optional[str] = None
    loan_amount: Optional[float] = None
    compliance_tier: Optional[str] = None
    stage: str
    missing_fields: List[str] = []
    progress: Dict[str, bool] = {}
    summary: str


class FullStateResponse(BaseModel):
    state: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    progress_log: List[Dict[str, Any]]


# ===============================================================
# Helper: Initial state factory
# ===============================================================
def create_initial_state() -> dict:
    app_id = f"APP_{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now().isoformat()
    return {
        "application_id": app_id,
        "loan_type": None,
        "compliance_tier": None,
        "preferred_currency": "TND",
        "intent": None,
        "stage": "collecting",
        "application_status": "collecting_data",
        "rejection_reason": None,
        "credit_score_fetched": False,
        "documents_uploaded": False,
        "document_result": {},
        "scoring_result": {},
        "risk_result": {},
        "decision_result": {},
        "thought_steps": [],
        "messages": [],
        "last_response": "",
        "status_message": None,
        "created_at": now,
        "updated_at": now,
    }


def get_next_pipeline_step(state: dict) -> Optional[str]:
    """Determine the next pipeline step needed, or None if done."""
    status = state.get("application_status", "")
    if status in ("approved", "rejected", "blocked"):
        return None

    # HALT: if documents failed validation, the pipeline must pause and
    # wait for the user to re-upload or provide missing values via chat
    if status == "documents_incomplete":
        return None

    has_docs = state.get("documents_uploaded", False)
    has_doc_result = bool(state.get("document_result"))
    has_scoring = bool(state.get("scoring_result"))
    has_risk = bool(state.get("risk_result"))
    has_decision = bool(state.get("decision_result"))
    loan_type = state.get("loan_type")

    if has_docs and not has_doc_result:
        return "document"
    if has_doc_result and not has_scoring:
        return "scoring"
    if loan_type == "business" and has_scoring and not has_risk:
        return "risk_assessment"
    if has_scoring and not has_decision:
        if loan_type == "personal" or has_risk:
            return "decision"
    return None


def build_status_response(state: dict) -> StatusResponse:
    """Build a StatusResponse from state."""
    missing = get_fields_to_ask(state)
    loan_type = state.get("loan_type")

    progress = {
        "loan_type_identified": loan_type is not None and loan_type != "unknown",
        "data_collected": len(missing) == 0,
        "credit_score_fetched": state.get("credit_score_fetched", False),
        "documents_uploaded": state.get("documents_uploaded", False),
        "documents_processed": bool(state.get("document_result")),
        "scoring_complete": bool(state.get("scoring_result")),
        "decision_made": bool(state.get("decision_result")),
    }
    if loan_type == "business":
        progress["risk_assessed"] = bool(state.get("risk_result"))

    return StatusResponse(
        application_id=state.get("application_id", "N/A"),
        application_status=state.get("application_status", "collecting_data"),
        loan_type=loan_type,
        loan_amount=state.get("loan_amount"),
        compliance_tier=state.get("compliance_tier"),
        stage=state.get("stage", "collecting"),
        missing_fields=missing,
        progress=progress,
        summary=get_status_summary(state),
    )


# ===============================================================
# Endpoints — Application Lifecycle
# ===============================================================

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "service": "AXE Finance API",
        "version": "1.0.0",
        "status": "running",
        "active_applications": len(APPLICATIONS),
    }


@app.post("/applications", response_model=CreateApplicationResponse, tags=["Applications"])
def create_application():
    """Create a new loan application and return the greeting."""
    state = create_initial_state()
    app_id = state["application_id"]
    APPLICATIONS[app_id] = state
    PROGRESS_LOGS[app_id] = []

    greeting = (
        "Hello! Welcome to AXE Finance. Whether you're exploring your options "
        "or ready to apply for a loan, I'm here to help. Feel free to ask me "
        "anything about our loans and policies, or just tell me you want to "
        "get started. How can I help?"
    )

    return CreateApplicationResponse(
        application_id=app_id,
        greeting=greeting,
        created_at=state["created_at"],
    )


@app.delete("/applications/{app_id}", tags=["Applications"])
def delete_application(app_id: str):
    """Delete an application and clean up its files."""
    if app_id not in APPLICATIONS:
        raise HTTPException(status_code=404, detail="Application not found")

    # Clean up upload directory
    upload_dir = os.path.join(
        os.getenv("TEMP_UPLOADS_DIR", "./temp_uploads"),
        app_id,
    )
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)

    del APPLICATIONS[app_id]
    if app_id in PROGRESS_LOGS:
        del PROGRESS_LOGS[app_id]

    return {"deleted": True, "application_id": app_id}


@app.get("/applications/{app_id}", response_model=FullStateResponse, tags=["Applications"])
def get_application_state(app_id: str):
    """Get the full state of an application (for debugging)."""
    state = get_application(app_id)

    # Convert messages to simple format
    chat_history = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"role": "assistant", "content": msg.content})

    # Strip non-serializable fields
    serializable_state = {
        k: v for k, v in state.items() if k != "messages"
    }

    return FullStateResponse(
        state=serializable_state,
        chat_history=chat_history,
        progress_log=PROGRESS_LOGS.get(app_id, []),
    )


@app.get("/applications/{app_id}/status", response_model=StatusResponse, tags=["Applications"])
def get_status(app_id: str):
    """Get a concise status summary."""
    state = get_application(app_id)
    return build_status_response(state)


# ===============================================================
# Endpoint — Chat (always available, even during processing)
# ===============================================================

@app.post("/applications/{app_id}/messages", response_model=ChatMessageResponse, tags=["Chat"])
def send_message(app_id: str, request: ChatMessageRequest):
    """
    Send a chat message. The orchestrator handles routing.
    Works concurrently with pipeline endpoints — FastAPI handles
    concurrent requests independently.

    Special case: if status is 'documents_incomplete', try to extract
    missing critical field values from the user message BEFORE routing.
    If we successfully fill a missing field, re-validate and possibly
    unblock the pipeline.
    """
    state = get_application(app_id)

    try:
        # ---- Manual fallback: recover missing fields from chat ----
        if state.get("application_status") == "documents_incomplete":
            state = _try_recover_missing_fields(state, request.message)
            APPLICATIONS[app_id] = state

        # Build invoke input with new message
        invoke_input = dict(state)
        invoke_input["messages"] = list(state.get("messages", [])) + [HumanMessage(content=request.message)]
        invoke_input["updated_at"] = datetime.now().isoformat()

        # Run through the orchestrator
        result = app_graph.invoke(invoke_input)

        if isinstance(result, dict):
            APPLICATIONS[app_id] = result
            state = result

        response = state.get("last_response", "Something went wrong.")
        intent = state.get("intent")
        missing = get_fields_to_ask(state)
        pipeline_step = get_next_pipeline_step(state)

        return ChatMessageResponse(
            response=response,
            application_status=state.get("application_status", "collecting_data"),
            intent=intent,
            missing_fields=missing,
            pipeline_ready=pipeline_step is not None,
        )

    except Exception as e:
        import traceback
        print(f"[Chat Error] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


def _try_recover_missing_fields(state: dict, message: str) -> dict:
    """
    When documents are incomplete, try to extract missing critical field
    values from the user message. Uses simple regex — no LLM needed.
    
    Example: user types "my monthly income is 3169"
    → fills monthly_income = 3169
    → re-validates and possibly unblocks the pipeline
    """
    import re
    from src.models.global_state import validate_document_extraction, CRITICAL_FIELDS_BY_LOAN_TYPE

    loan_type = state.get("loan_type", "personal")
    critical = CRITICAL_FIELDS_BY_LOAN_TYPE.get(loan_type, {})

    # Extract the first number from the message
    num_match = re.search(r'(\d+(?:\.\d+)?)\s*(K|k|M|m)?', message)
    if not num_match:
        return state

    raw_num = float(num_match.group(1))
    mult = num_match.group(2)
    if mult in ("K", "k"):
        raw_num *= 1000
    elif mult in ("M", "m"):
        raw_num *= 1_000_000

    # Map keywords in the message to field names
    msg_lower = message.lower()
    field_keywords = {
        "monthly_income": ["income", "salary", "salaire", "revenu"],
        "monthly_cash_flow": ["cash flow", "cashflow", "flux", "trésorerie"],
        "total_revenue": ["revenue", "chiffre d'affaires", "ca", "ventes"],
        "net_income": ["net income", "profit", "bénéfice", "net profit"],
        "total_assets": ["total assets", "actifs"],
        "total_liabilities": ["liabilities", "passifs", "dettes"],
        "years_in_operation": ["years in operation", "years old", "since", "depuis"],
    }

    # Find which critical field matches the message
    filled_field = None
    for field in critical:
        if state.get(field) is not None:
            continue  # already filled
        keywords = field_keywords.get(field, [])
        if any(kw in msg_lower for kw in keywords):
            state[field] = raw_num
            filled_field = field
            break

    if filled_field:
        print(f"[Recovery] Filled {filled_field} = {raw_num} from chat")
        # Re-validate
        report = validate_document_extraction(state)
        state["document_validation"] = report
        if not report["has_issues"]:
            state["application_status"] = "documents_processed"
            print("[Recovery] All critical fields now present. Unblocking pipeline.")

    return state


# ===============================================================
# Endpoint — Document Upload
# ===============================================================

@app.post("/applications/{app_id}/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents(app_id: str, files: List[UploadFile] = File(...)):
    """Upload one or more document files for the application."""
    state = get_application(app_id)

    upload_dir = os.path.join(
        os.getenv("TEMP_UPLOADS_DIR", "./temp_uploads"),
        app_id,
    )
    os.makedirs(upload_dir, exist_ok=True)

    uploaded_names = []
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        uploaded_names.append(file.filename)

    state["documents_uploaded"] = True

    # If this is a RE-UPLOAD (previous validation failed), reset the pipeline
    # state so the documents get reprocessed from scratch
    if state.get("application_status") == "documents_incomplete":
        state["document_result"] = {}
        state["document_validation"] = {}
        state["application_status"] = "processing_documents"
        add_progress(app_id, "Re-upload detected. Reprocessing documents from scratch...", "info")

    APPLICATIONS[app_id] = state

    add_progress(app_id, f"Received {len(uploaded_names)} document(s): {', '.join(uploaded_names)}", "info")

    return DocumentUploadResponse(
        uploaded_files=uploaded_names,
        application_status=state.get("application_status", "collecting_data"),
    )


# ===============================================================
# Background Pipeline — runs all steps in a thread
# ===============================================================
import threading

# Tracks which applications have a pipeline currently running in the background
PIPELINE_THREADS: Dict[str, bool] = {}


def _run_full_pipeline_background(app_id: str):
    """
    Runs the entire pipeline (document → scoring → risk → decision) in a
    background thread. Updates APPLICATIONS dict and PROGRESS_LOGS as it goes.
    The client polls /status to see updates in real time.
    """
    try:
        PIPELINE_THREADS[app_id] = True

        while True:
            state = APPLICATIONS.get(app_id)
            if state is None:
                break

            next_step = get_next_pipeline_step(state)
            if next_step is None:
                break

            print(f"[Background Pipeline] {app_id}: running {next_step}")

            try:
                if next_step == "document":
                    result = document_node(state)
                    state.update(result)
                    APPLICATIONS[app_id] = state

                    # Generate progress messages
                    doc_result = state.get("document_result", {})
                    for fname, doc_data in doc_result.items():
                        doc_type = doc_data.get("type", "unknown")
                        extracted = doc_data.get("result", {})
                        if doc_type == "cin_card":
                            cin_id = extracted.get("cin_national_id", "N/A")
                            quality = extracted.get("extraction_quality", "unknown")
                            add_progress(app_id, f"CIN Card ({fname}): ID {cin_id} extracted. Quality: {quality}", "success")
                            typed_id = state.get("national_id")
                            if typed_id and cin_id and typed_id == cin_id:
                                add_progress(app_id, "CIN cross-check: ID matches. No fraud detected.", "success")
                            elif typed_id and cin_id and typed_id != cin_id:
                                add_progress(app_id, f"CIN MISMATCH: typed {typed_id} vs CIN {cin_id}", "error")
                        elif doc_type == "salary_slip":
                            net = extracted.get("net_salary", "N/A")
                            employer = extracted.get("employer_name", "N/A")
                            add_progress(app_id, f"Salary slip ({fname}): Net {net}, Employer {employer}", "success")
                        elif doc_type == "bank_statement":
                            cf = extracted.get("monthly_cash_flow", "N/A")
                            rent = extracted.get("monthly_rent_or_mortgage", "N/A")
                            debts = extracted.get("existing_debt_payments", "N/A")
                            add_progress(app_id, f"Bank statement ({fname}): Cash flow {cf}, Rent {rent}, Debts {debts}", "success")
                        elif doc_type == "income_statement":
                            rev = extracted.get("total_revenue", "N/A")
                            ni = extracted.get("net_income", "N/A")
                            add_progress(app_id, f"Income statement ({fname}): Revenue {rev}, Net {ni}", "success")
                        elif doc_type == "balance_sheet":
                            assets = extracted.get("total_assets", "N/A")
                            liab = extracted.get("total_liabilities", "N/A")
                            add_progress(app_id, f"Balance sheet ({fname}): Assets {assets}, Liabilities {liab}", "success")
                        elif doc_type == "unknown":
                            add_progress(app_id, f"Could not identify {fname}. Please check document type.", "error")
                        else:
                            add_progress(app_id, f"Document ({fname}): Classified as {doc_type}", "info")

                    # Check if validation caught any issues
                    if state.get("application_status") == "documents_incomplete":
                        report = state.get("document_validation", {})
                        add_progress(app_id, f"Validation failed: {report.get('user_message', 'Issues found')}", "error")
                        add_progress(app_id, "Pipeline paused. Waiting for re-upload or manual value input.", "warning")
                        break  # exit the pipeline loop — user must act
                    else:
                        add_progress(app_id, "All documents processed and validated.", "success")

                elif next_step == "scoring":
                    result = scoring_node(state)
                    state.update(result)
                    APPLICATIONS[app_id] = state

                    scoring = state.get("scoring_result", {})
                    dti = scoring.get("dti")
                    risk = scoring.get("risk_category", "unknown")
                    if dti is not None:
                        dti_pct = f"{dti:.1%}"
                        verdict = "PASS" if dti <= 0.45 else "FAIL"
                        add_progress(app_id, f"DTI calculated: {dti_pct} (threshold 45%) — {verdict}",
                                     "success" if dti <= 0.45 else "warning")
                    else:
                        add_progress(app_id, "DTI could not be calculated", "warning")
                    if risk != "unknown":
                        add_progress(app_id, f"ML risk category: {risk}", "info")

                elif next_step == "risk_assessment":
                    result = risk_assessment_node(state)
                    state.update(result)
                    APPLICATIONS[app_id] = state

                    risk = state.get("risk_result", {})
                    level = risk.get("risk_level", "unknown")
                    passed = risk.get("passed_count", 0)
                    total = risk.get("total_ratios", 0)
                    add_progress(app_id, f"Risk assessment: {passed}/{total} ratios passed. Level: {level.upper()}", "success")

                elif next_step == "decision":
                    result = decision_node(state)
                    state.update(result)
                    APPLICATIONS[app_id] = state

                    dec = state.get("decision_result", {})
                    decision = dec.get("decision", "unknown").upper()
                    reason = dec.get("reason", "")
                    if decision == "APPROVED":
                        rate = dec.get("interest_rate")
                        amount = dec.get("approved_amount")
                        rate_str = f"{rate:.1%}" if rate else "N/A"
                        add_progress(app_id, f"DECISION: APPROVED. Amount {amount}, Rate {rate_str}", "success")
                    elif decision == "CONDITIONAL":
                        conditions = dec.get("conditions", [])
                        add_progress(app_id, f"DECISION: CONDITIONAL. {', '.join(conditions[:2])}", "warning")
                    else:
                        add_progress(app_id, f"DECISION: REJECTED. {reason}", "error")

            except Exception as e:
                import traceback
                print(f"[Background Pipeline] {next_step} failed: {traceback.format_exc()}")
                add_progress(app_id, f"Error in {next_step}: {str(e)[:100]}", "error")
                break

    finally:
        PIPELINE_THREADS[app_id] = False
        print(f"[Background Pipeline] {app_id}: finished")


# ===============================================================
# Endpoint — Start Background Pipeline (returns immediately)
# ===============================================================

@app.post("/applications/{app_id}/start-pipeline", tags=["Pipeline"])
def start_pipeline(app_id: str):
    """
    Start the full processing pipeline in a background thread.
    Returns immediately. Client polls /status to see progress.
    """
    state = get_application(app_id)

    if not state.get("documents_uploaded"):
        raise HTTPException(status_code=400, detail="No documents uploaded")

    if PIPELINE_THREADS.get(app_id, False):
        return {"started": False, "reason": "Pipeline already running"}

    thread = threading.Thread(target=_run_full_pipeline_background, args=(app_id,), daemon=True)
    thread.start()

    return {"started": True, "application_id": app_id}


@app.get("/applications/{app_id}/pipeline-status", tags=["Pipeline"])
def pipeline_status(app_id: str):
    """Check if the background pipeline is still running."""
    get_application(app_id)  # 404 if not found
    return {
        "running": PIPELINE_THREADS.get(app_id, False),
        "progress_log": PROGRESS_LOGS.get(app_id, [])[-30:],
    }


# ===============================================================
# Endpoint — Pipeline Steps (synchronous, return when done)
# ===============================================================

@app.post("/applications/{app_id}/process-documents", response_model=PipelineStepResponse, tags=["Pipeline"])
def process_documents(app_id: str):
    """Run the document processing step. Synchronous — returns when done."""
    state = get_application(app_id)

    if not state.get("documents_uploaded"):
        raise HTTPException(status_code=400, detail="No documents uploaded")

    try:
        result = document_node(state)
        state.update(result)
        APPLICATIONS[app_id] = state

        # Generate progress messages from extracted data
        doc_result = state.get("document_result", {})
        for fname, doc_data in doc_result.items():
            doc_type = doc_data.get("type", "unknown")
            extracted = doc_data.get("result", {})

            if doc_type == "cin_card":
                cin_id = extracted.get("cin_national_id", "N/A")
                quality = extracted.get("extraction_quality", "unknown")
                add_progress(app_id, f"CIN Card ({fname}): ID {cin_id} extracted. Quality: {quality}", "success")

                typed_id = state.get("national_id")
                if typed_id and cin_id and typed_id == cin_id:
                    add_progress(app_id, "CIN cross-check: ID matches. No fraud detected.", "success")
                elif typed_id and cin_id and typed_id != cin_id:
                    add_progress(app_id, f"CIN cross-check: MISMATCH! Typed {typed_id} vs CIN {cin_id}", "error")

            elif doc_type == "salary_slip":
                net = extracted.get("net_salary", "N/A")
                employer = extracted.get("employer_name", "N/A")
                add_progress(app_id, f"Salary slip ({fname}): Net {net}, Employer {employer}", "success")

            elif doc_type == "bank_statement":
                cf = extracted.get("monthly_cash_flow", "N/A")
                rent = extracted.get("monthly_rent_or_mortgage", "N/A")
                debts = extracted.get("existing_debt_payments", "N/A")
                add_progress(app_id, f"Bank statement ({fname}): Cash flow {cf}, Rent {rent}, Debts {debts}", "success")

            elif doc_type == "income_statement":
                rev = extracted.get("total_revenue", "N/A")
                ni = extracted.get("net_income", "N/A")
                add_progress(app_id, f"Income statement ({fname}): Revenue {rev}, Net income {ni}", "success")

            elif doc_type == "balance_sheet":
                assets = extracted.get("total_assets", "N/A")
                liab = extracted.get("total_liabilities", "N/A")
                add_progress(app_id, f"Balance sheet ({fname}): Assets {assets}, Liabilities {liab}", "success")

            else:
                add_progress(app_id, f"Document ({fname}): Classified as {doc_type}", "info")

        add_progress(app_id, "All documents processed.", "success")

        return PipelineStepResponse(
            step="document",
            completed=True,
            progress_messages=PROGRESS_LOGS.get(app_id, [])[-15:],
            application_status=state.get("application_status", "collecting_data"),
            next_step=get_next_pipeline_step(state),
        )

    except Exception as e:
        import traceback
        print(f"[Document Processing Error] {traceback.format_exc()}")
        add_progress(app_id, f"Error: {str(e)[:100]}", "error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/applications/{app_id}/score", response_model=PipelineStepResponse, tags=["Pipeline"])
def run_scoring(app_id: str):
    """Run the financial scoring step. Synchronous."""
    state = get_application(app_id)

    if not state.get("document_result"):
        raise HTTPException(status_code=400, detail="Documents must be processed first")

    try:
        result = scoring_node(state)
        state.update(result)
        APPLICATIONS[app_id] = state

        scoring = state.get("scoring_result", {})
        dti = scoring.get("dti")
        risk = scoring.get("risk_category", "unknown")

        if dti is not None:
            dti_pct = f"{dti:.1%}"
            verdict = "PASS" if dti <= 0.45 else "FAIL"
            add_progress(app_id, f"DTI calculated: {dti_pct} (threshold 45%) — {verdict}", "success" if dti <= 0.45 else "warning")
        else:
            add_progress(app_id, "DTI could not be calculated (missing data)", "warning")

        if risk != "unknown":
            add_progress(app_id, f"ML risk category: {risk}", "info")
        else:
            add_progress(app_id, "ML scoring API unavailable", "warning")

        return PipelineStepResponse(
            step="scoring",
            completed=True,
            progress_messages=PROGRESS_LOGS.get(app_id, [])[-15:],
            application_status=state.get("application_status", "collecting_data"),
            next_step=get_next_pipeline_step(state),
        )

    except Exception as e:
        import traceback
        print(f"[Scoring Error] {traceback.format_exc()}")
        add_progress(app_id, f"Error: {str(e)[:100]}", "error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/applications/{app_id}/risk-assessment", response_model=PipelineStepResponse, tags=["Pipeline"])
def run_risk_assessment(app_id: str):
    """Run risk assessment for business loans. Synchronous."""
    state = get_application(app_id)

    if state.get("loan_type") != "business":
        raise HTTPException(status_code=400, detail="Risk assessment is only for business loans")

    if not state.get("scoring_result"):
        raise HTTPException(status_code=400, detail="Scoring must be completed first")

    try:
        result = risk_assessment_node(state)
        state.update(result)
        APPLICATIONS[app_id] = state

        risk = state.get("risk_result", {})
        level = risk.get("risk_level", "unknown")
        passed = risk.get("passed_count", 0)
        total = risk.get("total_ratios", 0)
        add_progress(app_id, f"Risk assessment: {passed}/{total} ratios passed. Level: {level.upper()}", "success")

        return PipelineStepResponse(
            step="risk_assessment",
            completed=True,
            progress_messages=PROGRESS_LOGS.get(app_id, [])[-15:],
            application_status=state.get("application_status", "collecting_data"),
            next_step=get_next_pipeline_step(state),
        )

    except Exception as e:
        import traceback
        print(f"[Risk Assessment Error] {traceback.format_exc()}")
        add_progress(app_id, f"Error: {str(e)[:100]}", "error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/applications/{app_id}/decide", response_model=PipelineStepResponse, tags=["Pipeline"])
def make_decision(app_id: str):
    """Run the final decision step. Synchronous."""
    state = get_application(app_id)

    if not state.get("scoring_result"):
        raise HTTPException(status_code=400, detail="Scoring must be completed first")

    try:
        result = decision_node(state)
        state.update(result)
        APPLICATIONS[app_id] = state

        dec = state.get("decision_result", {})
        decision = dec.get("decision", "unknown").upper()
        reason = dec.get("reason", "")

        if decision == "APPROVED":
            rate = dec.get("interest_rate")
            amount = dec.get("approved_amount")
            rate_str = f"{rate:.1%}" if rate else "N/A"
            add_progress(app_id, f"DECISION: APPROVED. Amount {amount}, Rate {rate_str}", "success")
        elif decision == "CONDITIONAL":
            conditions = dec.get("conditions", [])
            add_progress(app_id, f"DECISION: CONDITIONAL. {', '.join(conditions[:2])}", "warning")
        else:
            add_progress(app_id, f"DECISION: REJECTED. {reason}", "error")

        return PipelineStepResponse(
            step="decision",
            completed=True,
            progress_messages=PROGRESS_LOGS.get(app_id, [])[-15:],
            application_status=state.get("application_status", "collecting_data"),
            next_step=get_next_pipeline_step(state),
        )

    except Exception as e:
        import traceback
        print(f"[Decision Error] {traceback.format_exc()}")
        add_progress(app_id, f"Error: {str(e)[:100]}", "error")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================================================
# Endpoint — Decision Letter Download
# ===============================================================

@app.get("/applications/{app_id}/letter", tags=["Documents"])
def download_letter(app_id: str):
    """Download the generated decision letter (PDF or HTML)."""
    state = get_application(app_id)
    decision = state.get("decision_result", {})

    letter_path = decision.get("letter_pdf_path") or decision.get("letter_html_path")
    if not letter_path or not os.path.exists(letter_path):
        raise HTTPException(status_code=404, detail="Decision letter not found")

    ext = "pdf" if letter_path.endswith(".pdf") else "html"
    media_type = "application/pdf" if ext == "pdf" else "text/html"

    return FileResponse(
        letter_path,
        media_type=media_type,
        filename=f"{app_id}_decision.{ext}",
    )


# ===============================================================
# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
# ===============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)