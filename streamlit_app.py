#!/usr/bin/env python3
# streamlit_app.py
#
# AXE Finance — Streamlit Testing Interface
# Run:  streamlit run streamlit_app.py
#
# NO THREADS. Pipeline steps run one at a time via st.rerun().
# Chat input is always responsive.

import os
import uuid
import shutil
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from src.graph.orchestrator import app_graph
from src.models.global_state import GlobalState, get_fields_to_ask, get_status_summary


# ---------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AXE Finance — AI Loan Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .status-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 16px; font-size: 13px; font-weight: 500;
    }
    .status-collecting  { background: #FFF3E0; color: #E65100; }
    .status-processing  { background: #E3F2FD; color: #1565C0; }
    .status-approved    { background: #E8F5E9; color: #2E7D32; }
    .status-rejected    { background: #FFEBEE; color: #C62828; }
    .thought-box {
        background: #FAFAFA; border-left: 3px solid #90CAF9;
        padding: 6px 10px; margin: 3px 0; font-size: 12px; color: #555;
        border-radius: 0 4px 4px 0;
    }
    .processing-banner {
        background: #E3F2FD; border: 1px solid #90CAF9;
        border-radius: 8px; padding: 10px 14px; margin: 8px 0;
        font-size: 14px; color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# State Init
# ---------------------------------------------------------------
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


if "state" not in st.session_state:
    st.session_state.state = create_initial_state()
    st.session_state.chat_history = []
    st.session_state.debug_mode = False
    st.session_state.pipeline_running = False  # Flag: auto-continue pipeline


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def get_status_color(status: str) -> str:
    if status in ("approved",):
        return "status-approved"
    elif status in ("rejected", "blocked"):
        return "status-rejected"
    elif status in ("collecting_data", "awaiting_documents"):
        return "status-collecting"
    return "status-processing"


def get_upload_dir() -> str:
    base = os.getenv("TEMP_UPLOADS_DIR", "./temp_uploads")
    app_id = st.session_state.state.get("application_id", "UNKNOWN")
    app_dir = os.path.join(base, app_id)
    os.makedirs(app_dir, exist_ok=True)
    return app_dir


def needs_pipeline_step(state: dict) -> bool:
    """Check if the pipeline has pending automatic work to do."""
    status = state.get("application_status", "")
    if status in ("approved", "rejected", "blocked"):
        return False

    has_docs = state.get("documents_uploaded", False)
    has_doc_result = bool(state.get("document_result"))
    has_scoring = bool(state.get("scoring_result"))
    has_risk = bool(state.get("risk_result"))
    has_decision = bool(state.get("decision_result"))
    loan_type = state.get("loan_type")

    if has_docs and not has_doc_result:
        return True
    if has_doc_result and not has_scoring:
        return True
    if loan_type == "business" and has_scoring and not has_risk:
        return True
    if has_scoring and not has_decision:
        if loan_type == "personal":
            return True
        elif has_risk:
            return True

    return False


def run_one_pipeline_step():
    """Run exactly ONE pipeline step, then return."""
    state = st.session_state.state

    # Determine what step is about to run (for logging)
    has_docs = state.get("documents_uploaded", False)
    has_doc_result = bool(state.get("document_result"))
    has_scoring = bool(state.get("scoring_result"))

    if has_docs and not has_doc_result:
        step_name = "Document processing"
    elif has_doc_result and not has_scoring:
        step_name = "Financial scoring"
    elif has_scoring and not state.get("risk_result") and state.get("loan_type") == "business":
        step_name = "Risk assessment"
    else:
        step_name = "Decision"

    print(f"[Pipeline] Running step: {step_name}")

    # Make sure there's a message for the graph to process
    msgs = state.get("messages", [])
    if not msgs or not isinstance(msgs[-1], HumanMessage):
        msgs = list(msgs) + [HumanMessage(content="Continue processing my application")]
        state["messages"] = msgs

    try:
        invoke_input = dict(state)
        result = app_graph.invoke(invoke_input)

        if isinstance(result, dict):
            st.session_state.state = result
            print(f"[Pipeline] {step_name} completed. Status: {result.get('application_status')}")

            response = result.get("last_response", "")
            if response and response.strip():
                if (not st.session_state.chat_history or
                        st.session_state.chat_history[-1] != ("assistant", response)):
                    st.session_state.chat_history.append(("assistant", response))
        else:
            print(f"[Pipeline] {step_name} returned non-dict: {type(result)}")
            st.session_state.pipeline_running = False

    except Exception as e:
        import traceback
        print(f"[Pipeline] {step_name} FAILED:\n{traceback.format_exc()}")
        # Add error to chat so user sees it
        st.session_state.chat_history.append(
            ("assistant", f"⚠️ {step_name} encountered an error: {str(e)[:100]}. Please try again.")
        )
        st.session_state.pipeline_running = False


def process_message(user_input: str):
    """Process a chat message from the user."""
    state = st.session_state.state

    st.session_state.chat_history.append(("user", user_input))

    try:
        invoke_input = dict(state)
        invoke_input["messages"] = list(state.get("messages", [])) + [HumanMessage(content=user_input)]
        invoke_input["updated_at"] = datetime.now().isoformat()

        result = app_graph.invoke(invoke_input)

        if isinstance(result, dict):
            st.session_state.state = result

        response = result.get("last_response", "Something went wrong.")
        st.session_state.chat_history.append(("assistant", response))

        # Check if this triggered the pipeline (e.g., docs uploaded)
        if needs_pipeline_step(st.session_state.state):
            st.session_state.pipeline_running = True

    except Exception as e:
        import traceback
        print(f"[Chat Error]\n{traceback.format_exc()}")
        st.session_state.chat_history.append(("assistant", f"⚠️ Error: {str(e)[:100]}"))
        # Save the message even if graph failed
        state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=user_input)]
        st.session_state.state = state


def handle_document_upload(uploaded_files):
    """Save files, give instant feedback, trigger pipeline."""
    if not uploaded_files:
        return

    # Guard: don't re-upload if already uploaded
    if st.session_state.state.get("documents_uploaded") and st.session_state.pipeline_running:
        st.toast("Documents already uploaded and processing!")
        return

    upload_dir = get_upload_dir()
    file_names = []

    for f in uploaded_files:
        path = os.path.join(upload_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        file_names.append(f.name)

    st.session_state.state["documents_uploaded"] = True

    st.session_state.chat_history.append(
        ("user", f"I've uploaded: {', '.join(file_names)}")
    )
    st.session_state.chat_history.append(
        ("assistant",
         f"Got it! I received {len(file_names)} document(s): {', '.join(file_names)}. "
         f"Processing now — you'll see each step complete in the sidebar.")
    )

    st.session_state.pipeline_running = True


def reset_application():
    old_dir = get_upload_dir()
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir, ignore_errors=True)
    st.session_state.state = create_initial_state()
    st.session_state.chat_history = []
    st.session_state.pipeline_running = False


# ---------------------------------------------------------------
# Run ONE pipeline step if needed (before rendering UI)
# ---------------------------------------------------------------
if st.session_state.pipeline_running and needs_pipeline_step(st.session_state.state):
    # Figure out step name for spinner
    _s = st.session_state.state
    if _s.get("documents_uploaded") and not _s.get("document_result"):
        _step = "Processing documents..."
    elif _s.get("document_result") and not _s.get("scoring_result"):
        _step = "Calculating financial ratios..."
    elif _s.get("loan_type") == "business" and _s.get("scoring_result") and not _s.get("risk_result"):
        _step = "Running risk assessment..."
    else:
        _step = "Making final decision..."

    with st.spinner(_step):
        run_one_pipeline_step()

    if needs_pipeline_step(st.session_state.state):
        st.rerun()
    else:
        st.session_state.pipeline_running = False
        st.rerun()


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
state = st.session_state.state

with st.sidebar:
    st.markdown("### 🏦 AXE Finance")
    st.caption("AI Loan Assistant")
    st.divider()

    app_id = state.get("application_id", "N/A")
    status = state.get("application_status", "collecting_data")
    loan_type = state.get("loan_type")

    st.markdown(f"**Application:** `{app_id}`")
    status_label = status.replace("_", " ").title()
    color = get_status_color(status)
    st.markdown(f'<span class="status-badge {color}">{status_label}</span>', unsafe_allow_html=True)

    if loan_type:
        st.markdown(f"**Type:** {loan_type.title()}")
    if state.get("loan_amount"):
        try:
            amt = float(state["loan_amount"])
            st.markdown(f"**Amount:** {amt:,.0f} TND")
        except (ValueError, TypeError):
            st.markdown(f"**Amount:** {state['loan_amount']} TND")
    if state.get("compliance_tier"):
        st.markdown(f"**Tier:** {state['compliance_tier']}")

    if st.session_state.pipeline_running:
        st.markdown('<div class="processing-banner">⏳ Pipeline running...</div>', unsafe_allow_html=True)

    st.divider()

    # Progress
    st.markdown("**Progress**")
    missing = get_fields_to_ask(state)
    checks = [
        ("Loan type identified", loan_type is not None and loan_type != "unknown"),
        ("Data collected", len(missing) == 0),
        ("Credit score", state.get("credit_score_fetched", False)),
        ("Documents uploaded", state.get("documents_uploaded", False)),
        ("Documents processed", bool(state.get("document_result"))),
        ("Scoring complete", bool(state.get("scoring_result"))),
    ]
    if loan_type == "business":
        checks.append(("Risk assessed", bool(state.get("risk_result"))))
    checks.append(("Decision made", bool(state.get("decision_result"))))

    for label, done in checks:
        st.markdown(f"{'✅' if done else '⬜'} {label}")

    if missing:
        st.caption(f"Next: {missing[0].replace('_', ' ')}")

    st.divider()

    # Chain of thought
    st.markdown("**Chain of thought**")
    thoughts = state.get("thought_steps", [])
    thought_container = st.container(height=220)
    with thought_container:
        if thoughts:
            for i, t in enumerate(thoughts):
                st.markdown(f'<div class="thought-box">{i+1}. {t}</div>', unsafe_allow_html=True)
        else:
            st.caption("No thoughts yet.")

    st.divider()

    # Upload
    st.markdown("**Upload documents**")
    fields_done = len(get_fields_to_ask(state)) == 0

    if not fields_done:
        st.caption("Complete the application form first, then upload documents.")
    else:
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True,
            key="doc_upload",
            label_visibility="collapsed",
        )
        if uploaded_files:
            if st.button("📤 Upload and process", type="primary", use_container_width=True):
                handle_document_upload(uploaded_files)
                st.rerun()

    # Decision letter
    decision = state.get("decision_result", {})
    letter_path = decision.get("letter_pdf_path") or decision.get("letter_html_path")
    if letter_path and os.path.exists(letter_path):
        st.divider()
        with open(letter_path, "rb") as f:
            ext = "pdf" if letter_path.endswith(".pdf") else "html"
            st.download_button(
                f"📄 Decision letter (.{ext})",
                data=f.read(),
                file_name=f"{app_id}_decision.{ext}",
                mime=f"application/{ext}",
                use_container_width=True,
            )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New", use_container_width=True):
            reset_application()
            st.rerun()
    with col2:
        st.session_state.debug_mode = st.toggle("Debug", value=st.session_state.debug_mode)


# ---------------------------------------------------------------
# Main Chat
# ---------------------------------------------------------------
st.markdown("## 🏦 AXE Finance — AI Loan Assistant")

if not st.session_state.chat_history:
    st.session_state.chat_history.append(
        ("assistant",
         "Hello! Welcome to AXE Finance. I'm your AI loan assistant. "
         "What type of loan are you looking for — personal or business?")
    )

for role, content in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant", avatar="🏦"):
            st.write(content)

# Chat input — always available
user_input = st.chat_input("Type your message...")
if user_input:
    process_message(user_input)
    st.rerun()


# ---------------------------------------------------------------
# Debug Panel
# ---------------------------------------------------------------
if st.session_state.debug_mode:
    st.divider()
    tab1, tab2, tab3 = st.tabs(["State", "Scoring & Risk", "Decision"])

    with tab1:
        display = {k: v for k, v in state.items() if k not in ("messages", "thought_steps")}
        st.json(display)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scoring**")
            s = state.get("scoring_result")
            st.json(s) if s else st.caption("Not yet")
        with c2:
            st.markdown("**Risk**")
            r = state.get("risk_result")
            if r:
                st.json({
                    "risk_level": r.get("risk_level"),
                    "recommendation": r.get("recommendation"),
                    "passed": r.get("passed_count"),
                    "failed": r.get("failed_count"),
                })
            else:
                st.caption("Not yet")

    with tab3:
        d = state.get("decision_result")
        if d:
            st.json({k: v for k, v in d.items() if k != "explanation"})
            if d.get("explanation"):
                st.text_area("Explanation", d["explanation"], height=120, disabled=True)
        else:
            st.caption("No decision yet")