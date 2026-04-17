#!/usr/bin/env python3
# streamlit_app.py
#
# AXE Finance — Streamlit Client (Pure REST Client)
#
# THIN CLIENT. No business logic. All processing happens in the FastAPI
# backend (api.py). The pipeline runs in a background THREAD on the API
# server, so the client only POLLS for status updates. The chat input
# stays interactive at all times.
#
# Run:
#   1. uvicorn api:app --reload --port 8000
#   2. streamlit run streamlit_app.py

import os
import time
import requests
import streamlit as st

# ===============================================================
# Config
# ===============================================================
API_BASE_URL = os.getenv("AXE_API_URL", "http://localhost:8000")
POLL_INTERVAL = 2  # seconds between pipeline status polls

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
    .status-collecting { background: #FFF3E0; color: #E65100; }
    .status-processing { background: #E3F2FD; color: #1565C0; }
    .status-approved   { background: #E8F5E9; color: #2E7D32; }
    .status-rejected   { background: #FFEBEE; color: #C62828; }
    .progress-msg {
        background: #F1F8E9; border-left: 3px solid #8BC34A;
        padding: 8px 12px; margin: 4px 0; font-size: 13px; color: #33691E;
        border-radius: 0 6px 6px 0;
    }
    .progress-warn {
        background: #FFF3E0; border-left: 3px solid #FF9800;
        padding: 8px 12px; margin: 4px 0; font-size: 13px; color: #E65100;
        border-radius: 0 6px 6px 0;
    }
    .progress-err {
        background: #FFEBEE; border-left: 3px solid #F44336;
        padding: 8px 12px; margin: 4px 0; font-size: 13px; color: #C62828;
        border-radius: 0 6px 6px 0;
    }
    .processing-banner {
        background: #E3F2FD; border: 1px solid #90CAF9;
        border-radius: 8px; padding: 10px 14px; margin: 8px 0;
        font-size: 14px; color: #1565C0;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)


# ===============================================================
# API Client
# ===============================================================
class APIError(Exception):
    pass


def api_post(path: str, json=None, files=None, timeout: int = 30) -> dict:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json, files=files, timeout=timeout)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"API error {resp.status_code}: {detail}")
        return resp.json()
    except requests.RequestException as e:
        raise APIError(f"Cannot reach API: {e}")


def api_get(path: str, timeout: int = 10) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"API error {resp.status_code}: {detail}")
        return resp.json()
    except requests.RequestException as e:
        raise APIError(f"Cannot reach API: {e}")


def api_delete(path: str) -> dict:
    try:
        resp = requests.delete(f"{API_BASE_URL}{path}", timeout=10)
        return resp.json() if resp.status_code < 400 else {"deleted": False}
    except requests.RequestException:
        return {"deleted": False}


def api_get_file(path: str) -> bytes:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", timeout=10)
        return resp.content if resp.status_code < 400 else None
    except requests.RequestException:
        return None


# ===============================================================
# High-Level API Methods
# ===============================================================
def create_new_application() -> dict:
    return api_post("/applications")


def send_chat_message(app_id: str, message: str) -> dict:
    return api_post(f"/applications/{app_id}/messages", json={"message": message}, timeout=60)


def upload_documents(app_id: str, files) -> dict:
    files_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    return api_post(f"/applications/{app_id}/documents", files=files_data)


def start_pipeline(app_id: str) -> dict:
    """Triggers the background pipeline. Returns immediately."""
    return api_post(f"/applications/{app_id}/start-pipeline")


def get_pipeline_status(app_id: str) -> dict:
    """Poll the background pipeline status. Fast call (~50ms)."""
    return api_get(f"/applications/{app_id}/pipeline-status")


def get_status(app_id: str) -> dict:
    return api_get(f"/applications/{app_id}/status")


def get_full_state(app_id: str) -> dict:
    return api_get(f"/applications/{app_id}")


def delete_application(app_id: str) -> dict:
    return api_delete(f"/applications/{app_id}")


def download_letter(app_id: str) -> bytes:
    return api_get_file(f"/applications/{app_id}/letter")


# ===============================================================
# Session State
# ===============================================================
if "app_id" not in st.session_state:
    st.session_state.app_id = None
    st.session_state.chat_history = []  # list of (role, content)
    st.session_state.last_status = {}
    st.session_state.pipeline_running = False
    st.session_state.seen_progress_count = 0  # how many progress messages we've shown
    st.session_state.api_available = None
    st.session_state.debug_mode = False
    st.session_state.last_pushed_response = ""


def check_api():
    try:
        api_get("/")
        return True
    except APIError:
        return False


# ===============================================================
# UI Helpers
# ===============================================================
def get_status_color(status: str) -> str:
    if status == "approved":
        return "status-approved"
    elif status in ("rejected", "blocked"):
        return "status-rejected"
    elif status in ("collecting_data", "awaiting_documents"):
        return "status-collecting"
    return "status-processing"


def init_application():
    try:
        result = create_new_application()
        st.session_state.app_id = result["application_id"]
        st.session_state.chat_history = [("assistant", result["greeting"])]
        st.session_state.last_pushed_response = result["greeting"]
        st.session_state.pipeline_running = False
        st.session_state.seen_progress_count = 0
        st.session_state.last_status = get_status(st.session_state.app_id)
    except APIError as e:
        st.error(f"Failed to create application: {e}")


def reset_application():
    if st.session_state.app_id:
        delete_application(st.session_state.app_id)
    st.session_state.app_id = None
    st.session_state.chat_history = []
    st.session_state.last_pushed_response = ""
    st.session_state.last_status = {}
    st.session_state.pipeline_running = False
    st.session_state.seen_progress_count = 0
    init_application()


def handle_user_message(message: str):
    st.session_state.chat_history.append(("user", message))
    try:
        result = send_chat_message(st.session_state.app_id, message)
        st.session_state.chat_history.append(("assistant", result["response"]))
        st.session_state.last_pushed_response = result["response"]
        st.session_state.last_status = get_status(st.session_state.app_id)
    except APIError as e:
        st.session_state.chat_history.append(("assistant", f"Error: {str(e)[:120]}"))


def handle_upload(uploaded_files):
    try:
        result = upload_documents(st.session_state.app_id, uploaded_files)
        names = ", ".join(result["uploaded_files"])
        st.session_state.chat_history.append(("user", f"Uploaded: {names}"))
        st.session_state.chat_history.append(
            ("assistant", f"Got it! Processing {len(result['uploaded_files'])} document(s) in the background. "
                          f"You can keep chatting while I work on it!")
        )
        st.session_state.last_status = get_status(st.session_state.app_id)

        # Start the background pipeline (returns immediately)
        start_pipeline(st.session_state.app_id)
        st.session_state.pipeline_running = True
        st.session_state.seen_progress_count = len(st.session_state.chat_history)

    except APIError as e:
        st.error(f"Upload failed: {e}")


def poll_pipeline():
    """
    Poll pipeline status. Adds new progress messages to chat.
    Also pulls assistant messages pushed by the background pipeline
    (validation failures, decisions) using last_pushed_response tracker.
    Updates st.session_state.pipeline_running each poll and fetches full
    state only when the backend may have changed (running/new progress/just-finished).
    """
    try:
        result = get_pipeline_status(st.session_state.app_id)
        progress_log = result.get("progress_log", [])
        was_running = st.session_state.pipeline_running

        # Add only NEW progress messages to chat
        already_seen = sum(1 for entry in st.session_state.chat_history if entry[0] == "progress")
        new_messages = progress_log[already_seen:]
        for msg in new_messages:
            st.session_state.chat_history.append(("progress", msg))

        # Refresh status
        st.session_state.last_status = get_status(st.session_state.app_id)

        # Pull any new assistant messages pushed by the background pipeline.
        # This is polled every cycle so users see background assistant updates
        # (e.g., validation failures/decisions) as soon as they are available.
        # Update running flag
        st.session_state.pipeline_running = result.get("running", False)
        should_fetch_full_state = st.session_state.pipeline_running or was_running or bool(new_messages)

        if should_fetch_full_state:
            try:
                full = get_full_state(st.session_state.app_id)
                api_last_response = full["state"].get("last_response", "")
                if api_last_response and api_last_response != st.session_state.last_pushed_response:
                    if not any(
                        entry[0] == "assistant" and entry[1] == api_last_response
                        for entry in st.session_state.chat_history
                    ):
                        st.session_state.chat_history.append(("assistant", api_last_response))
                    st.session_state.last_pushed_response = api_last_response
            except APIError:
                # Non-blocking: polling should continue even if full-state fetch fails.
                pass

    except APIError:
        pass


# ===============================================================
# API Health Check
# ===============================================================
if st.session_state.api_available is None:
    st.session_state.api_available = check_api()

if not st.session_state.api_available:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}. "
        f"Make sure the FastAPI server is running:\n\n"
        f"`uvicorn api:app --reload --port 8000`"
    )
    if st.button("Retry"):
        st.session_state.api_available = check_api()
        st.rerun()
    st.stop()

if st.session_state.app_id is None:
    init_application()


# ===============================================================
# Poll pipeline if running (fast, non-blocking)
# ===============================================================
if st.session_state.pipeline_running:
    poll_pipeline()


# ===============================================================
# Sidebar
# ===============================================================
status = st.session_state.last_status

with st.sidebar:
    st.markdown("### 🏦 AXE Finance")
    st.caption("AI Loan Assistant")
    st.caption(f"API: `{API_BASE_URL}`")
    st.divider()

    app_id = status.get("application_id", st.session_state.app_id or "N/A")
    app_status = status.get("application_status", "collecting_data")
    loan_type = status.get("loan_type")

    st.markdown(f"**Application:** `{app_id}`")
    status_label = app_status.replace("_", " ").title()
    color = get_status_color(app_status)
    st.markdown(f'<span class="status-badge {color}">{status_label}</span>', unsafe_allow_html=True)

    if loan_type:
        st.markdown(f"**Type:** {loan_type.title()}")
    if status.get("loan_amount"):
        try:
            st.markdown(f"**Amount:** {float(status['loan_amount']):,.0f} TND")
        except (ValueError, TypeError):
            st.markdown(f"**Amount:** {status['loan_amount']} TND")
    if status.get("compliance_tier"):
        st.markdown(f"**Tier:** {status['compliance_tier']}")

    if st.session_state.pipeline_running:
        st.markdown('<div class="processing-banner">⏳ Processing in background... Chat is still active!</div>',
                    unsafe_allow_html=True)

    st.divider()

    # Progress checklist
    st.markdown("**Progress**")
    progress = status.get("progress", {})
    checks = [
        ("Loan type identified", progress.get("loan_type_identified", False)),
        ("Data collected", progress.get("data_collected", False)),
        ("Credit score", progress.get("credit_score_fetched", False)),
        ("Documents uploaded", progress.get("documents_uploaded", False)),
        ("Documents processed", progress.get("documents_processed", False)),
        ("Scoring complete", progress.get("scoring_complete", False)),
    ]
    if loan_type == "business":
        checks.append(("Risk assessed", progress.get("risk_assessed", False)))
    checks.append(("Decision made", progress.get("decision_made", False)))

    for label, done in checks:
        st.markdown(f"{'✅' if done else '⬜'} {label}")

    missing = status.get("missing_fields", [])
    if missing:
        st.caption(f"Next: {missing[0].replace('_', ' ')}")

    st.divider()

    # Document upload
    st.markdown("**Upload documents**")
    st.caption("Upload loan documents (e.g., CIN, salary slip, bank statement) at any time.")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True,
        key="doc_upload",
        label_visibility="collapsed",
    )
    if uploaded_files:
        if st.button("Upload and process", type="primary", use_container_width=True):
            handle_upload(uploaded_files)
            st.rerun()

    # Decision letter download
    if progress.get("decision_made"):
        st.divider()
        letter_bytes = download_letter(st.session_state.app_id)
        if letter_bytes:
            st.download_button(
                "Decision letter",
                data=letter_bytes,
                file_name=f"{app_id}_decision.html",
                mime="text/html",
                use_container_width=True,
            )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New", use_container_width=True):
            reset_application()
            st.rerun()
    with col2:
        st.session_state.debug_mode = st.toggle("Debug", value=st.session_state.debug_mode)


# ===============================================================
# Main Chat Area
# ===============================================================
st.markdown("## 🏦 AXE Finance — AI Loan Assistant")

if st.session_state.pipeline_running:
    st.markdown(
        '<div class="processing-banner">'
        '⚙️ Documents are being processed in the background. '
        'You can keep chatting — ask policy questions or check your status!'
        '</div>',
        unsafe_allow_html=True,
    )

for entry in st.session_state.chat_history:
    if isinstance(entry, tuple):
        role, content = entry
    else:
        continue

    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    elif role == "progress":
        if isinstance(content, dict):
            level = content.get("level", "info")
            msg = content.get("message", "")
            css_class = "progress-msg" if level == "success" else \
                        "progress-warn" if level == "warning" else \
                        "progress-err" if level == "error" else "progress-msg"
            with st.chat_message("assistant", avatar="⚙️"):
                st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="⚙️"):
                st.markdown(f'<div class="progress-msg">{content}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🏦"):
            st.write(content)

# Chat input — ALWAYS available, even during background processing
user_input = st.chat_input("Type your message...")
if user_input:
    handle_user_message(user_input)
    st.rerun()


# ===============================================================
# Auto-refresh while pipeline is running
# Sleep then rerun — chat input remains usable BEFORE this point
# ===============================================================
if st.session_state.pipeline_running:
    time.sleep(POLL_INTERVAL)
    st.rerun()


# ===============================================================
# Debug Panel
# ===============================================================
if st.session_state.debug_mode:
    st.divider()
    try:
        full = get_full_state(st.session_state.app_id)
        tab1, tab2, tab3 = st.tabs(["State", "Progress Log", "Status"])
        with tab1:
            st.json(full["state"])
        with tab2:
            for msg in full.get("progress_log", []):
                st.text(f"[{msg.get('level','info').upper()}] {msg.get('message','')}")
        with tab3:
            st.json(status)
    except APIError as e:
        st.error(f"Debug fetch failed: {e}")
