# src/nodes/document_node.py

"""
Document Processing Node.

Pipeline per file:
  1. extract_text (PyMuPDF/pdfplumber)
  2. If text is too short → ocr_extract_text (Tesseract fallback)
  3. classify_document_type (keyword scoring)
  4. Route by type:
       cin_card        → parse_cin_card tool (regex extraction)
       salary_slip     → Agent LLM parsing
       bank_statement  → Agent LLM parsing
       income_statement / balance_sheet → Agent LLM parsing
  5. Pre-fill GlobalState from extracted values
  6. Cross-check CIN ID vs typed national_id
"""

import os
from src.models.global_state import GlobalState, add_thought
from src.agents.document_agent import DocumentAgent
from src.tools.document_tools import (
    extract_text,
    ocr_extract_text,
    classify_document_type,
    parse_cin_card,
    cross_check_cin,
)


# Minimum characters to consider text extraction successful
MIN_TEXT_LENGTH = 50


def document_node(state: GlobalState) -> GlobalState:
    """
    Processes all uploaded documents.
    The node orchestrates; tools and agent do the work.
    """
    state["application_status"] = "processing_documents"
    state["stage"] = "processing"
    add_thought(state, "Processing uploaded documents...")

    # 1. Find files
    upload_dir = os.getenv("TEMP_UPLOADS_DIR", "./temp_uploads")
    app_id = state.get("application_id", "UNKNOWN")
    app_dir = os.path.join(upload_dir, app_id)

    if not os.path.exists(app_dir):
        add_thought(state, f"Upload directory not found: {app_dir}")
        return state

    files_to_process = [
        os.path.join(app_dir, f)
        for f in os.listdir(app_dir)
        if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff'))
    ]

    if not files_to_process:
        add_thought(state, "No documents found in upload directory.")
        return state

    add_thought(state, f"Found {len(files_to_process)} document(s) to process.")

    # 2. Process each file: extract → classify → parse
    all_results = {}
    cin_result = None

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        add_thought(state, f"Processing: {file_name}")

        # Step A: Extract text
        text_result = extract_text.invoke({"file_path": file_path})

        extracted_text = ""
        if text_result.get("success") and len(text_result.get("text", "")) >= MIN_TEXT_LENGTH:
            extracted_text = text_result["text"]
        else:
            # OCR fallback for scanned documents
            add_thought(state, f"Regular extraction failed for {file_name}, trying OCR...")
            ocr_result = ocr_extract_text.invoke({"file_path": file_path})
            if ocr_result.get("success"):
                extracted_text = ocr_result["text"]
            else:
                add_thought(state, f"OCR also failed for {file_name}: {ocr_result.get('error')}")
                all_results[file_name] = {"error": "Text extraction failed", "file": file_path}
                continue

        # Step B: Classify document type
        doc_type = classify_document_type.invoke({
            "file_name": file_name,
            "extracted_text": extracted_text,
        })
        add_thought(state, f"Classified {file_name} as: {doc_type}")

        # Step C: Route by document type
        if doc_type == "cin_card":
            cin_result = _process_cin_card(state, extracted_text, file_name)
            all_results[file_name] = {"type": "cin_card", "result": cin_result}

        else:
            # Financial documents → delegate to agent's LLM
            doc_result = _process_financial_doc(state, extracted_text, doc_type, file_name)
            all_results[file_name] = {"type": doc_type, "result": doc_result}

    # 3. Store combined result
    state["document_result"] = all_results

    # 4. Cross-check CIN if we have both typed and extracted IDs
    _run_cross_checks(state)

    return state


def _process_cin_card(state: GlobalState, text: str, file_name: str) -> dict:
    """Process a CIN card using the parse_cin_card tool."""
    add_thought(state, "Parsing CIN card...")

    result = parse_cin_card.invoke({"extracted_text": text})

    # Pre-fill GlobalState from CIN extraction
    cin_fields = {
        "cin_national_id": result.get("cin_national_id"),
        "cin_date_of_birth": result.get("cin_date_of_birth"),
        "name": result.get("name"),
        "home_address": result.get("home_address"),
    }

    for field, value in cin_fields.items():
        if value is not None and state.get(field) is None:
            state[field] = value

    quality = result.get("extraction_quality", "unknown")
    fields_found = result.get("fields_extracted", 0)
    add_thought(state, f"CIN parsed: {fields_found}/4 fields extracted (quality: {quality})")

    return result


def _process_financial_doc(state: GlobalState, text: str, doc_type: str, file_name: str) -> dict:
    """Process a financial document using the DocumentAgent's LLM."""
    agent = DocumentAgent()
    result = agent.run(
        goal=f"Extract all financial data from this {doc_type} document.",
        doc_type=doc_type,
        loan_type=state.get("loan_type", "unknown"),
        extracted_text=text[:4000],  # Limit text to avoid token overflow
    )

    # Pre-fill GlobalState from extracted values
    FIELD_MAP = {
        # Salary slip fields
        "net_salary": "monthly_income",
        "gross_salary": "gross_salary",
        "employer_name": "employer_name",
        "employment_status": "employment_status",
        "employment_duration_months": "employment_duration_months",
        # Bank statement fields
        "monthly_rent_or_mortgage": "monthly_rent_or_mortgage",
        "existing_debt_payments": "existing_debt_payments",
        "monthly_cash_flow": "monthly_cash_flow",
        "total_monthly_credits": "total_monthly_credits",
        # Business: income statement
        "total_revenue": "total_revenue",
        "gross_profit": "gross_profit",
        "total_expenses": "total_expenses",
        "net_income": "net_income",
        # Business: balance sheet
        "total_assets": "total_assets",
        "current_assets": "current_assets",
        "total_liabilities": "total_liabilities",
        "current_liabilities": "current_liabilities",
        "equity": "equity",
        # Business: registration
        "business_name": "business_name",
        "business_registration_number": "business_registration_number",
        "legal_structure": "legal_structure",
        "business_address": "business_address",
        "years_in_operation": "years_in_operation",
    }

    for doc_field, state_field in FIELD_MAP.items():
        extracted_value = result.get(doc_field)
        if extracted_value is not None and state.get(state_field) is None:
            state[state_field] = extracted_value

    add_thought(state, f"Extracted financial data from {doc_type}: {file_name}")
    return result


def _run_cross_checks(state: GlobalState) -> None:
    """Run compliance cross-checks after all documents are processed."""

    typed_id = state.get("national_id")
    cin_id = state.get("cin_national_id")

    # Cross-check national ID
    if typed_id and cin_id:
        check = cross_check_cin.invoke({
            "typed_national_id": typed_id,
            "cin_national_id": cin_id,
            "typed_dob": state.get("date_of_birth"),
            "cin_dob": state.get("cin_date_of_birth"),
        })

        if check.get("fraud_flag"):
            state["application_status"] = "blocked"
            state["rejection_reason"] = (
                f"FRAUD FLAG: {'; '.join(check.get('mismatches', []))}. "
                f"Application blocked for manual review."
            )
            add_thought(state, f"⚠️ CROSS-CHECK FAILED: {check.get('mismatches')}")
            return

        add_thought(state, "CIN cross-check passed: IDs match.")

        # Also check DOB if available
        if check.get("dob_match") is False:
            add_thought(state, f"⚠️ DOB mismatch detected (non-blocking): typed vs CIN")

    elif cin_id is None:
        add_thought(state, "CIN card not uploaded or CIN number not extracted — cross-check skipped.")

    # If we got here without blocking, mark as processed
    if state.get("application_status") != "blocked":
        state["application_status"] = "documents_processed"
        add_thought(state, "All document processing and cross-checks complete.")