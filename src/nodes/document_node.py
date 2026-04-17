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
    extract_tables_as_markdown,
    ocr_extract_text,
    classify_document_type,
    parse_cin_card,
    cross_check_cin,
)


# Document types that commonly contain tables — these trigger table extraction
TABULAR_DOC_TYPES = {
    "bank_statement",
    "income_statement",
    "balance_sheet",
    "tax_return",
}


# Minimum characters to consider text extraction successful
MIN_TEXT_LENGTH = 50


def document_node(state: GlobalState) -> GlobalState:
    """
    Processes all uploaded documents.
    The node orchestrates; tools and agent do the work.
    """
    state["application_status"] = "processing"
    state["stage"] = "processing"
    add_thought(state, "Processing uploaded documents...")
    if state.get("processed_files") is None:
        state["processed_files"] = []
    if state.get("cin_missing_fields") is None:
        state["cin_missing_fields"] = []

    # 1. Find files
    upload_dir = os.getenv("TEMP_UPLOADS_DIR", "./temp_uploads")
    app_id = state.get("application_id", "UNKNOWN")
    app_dir = os.path.join(upload_dir, app_id)

    if not os.path.exists(app_dir):
        add_thought(state, f"Upload directory not found: {app_dir}")
        return state

    all_files = [
        os.path.join(app_dir, f)
        for f in os.listdir(app_dir)
        if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff'))
    ]

    if not all_files:
        add_thought(state, "No documents found in upload directory.")
        return state

    # Filter out files we've already processed (by name + size)
    # This enables incremental upload: user adds a new doc later,
    # we only process the new one. If the same filename has a different
    # size, treat it as a re-upload and process it again.
    processed = set(state.get("processed_files") or [])
    files_to_process = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            file_size = 0
        file_key = f"{file_name}:{file_size}"
        if file_key not in processed:
            files_to_process.append((file_path, file_key))

    replaced_keys_by_file = {}
    for file_path, file_key in files_to_process:
        file_name = os.path.basename(file_path)
        replaced_keys = [
            key for key in processed
            if key.startswith(f"{file_name}:") and key != file_key
        ]
        if replaced_keys:
            replaced_keys_by_file[file_name] = replaced_keys

    if not files_to_process:
        add_thought(state, f"All {len(all_files)} document(s) already processed. Running validation checks only.")
    else:
        add_thought(state, f"Found {len(all_files)} document(s) total, {len(files_to_process)} new to process.")

    # 2. Process each file: extract → classify → parse
    # Keep prior results — only ADD new ones
    all_results = dict(state.get("document_result") or {})

    for file_path, file_key in files_to_process:
        file_name = os.path.basename(file_path)
        add_thought(state, f"Processing: {file_name}")

        # Explicit replacement support: same filename re-uploaded with new size
        replaced_keys = replaced_keys_by_file.get(file_name, [])
        if replaced_keys:
            state["processed_files"] = [
                key for key in state.get("processed_files", [])
                if key not in replaced_keys
            ]
            add_thought(state, f"Detected replaced file for {file_name}. Reprocessing latest version only.")

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
                # Mark as processed even though it failed — don't infinitely retry corrupted files
                if file_key not in state["processed_files"]:
                    state["processed_files"].append(file_key)
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

        elif doc_type in ("unknown", None, ""):
            # We don't know what this document is — DO NOT send it to the LLM
            # (we'd get hallucinated financial data for a random document type)
            add_thought(state, f"WARNING: Could not identify document type of {file_name}. Skipping extraction.")
            all_results[file_name] = {
                "type": "unknown",
                "result": {"error": "Document type could not be identified"},
            }

        else:
            # Known financial document → delegate to agent's LLM
            doc_result = _process_financial_doc(
                state, extracted_text, doc_type, file_name, file_path
            )
            all_results[file_name] = {"type": doc_type, "result": doc_result}

        # Mark this file as processed (regardless of success/failure)
        # so we don't re-process it on the next pipeline run
        if file_key not in state["processed_files"]:
            state["processed_files"].append(file_key)

    # 3. Store combined result
    state["document_result"] = all_results

    # 4. Cross-check CIN if we have both typed and extracted IDs
    _run_cross_checks(state)

    # 5. Validate that we got everything we need from the documents
    from src.models.global_state import validate_document_extraction
    report = validate_document_extraction(state)
    state["document_validation"] = report

    cin_missing_fields = list(state.get("cin_missing_fields") or [])
    identity_mismatch = state.get("identity_mismatch")

    if identity_mismatch or cin_missing_fields or report["has_issues"]:
        state["application_status"] = "needs_correction"
        add_thought(state, f"Document validation FAILED: {report['user_message']}")
    else:
        state["application_status"] = "ready_for_scoring"
        add_thought(state, "Document validation passed: all critical data extracted.")

    return state


def _process_cin_card(state: GlobalState, text: str, file_name: str) -> dict:
    """Process a CIN card using the parse_cin_card tool."""
    add_thought(state, "Parsing CIN card...")

    result = parse_cin_card.invoke({"extracted_text": text})

    # Pre-fill GlobalState from CIN extraction
    # Both the CIN-specific fields (for cross-checking) AND the
    # chat-collectible fields (so we don't ask the user again)
    cin_id = result.get("cin_national_id")
    cin_dob = result.get("cin_date_of_birth")
    cin_name = result.get("name")
    cin_addr = result.get("home_address")

    pre_fill = {
        # CIN-specific (for cross-check vs typed national_id)
        "cin_national_id": cin_id,
        "cin_date_of_birth": cin_dob,
        # Chat-collectible (so collect_node doesn't re-ask)
        "national_id": cin_id,
        "date_of_birth": cin_dob,
        "name": cin_name,
        "home_address": cin_addr,
    }

    filled_chat_fields = []
    for field, value in pre_fill.items():
        if value is not None and state.get(field) is None:
            state[field] = value
            if field in ("national_id", "date_of_birth", "name"):
                filled_chat_fields.append(field)

    if filled_chat_fields:
        add_thought(
            state,
            f"CIN auto-filled chat fields: {', '.join(filled_chat_fields)} "
            f"(no need to ask user again)"
        )

    quality = result.get("extraction_quality", "unknown")
    fields_found = result.get("fields_extracted", 0)
    missing_fields = []
    for field in ["cin_national_id", "cin_date_of_birth", "name", "home_address"]:
        if result.get(field) in (None, ""):
            missing_fields.append(field)
    result["missing_fields"] = missing_fields

    if quality == "partial" and missing_fields:
        state["cin_missing_fields"] = missing_fields
        state["application_status"] = "needs_correction"
        add_thought(
            state,
            f"CIN extraction is partial. User confirmation needed for: {', '.join(missing_fields)}."
        )
    else:
        state["cin_missing_fields"] = []

    add_thought(state, f"CIN parsed: {fields_found}/4 fields extracted (quality: {quality})")

    return result


def _process_financial_doc(state: GlobalState, text: str, doc_type: str,
                           file_name: str, file_path: str) -> dict:
    """
    Process a financial document using the DocumentAgent's LLM.
    
    For tabular document types (bank_statement, income_statement, balance_sheet,
    tax_return), we first try to extract structured tables via pdfplumber and
    feed them to the LLM as markdown. This gives the LLM much cleaner input
    than raw flattened PDF text.
    """
    # Default: use the raw extracted text
    llm_input = text
    extraction_note = "raw text (PyMuPDF)"

    # For tabular document types, try structured table extraction
    if doc_type in TABULAR_DOC_TYPES:
        try:
            table_result = extract_tables_as_markdown.invoke({"file_path": file_path})

            if table_result.get("success") and table_result.get("has_tables"):
                tables_md = table_result.get("tables_markdown", "")
                header_text = table_result.get("header_text", "")
                table_count = table_result.get("table_count", 0)

                # Combine header + tables for the LLM
                # Header first (shorter context), then tables (main data)
                parts = []
                if header_text.strip():
                    parts.append(f"--- DOCUMENT HEADER / NARRATIVE ---\n{header_text[:1500]}")
                parts.append(f"--- EXTRACTED TABLES ({table_count} found) ---\n{tables_md}")
                llm_input = "\n\n".join(parts)

                extraction_note = f"structured tables ({table_count}) + header"
                add_thought(
                    state,
                    f"Table extraction: found {table_count} tables in {file_name}, "
                    f"feeding structured markdown to LLM."
                )
            else:
                # Tables not found → fall back to raw text (already set above)
                add_thought(
                    state,
                    f"Table extraction: no tables detected in {file_name}, using raw text."
                )

        except Exception as e:
            # Graceful degradation — use raw text
            add_thought(
                state,
                f"Table extraction failed for {file_name}: {str(e)[:80]}. Using raw text."
            )

    # Call the LLM with the best input we have
    agent = DocumentAgent()
    result = agent.run(
        goal=f"Extract all financial data from this {doc_type} document.",
        doc_type=doc_type,
        loan_type=state.get("loan_type", "unknown"),
        extracted_text=llm_input[:4500],  # Slightly higher limit for table content
    )

    # Tag the result with the extraction method for debugging/reporting
    result["_extraction_method"] = extraction_note

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

        if check.get("id_match") is False:
            state["application_status"] = "needs_correction"
            state["identity_mismatch"] = {
                "typed_national_id": typed_id,
                "cin_national_id": cin_id,
                "mismatches": check.get("mismatches", []),
            }
            add_thought(state, f"⚠️ CIN ID mismatch detected. Waiting for user confirmation.")
            return

        state["identity_mismatch"] = {}

        add_thought(state, "CIN cross-check passed: IDs match.")

        # Also check DOB if available
        if check.get("dob_match") is False:
            add_thought(state, f"⚠️ DOB mismatch detected (non-blocking): typed vs CIN")

    elif cin_id is None:
        add_thought(state, "CIN card not uploaded or CIN number not extracted — cross-check skipped.")
        state["identity_mismatch"] = {}

    add_thought(state, "All document processing and cross-checks complete.")
