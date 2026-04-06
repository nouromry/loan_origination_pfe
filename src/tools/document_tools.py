# src/tools/document_tools.py

import os
import re
from langchain_core.tools import tool
from typing import Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@tool
def extract_text(file_path: str) -> Dict[str, Any]:
    """Extracts raw text from a PDF document.
    
    Uses a fallback chain: PyMuPDF → pdfplumber.
    Returns the extracted text, page count, and extraction method used.
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    # Strategy 1: PyMuPDF (fast, handles most PDFs)
    if fitz is not None:
        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()

            if full_text.strip():
                return {
                    "success": True,
                    "text": full_text.strip(),
                    "pages": num_pages,
                    "method": "pymupdf",
                }
        except Exception:
            pass  # Fall through to next strategy

    # Strategy 2: pdfplumber (better for table-heavy PDFs)
    if pdfplumber is not None:
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

            if full_text.strip():
                return {
                    "success": True,
                    "text": full_text.strip(),
                    "pages": num_pages,
                    "method": "pdfplumber",
                }
        except Exception:
            pass

    return {
        "success": False,
        "error": "All text extraction methods failed. Document may need OCR.",
    }


@tool
def classify_document_type(file_name: str, extracted_text: str) -> str:
    """Classifies the document based on keywords in the extracted text.
    
    Returns one of: salary_slip, bank_statement, cin_card, income_statement,
    balance_sheet, business_registration, tax_return, collateral_appraisal,
    business_plan, or unknown.
    """
    text_lower = extracted_text.lower()

    # Score each document type by keyword matches (more keywords = higher confidence)
    scores: Dict[str, int] = {
        "salary_slip": 0,
        "bank_statement": 0,
        "cin_card": 0,
        "income_statement": 0,
        "balance_sheet": 0,
        "business_registration": 0,
        "tax_return": 0,
        "collateral_appraisal": 0,
        "business_plan": 0,
    }

    # Salary slip keywords
    for kw in ["salary", "payslip", "bulletin de paie", "net pay", "gross pay",
                "employer", "employeur", "salaire", "fiche de paie"]:
        if kw in text_lower:
            scores["salary_slip"] += 1

    # Bank statement keywords
    for kw in ["account statement", "relevé bancaire", "transaction", "credit",
                "debit", "opening balance", "closing balance", "solde"]:
        if kw in text_lower:
            scores["bank_statement"] += 1

    # CIN card keywords
    for kw in ["république tunisienne", "carte d'identité", "carte nationale",
                "cin", "identité nationale", "date de naissance"]:
        if kw in text_lower:
            scores["cin_card"] += 1

    # Income statement keywords
    for kw in ["income statement", "compte de résultat", "revenue", "chiffre d'affaires",
                "net income", "résultat net", "total expenses", "charges"]:
        if kw in text_lower:
            scores["income_statement"] += 1

    # Balance sheet keywords
    for kw in ["balance sheet", "bilan", "total assets", "actif total",
                "total liabilities", "passif", "equity", "capitaux propres"]:
        if kw in text_lower:
            scores["balance_sheet"] += 1

    # Business registration
    for kw in ["registre du commerce", "business registration", "patente",
                "registre national", "matricule fiscal"]:
        if kw in text_lower:
            scores["business_registration"] += 1

    # Tax return
    for kw in ["tax return", "déclaration fiscale", "impôt", "fiscal year"]:
        if kw in text_lower:
            scores["tax_return"] += 1

    # Collateral appraisal
    for kw in ["appraisal", "évaluation", "collateral", "market value", "property"]:
        if kw in text_lower:
            scores["collateral_appraisal"] += 1

    # Business plan
    for kw in ["business plan", "plan d'affaires", "financial projections",
                "market analysis", "executive summary"]:
        if kw in text_lower:
            scores["business_plan"] += 1

    # Return the type with the highest score, or unknown
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "unknown"
    return best_type


# ---------------------------------------------------------------
# CIN Card Parsing
# ---------------------------------------------------------------

@tool
def parse_cin_card(extracted_text: str) -> Dict[str, Any]:
    """Parse a Tunisian CIN (Carte d'Identité Nationale) card from extracted text.

    Extracts: national_id (8 digits), full_name, date_of_birth, address.
    Uses regex patterns first, then returns what it found.

    Handles both French and Arabic text from the CIN card.
    The Tunisian CIN contains:
      - CIN number: 8 digits (front of card)
      - Full name in Arabic and French/Latin
      - Date of birth (dd/mm/yyyy or dd-mm-yyyy)
      - Place of birth
      - Address
    """
    if not extracted_text or not extracted_text.strip():
        return {"success": False, "error": "Empty text — CIN card may need OCR"}

    text = extracted_text.strip()
    result: Dict[str, Any] = {"success": True}

    # 1. Extract CIN number (8 consecutive digits)
    cin_match = re.search(r'(?<!\d)(\d{8})(?!\d)', text)
    if cin_match:
        result["cin_national_id"] = cin_match.group(1)
    else:
        result["cin_national_id"] = None
        result["cin_id_warning"] = "Could not find 8-digit CIN number"

    # 2. Extract date of birth
    #    Tunisian format: dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy
    dob_patterns = [
        r'(?:n[ée]\s+le|date\s+de\s+naissance|born)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',  # Any dd/mm/yyyy pattern
    ]
    result["cin_date_of_birth"] = None
    for pattern in dob_patterns:
        dob_match = re.search(pattern, text, re.IGNORECASE)
        if dob_match:
            result["cin_date_of_birth"] = dob_match.group(1)
            break

    # 3. Extract name
    #    Common patterns on Tunisian CIN:
    #    "Nom et Prénom: ..." or "Nom: ... Prénom: ..."
    name_patterns = [
        r'(?:nom\s+et\s+pr[ée]nom|full\s+name)\s*:?\s*([A-Za-zÀ-ÿ\s\-]+)',
        r'nom\s*:?\s*([A-Za-zÀ-ÿ\-]+)\s+pr[ée]nom\s*:?\s*([A-Za-zÀ-ÿ\s\-]+)',
        r'(?:mr?s?\.?|m(?:me|lle)\.?)\s+([A-Za-zÀ-ÿ\s\-]{3,40})',
    ]
    result["name"] = None
    for i, pattern in enumerate(name_patterns):
        name_match = re.search(pattern, text, re.IGNORECASE)
        if name_match:
            if i == 1 and name_match.group(2):
                # "Nom: X Prénom: Y" → "Y X" (first last)
                result["name"] = f"{name_match.group(2).strip()} {name_match.group(1).strip()}"
            else:
                result["name"] = name_match.group(1).strip()
            break

    # 4. Extract address
    address_patterns = [
        r'(?:adresse|address|domicile)\s*:?\s*(.+?)(?:\n|$)',
        r'(?:r[ée]sidant\s+[àa]|demeurant\s+[àa])\s*(.+?)(?:\n|$)',
    ]
    result["home_address"] = None
    for pattern in address_patterns:
        addr_match = re.search(pattern, text, re.IGNORECASE)
        if addr_match:
            result["home_address"] = addr_match.group(1).strip()
            break

    # 5. Quality assessment
    fields_found = sum(1 for k in ["cin_national_id", "cin_date_of_birth", "name", "home_address"]
                       if result.get(k) is not None)
    result["fields_extracted"] = fields_found
    result["extraction_quality"] = "good" if fields_found >= 3 else "partial" if fields_found >= 1 else "failed"

    return result


@tool
def cross_check_cin(typed_national_id: str, cin_national_id: str,
                    typed_dob: str = None, cin_dob: str = None) -> Dict[str, Any]:
    """Cross-check user-typed identity data against CIN card extracted data.

    Compares:
      - National ID: exact match (8-digit string)
      - Date of birth: fuzzy date match (handles format differences)

    Returns a dict with match results and a fraud_flag boolean.
    """
    result = {
        "id_match": False,
        "dob_match": None,  # None = not checked (missing data)
        "fraud_flag": False,
        "mismatches": [],
    }

    # 1. National ID check (critical)
    typed_clean = str(typed_national_id).strip()
    cin_clean = str(cin_national_id).strip()

    if typed_clean == cin_clean:
        result["id_match"] = True
    else:
        result["id_match"] = False
        result["fraud_flag"] = True
        result["mismatches"].append(
            f"National ID mismatch: typed '{typed_clean}' vs CIN '{cin_clean}'"
        )

    # 2. Date of birth check (if both available)
    if typed_dob and cin_dob:
        # Normalize dates: strip separators, compare digits
        def normalize_date(d: str) -> str:
            return re.sub(r'[/\-\.\s]', '', d.strip())

        if normalize_date(typed_dob) == normalize_date(cin_dob):
            result["dob_match"] = True
        else:
            result["dob_match"] = False
            result["mismatches"].append(
                f"Date of birth mismatch: typed '{typed_dob}' vs CIN '{cin_dob}'"
            )

    return result


# ---------------------------------------------------------------
# OCR Fallback for Scanned Documents
# ---------------------------------------------------------------

@tool
def ocr_extract_text(file_path: str) -> Dict[str, Any]:
    """Extract text from a scanned PDF or image using OCR (Tesseract).

    Fallback when regular text extraction returns empty/short text.
    Supports: PDF (rasterizes pages), PNG, JPG, TIFF.
    Languages: French + Arabic (common on Tunisian documents).
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    if not TESSERACT_AVAILABLE:
        return {
            "success": False,
            "error": "OCR not available. Install: pip install pytesseract Pillow, plus Tesseract system package.",
        }

    try:
        ext = os.path.splitext(file_path)[1].lower()
        ocr_text = ""

        if ext == ".pdf":
            # Rasterize PDF pages to images, then OCR each
            if fitz is None:
                return {"success": False, "error": "PyMuPDF needed for PDF OCR. pip install pymupdf"}

            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                # Render page at 300 DPI for good OCR quality
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang="fra+ara")
                ocr_text += page_text + "\n"
            doc.close()

        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            img = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(img, lang="fra+ara")

        else:
            return {"success": False, "error": f"Unsupported file type for OCR: {ext}"}

        if ocr_text.strip():
            return {
                "success": True,
                "text": ocr_text.strip(),
                "method": "tesseract_ocr",
                "languages": "fra+ara",
            }
        else:
            return {"success": False, "error": "OCR produced no text. Image quality may be too low."}

    except Exception as e:
        return {"success": False, "error": f"OCR failed: {str(e)}"}