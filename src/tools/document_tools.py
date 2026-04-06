# src/tools/document_tools.py

"""
Document processing tools — TRILINGUAL (Arabic + French + English).

All keyword detection, document classification, and field extraction
supports Arabic, French, and English text from Tunisian documents.
"""

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


# ---------------------------------------------------------------
# Arabic month mapping (used across multiple tools)
# ---------------------------------------------------------------
AR_MONTHS = {
    "جانفي": "01", "يناير": "01",
    "فيفري": "02", "فبراير": "02",
    "مارس": "03",
    "أفريل": "04", "أبريل": "04",
    "ماي": "05", "مايو": "05",
    "جوان": "06", "يونيو": "06",
    "جويلية": "07", "يوليو": "07",
    "أوت": "08", "أغسطس": "08",
    "سبتمبر": "09",
    "أكتوبر": "10",
    "نوفمبر": "11",
    "ديسمبر": "12",
}


# ---------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------

@tool
def extract_text(file_path: str) -> Dict[str, Any]:
    """Extracts raw text from a PDF document.

    Uses a fallback chain: PyMuPDF -> pdfplumber.
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
            pass

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


# ---------------------------------------------------------------
# OCR — Trilingual (Arabic + French + English)
# ---------------------------------------------------------------

@tool
def ocr_extract_text(file_path: str) -> Dict[str, Any]:
    """Extract text from a scanned PDF or image using OCR (Tesseract).

    Supports: PDF (rasterizes pages), PNG, JPG, TIFF.
    Languages: Arabic + French + English (trilingual).
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    if not TESSERACT_AVAILABLE:
        return {
            "success": False,
            "error": "OCR not available. Install: pip install pytesseract Pillow, plus Tesseract system package.",
        }

    # Trilingual OCR: Arabic + French + English
    ocr_lang = "ara+fra+eng"

    try:
        ext = os.path.splitext(file_path)[1].lower()
        ocr_text = ""

        if ext == ".pdf":
            if fitz is None:
                return {"success": False, "error": "PyMuPDF needed for PDF OCR. pip install pymupdf"}

            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang=ocr_lang)
                ocr_text += page_text + "\n"
            doc.close()

        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            img = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)

        else:
            return {"success": False, "error": f"Unsupported file type for OCR: {ext}"}

        if ocr_text.strip():
            return {
                "success": True,
                "text": ocr_text.strip(),
                "method": "tesseract_ocr",
                "languages": ocr_lang,
            }
        else:
            return {"success": False, "error": "OCR produced no text. Image quality may be too low."}

    except Exception as e:
        return {"success": False, "error": f"OCR failed: {str(e)}"}


# ---------------------------------------------------------------
# Document Classification — Trilingual
# ---------------------------------------------------------------

@tool
def classify_document_type(file_name: str, extracted_text: str) -> str:
    """Classifies the document based on trilingual keyword matching.

    Supports Arabic, French, and English keywords.
    Returns one of: salary_slip, bank_statement, cin_card, income_statement,
    balance_sheet, business_registration, tax_return, collateral_appraisal,
    business_plan, or unknown.
    """
    text = extracted_text.lower()

    # Also check the original (non-lowered) for Arabic since Arabic has no case
    text_orig = extracted_text

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

    # --- Salary Slip ---
    # English
    for kw in ["salary", "payslip", "pay slip", "net pay", "gross pay", "employer",
                "earnings", "deductions", "take-home"]:
        if kw in text:
            scores["salary_slip"] += 1
    # French
    for kw in ["bulletin de paie", "fiche de paie", "salaire", "employeur",
                "salaire brut", "salaire net", "rémunération"]:
        if kw in text:
            scores["salary_slip"] += 1
    # Arabic
    for kw in ["كشف الراتب", "بيان الراتب", "الراتب الصافي", "الراتب الإجمالي",
                "المشغل", "الأجر", "قسيمة الأجر"]:
        if kw in text_orig:
            scores["salary_slip"] += 1

    # --- Bank Statement ---
    # English
    for kw in ["account statement", "bank statement", "transaction", "opening balance",
                "closing balance", "debit", "credit", "account number"]:
        if kw in text:
            scores["bank_statement"] += 1
    # French
    for kw in ["relevé bancaire", "relevé de compte", "solde", "opérations",
                "virement", "retrait", "dépôt"]:
        if kw in text:
            scores["bank_statement"] += 1
    # Arabic
    for kw in ["كشف حساب", "كشف بنكي", "رصيد", "عمليات", "حوالة",
                "سحب", "إيداع", "رقم الحساب"]:
        if kw in text_orig:
            scores["bank_statement"] += 1

    # --- CIN Card ---
    # French
    for kw in ["république tunisienne", "carte d'identité", "carte nationale",
                "identité nationale", "date de naissance"]:
        if kw in text:
            scores["cin_card"] += 1
    # Arabic
    for kw in ["الجمهورية التونسية", "بطاقة التعريف", "بطاقة التعريف الوطنية",
                "تاريخ الولادة", "اللقب", "الاسم", "عنوانها"]:
        if kw in text_orig:
            scores["cin_card"] += 1
    # English
    for kw in ["national identity", "national id card", "identity card"]:
        if kw in text:
            scores["cin_card"] += 1
    # Also check for 8-digit number which is typical CIN format
    if re.search(r'(?<!\d)\d{8}(?!\d)', extracted_text):
        scores["cin_card"] += 1

    # --- Income Statement ---
    # English
    for kw in ["income statement", "profit and loss", "p&l", "revenue",
                "net income", "total expenses", "operating income", "gross profit"]:
        if kw in text:
            scores["income_statement"] += 1
    # French
    for kw in ["compte de résultat", "chiffre d'affaires", "résultat net",
                "charges", "produits", "résultat d'exploitation"]:
        if kw in text:
            scores["income_statement"] += 1
    # Arabic
    for kw in ["بيان الدخل", "قائمة الدخل", "الإيرادات", "صافي الربح",
                "المصاريف", "الأرباح", "رقم المعاملات"]:
        if kw in text_orig:
            scores["income_statement"] += 1

    # --- Balance Sheet ---
    # English
    for kw in ["balance sheet", "total assets", "total liabilities", "equity",
                "shareholders equity", "current assets", "current liabilities"]:
        if kw in text:
            scores["balance_sheet"] += 1
    # French
    for kw in ["bilan", "actif total", "passif", "capitaux propres",
                "actif circulant", "passif circulant", "fonds propres"]:
        if kw in text:
            scores["balance_sheet"] += 1
    # Arabic
    for kw in ["الميزانية", "الميزانية العمومية", "الأصول", "الخصوم",
                "حقوق الملكية", "رؤوس الأموال", "أصول متداولة"]:
        if kw in text_orig:
            scores["balance_sheet"] += 1

    # --- Business Registration ---
    # French
    for kw in ["registre du commerce", "registre national", "matricule fiscal",
                "patente", "code d'activité", "forme juridique"]:
        if kw in text:
            scores["business_registration"] += 1
    # Arabic
    for kw in ["السجل التجاري", "المعرف الجبائي", "الشكل القانوني",
                "رقم التسجيل", "النشاط الرئيسي"]:
        if kw in text_orig:
            scores["business_registration"] += 1
    # English
    for kw in ["business registration", "trade register", "tax id", "company registration",
                "legal form", "incorporation"]:
        if kw in text:
            scores["business_registration"] += 1

    # --- Tax Return ---
    # English
    for kw in ["tax return", "fiscal year", "taxable income", "tax declaration"]:
        if kw in text:
            scores["tax_return"] += 1
    # French
    for kw in ["déclaration fiscale", "impôt", "exercice fiscal",
                "déclaration annuelle", "revenu imposable"]:
        if kw in text:
            scores["tax_return"] += 1
    # Arabic
    for kw in ["التصريح الضريبي", "الإقرار الضريبي", "الدخل الخاضع للضريبة",
                "السنة المالية"]:
        if kw in text_orig:
            scores["tax_return"] += 1

    # --- Collateral Appraisal ---
    # English
    for kw in ["appraisal", "collateral", "market value", "property valuation",
                "estimated value"]:
        if kw in text:
            scores["collateral_appraisal"] += 1
    # French
    for kw in ["évaluation", "garantie", "valeur vénale", "expertise immobilière",
                "valeur estimée"]:
        if kw in text:
            scores["collateral_appraisal"] += 1
    # Arabic
    for kw in ["تقييم", "ضمان", "القيمة السوقية", "خبرة عقارية"]:
        if kw in text_orig:
            scores["collateral_appraisal"] += 1

    # --- Business Plan ---
    # English
    for kw in ["business plan", "financial projections", "market analysis",
                "executive summary", "growth strategy"]:
        if kw in text:
            scores["business_plan"] += 1
    # French
    for kw in ["plan d'affaires", "plan de développement", "étude de marché",
                "projections financières", "résumé exécutif"]:
        if kw in text:
            scores["business_plan"] += 1
    # Arabic
    for kw in ["خطة العمل", "خطة الأعمال", "دراسة السوق", "التوقعات المالية"]:
        if kw in text_orig:
            scores["business_plan"] += 1

    # Return highest score or unknown
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "unknown"
    return best_type


# ---------------------------------------------------------------
# CIN Card Parsing — Trilingual
# ---------------------------------------------------------------

@tool
def parse_cin_card(extracted_text: str) -> Dict[str, Any]:
    """Parse a Tunisian CIN card from extracted text.

    Supports Arabic, French, and English OCR output.
    Extracts: national_id (8 digits), name, date_of_birth, address.
    """
    if not extracted_text or not extracted_text.strip():
        return {"success": False, "error": "Empty text — CIN card may need OCR"}

    text = extracted_text.strip()
    result: Dict[str, Any] = {"success": True}

    # ---- 1. CIN Number (8 digits) — language-independent ----
    cin_match = re.search(r'(?<!\d)(\d{8})(?!\d)', text)
    if cin_match:
        result["cin_national_id"] = cin_match.group(1)
    else:
        result["cin_national_id"] = None
        result["cin_id_warning"] = "Could not find 8-digit CIN number"

    # ---- 2. Date of Birth ----
    result["cin_date_of_birth"] = None

    # Arabic: تاريخ الولادة 25 سبتمبر 2001
    ar_dob = re.search(r'تاريخ\s*(?:الولادة|الميلاد)\s+(\d{1,2})\s+(\S+)\s+(\d{4})', text)
    if ar_dob:
        day, month_ar, year = ar_dob.group(1), ar_dob.group(2), ar_dob.group(3)
        month_num = AR_MONTHS.get(month_ar, "00")
        if month_num != "00":
            result["cin_date_of_birth"] = f"{day.zfill(2)}/{month_num}/{year}"

    # French: née le 25/09/2001 | date de naissance: 25/09/2001
    if not result["cin_date_of_birth"]:
        fr_patterns = [
            r'n[ée]e?\s+le\s+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'date\s+de\s+naissance\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        ]
        for p in fr_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                result["cin_date_of_birth"] = m.group(1)
                break

    # English: born: 25/09/2001 | date of birth: 25/09/2001
    if not result["cin_date_of_birth"]:
        en_patterns = [
            r'(?:born|date\s+of\s+birth)\s*:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        ]
        for p in en_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                result["cin_date_of_birth"] = m.group(1)
                break

    # Generic fallback: any dd/mm/yyyy
    if not result["cin_date_of_birth"]:
        m = re.search(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})', text)
        if m:
            result["cin_date_of_birth"] = m.group(1)

    # ---- 3. Name ----
    result["name"] = None

    # Arabic: اللقب العمري / الاسم نور
    ar_laqab = re.search(r'اللقب\s+(\S+)', text)
    ar_ism = re.search(r'الاسم\s+(\S+)', text)
    if ar_laqab and ar_ism:
        result["name"] = f"{ar_ism.group(1)} {ar_laqab.group(1)}"

    # Arabic alt: الاسم واللقب نور العمري
    if not result["name"]:
        m = re.search(r'الاسم\s+واللقب\s+(.+?)(?:\n|$)', text)
        if m:
            result["name"] = m.group(1).strip()

    # French: Nom: OMRI\nPrénom: NOUR
    if not result["name"]:
        m = re.search(r'nom\s*:?\s*([A-Za-zÀ-ÿ\-]+)\s*[\n,;]\s*pr[ée]nom\s*:?\s*([A-Za-zÀ-ÿ\-]+)', text, re.IGNORECASE)
        if m:
            result["name"] = f"{m.group(2).strip()} {m.group(1).strip()}"

    # French: Nom et Prénom: NOUR OMRI
    if not result["name"]:
        m = re.search(r'nom\s+et\s+pr[ée]nom\s*:?\s*([A-Za-zÀ-ÿ\s\-]{3,40})', text, re.IGNORECASE)
        if m:
            result["name"] = m.group(1).strip()

    # English: Name: NOUR OMRI | Full Name: NOUR OMRI
    if not result["name"]:
        m = re.search(r'(?:full\s+)?name\s*:?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ \-]{1,38})', text, re.IGNORECASE)
        if m:
            result["name"] = m.group(1).strip()

    # ---- 4. Address ----
    result["home_address"] = None

    # Arabic: عنوانها القصرين | العنوان القصرين
    ar_addr = re.search(r'(?:عنوانها|العنوان)\s+(.+?)(?:\n|$)', text)
    if ar_addr:
        result["home_address"] = ar_addr.group(1).strip()

    # French: Adresse: Kasserine | Domicile: Kasserine
    if not result["home_address"]:
        m = re.search(r'(?:adresse|domicile)\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            result["home_address"] = m.group(1).strip()

    # English: Address: Kasserine
    if not result["home_address"]:
        m = re.search(r'address\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            result["home_address"] = m.group(1).strip()

    # ---- 5. Quality Assessment ----
    fields_found = sum(1 for k in ["cin_national_id", "cin_date_of_birth", "name", "home_address"]
                       if result.get(k) is not None)
    result["fields_extracted"] = fields_found
    result["extraction_quality"] = "good" if fields_found >= 3 else "partial" if fields_found >= 1 else "failed"

    return result


# ---------------------------------------------------------------
# CIN Cross-Check
# ---------------------------------------------------------------

@tool
def cross_check_cin(typed_national_id: str, cin_national_id: str,
                    typed_dob: str = None, cin_dob: str = None) -> Dict[str, Any]:
    """Cross-check user-typed identity data against CIN card extracted data.

    Compares national ID (exact match) and date of birth (fuzzy date match).
    Returns match results and a fraud_flag boolean.
    """
    result = {
        "id_match": False,
        "dob_match": None,
        "fraud_flag": False,
        "mismatches": [],
    }

    # National ID check
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

    # Date of birth check
    if typed_dob and cin_dob:
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