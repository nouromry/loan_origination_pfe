# src/tools/extraction_tools.py

import re
from langchain_core.tools import tool
from typing import Optional


@tool
def extract_national_id(message: str) -> Optional[str]:
    """Extract an 8-digit Tunisian CIN / National ID from a message.
    
    Looks for exactly 8 consecutive digits that are NOT part of a longer
    number (e.g., phone numbers, amounts). Validates boundaries to avoid
    matching substrings of larger numeric values.
    """
    # Remove common amount patterns first to avoid false matches
    cleaned = re.sub(r'[\d,]+\.\d+', '', message)  # Remove decimals like 50000.00
    # Match exactly 8 digits at word boundaries
    match = re.search(r'(?<!\d)\d{8}(?!\d)', cleaned)
    return match.group(0) if match else None


@tool
def extract_email(message: str) -> Optional[str]:
    """Extract a valid email address from a message."""
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', message)
    return match.group(0) if match else None


@tool
def extract_phone(message: str) -> Optional[str]:
    """Extract a Tunisian phone number (8 digits, optionally prefixed with +216).
    
    Accepts formats: 98765432, +216 98765432, 00216-98-765-432, etc.
    Returns normalized 8-digit format.
    """
    # Remove common separators
    cleaned = re.sub(r'[\s\-\.]', '', message)
    # Match with optional country code
    match = re.search(r'(?:\+?216|00216)?(\d{8})(?!\d)', cleaned)
    return match.group(1) if match else None


@tool
def extract_amount(message: str) -> Optional[float]:
    """Extract a monetary amount, handling formats like 50K, 1.5M, 50,000, $50000, 50000 TND.
    
    Returns the numeric value as a float.
    """
    # Remove currency symbols and codes but keep the number
    cleaned = message.upper().replace(',', '').replace('TND', '').replace('$', '').replace('€', '').strip()
    
    # Match numbers followed optionally by K or M
    match = re.search(r'(\d+(?:\.\d+)?)\s*(K|M)?(?:\s|$)', cleaned)
    
    if not match:
        return None

    base_val = float(match.group(1))
    multiplier = match.group(2)

    if multiplier == 'K':
        return base_val * 1_000
    elif multiplier == 'M':
        return base_val * 1_000_000
    return base_val


@tool
def extract_loan_term(message: str) -> Optional[int]:
    """Extract loan term in months from a message.
    
    Handles: '36 months', '3 years', '2 ans', '24 mois', etc.
    Returns the term in months.
    """
    message_lower = message.lower()
    
    # Try months first (English + French)
    match = re.search(r'(\d+)\s*(?:months?|mois)', message_lower)
    if match:
        return int(match.group(1))
    
    # Try years (English + French)
    match = re.search(r'(\d+)\s*(?:years?|ans?)', message_lower)
    if match:
        return int(match.group(1)) * 12

    # Try bare number if context suggests it's a term
    match = re.search(r'\b(\d{1,3})\b', message_lower)
    if match:
        val = int(match.group(1))
        if 6 <= val <= 360:  # Reasonable month range
            return val

    return None
