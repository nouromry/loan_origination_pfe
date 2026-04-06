# src/tools/db_tools.py

"""
Database persistence tools.

These save the application state and decision results to PostgreSQL.
Uses the existing postgres.py singleton and schema.sql tables.

Designed to be called from the decision_node AFTER the decision is made.
If the DB is unavailable, these fail gracefully — the application
still works, we just don't persist.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.tools import tool


def _get_db():
    """Lazy import to avoid crashing if PostgreSQL isn't configured."""
    try:
        from src.infrastructure.postgres import db
        return db
    except Exception:
        return None


def _hash_pii(value: str) -> str:
    """One-way hash for PII fields (national_id, phone)."""
    if not value:
        return ""
    return hashlib.sha256(value.encode()).hexdigest()


@tool
def save_application(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save or update the loan application in PostgreSQL.

    Creates/updates rows in: users, applications.
    PII fields (national_id, phone) are hashed before storage.
    
    Returns success status and any errors.
    """
    db = _get_db()
    if db is None:
        return {"success": False, "error": "PostgreSQL not configured"}

    try:
        app_id = state.get("application_id")
        name = state.get("name", "Unknown")
        national_id = state.get("national_id", "")
        email = state.get("email")
        phone = state.get("phone", "")

        # 1. Upsert user
        national_id_hash = _hash_pii(national_id)
        phone_hash = _hash_pii(phone)

        db.execute_query(
            """
            INSERT INTO users (name, national_id_hash, email, phone_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (national_id_hash) DO UPDATE
            SET name = EXCLUDED.name, email = EXCLUDED.email
            RETURNING user_id
            """,
            (name, national_id_hash, email, phone_hash)
        )

        # Get user_id
        user_rows = db.execute_query(
            "SELECT user_id FROM users WHERE national_id_hash = %s",
            (national_id_hash,)
        )
        user_id = user_rows[0]["user_id"] if user_rows else None

        # 2. Upsert application
        db.execute_query(
            """
            INSERT INTO applications (
                application_id, user_id, loan_amount, loan_term_months,
                loan_type, loan_purpose, monthly_income, credit_score,
                status, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (application_id) DO UPDATE SET
                status = EXCLUDED.status,
                credit_score = EXCLUDED.credit_score,
                monthly_income = EXCLUDED.monthly_income,
                updated_at = EXCLUDED.updated_at
            """,
            (
                app_id, user_id,
                state.get("loan_amount"), state.get("loan_term_months"),
                state.get("loan_type", "personal"),
                state.get("loan_purpose_category"),
                state.get("monthly_income"),
                state.get("credit_score"),
                state.get("application_status", "pending"),
                datetime.now(),
            )
        )

        return {"success": True, "application_id": app_id, "user_id": user_id}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def save_decision(application_id: str, decision_result: Dict[str, Any],
                  scoring_result: Dict[str, Any] = None,
                  risk_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Save the credit decision and financial ratios to PostgreSQL.

    Creates rows in: decisions, financial_ratios, audit_log.
    """
    db = _get_db()
    if db is None:
        return {"success": False, "error": "PostgreSQL not configured"}

    try:
        # 1. Save decision
        decision = decision_result.get("decision", "unknown")
        interest_rate = decision_result.get("interest_rate")
        monthly_payment = decision_result.get("monthly_payment")

        # Extract risk info
        risk_category = "unknown"
        default_prob = None
        if scoring_result:
            risk_category = scoring_result.get("risk_category", "unknown")
            default_prob = scoring_result.get("default_probability")

        # Build decision factors JSON
        factors = {
            "scoring": scoring_result or {},
            "risk_assessment": risk_result or {},
            "explanation": decision_result.get("explanation", ""),
        }

        db.execute_query(
            """
            INSERT INTO decisions (
                application_id, decision, interest_rate, monthly_payment,
                ml_risk_score, risk_tier, decision_factors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                application_id, decision, interest_rate, monthly_payment,
                default_prob, risk_category,
                json.dumps(factors, default=str),
            )
        )

        # 2. Save financial ratios
        dti = None
        dscr = None
        current_ratio = None
        debt_to_equity = None
        npm = None

        if scoring_result:
            dti = scoring_result.get("dti")

        if risk_result:
            ratios = risk_result.get("ratios", {})
            dscr_data = ratios.get("dscr", {})
            dscr = dscr_data.get("dscr") if isinstance(dscr_data, dict) else None
            cr_data = ratios.get("current_ratio", {})
            current_ratio = cr_data.get("current_ratio") if isinstance(cr_data, dict) else None
            de_data = ratios.get("debt_to_equity", {})
            debt_to_equity = de_data.get("debt_to_equity") if isinstance(de_data, dict) else None
            npm_data = ratios.get("net_profit_margin", {})
            npm = npm_data.get("net_profit_margin") if isinstance(npm_data, dict) else None

        db.execute_query(
            """
            INSERT INTO financial_ratios (
                application_id, dti, dscr, current_ratio,
                debt_to_equity, net_profit_margin
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (application_id, dti, dscr, current_ratio, debt_to_equity, npm)
        )

        # 3. Audit log
        db.execute_query(
            """
            INSERT INTO audit_log (application_id, agent_name, action, details)
            VALUES (%s, %s, %s, %s)
            """,
            (
                application_id, "decision_node",
                f"decision_{decision}",
                json.dumps({
                    "decision": decision,
                    "interest_rate": interest_rate,
                    "risk_tier": risk_category,
                }, default=str),
            )
        )

        return {"success": True, "decision_saved": True}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def save_audit_log(application_id: str, agent_name: str,
                   action: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Write an entry to the audit log table.
    
    Called at key processing steps for compliance tracking.
    """
    db = _get_db()
    if db is None:
        return {"success": False, "error": "PostgreSQL not configured"}

    try:
        db.execute_query(
            """
            INSERT INTO audit_log (application_id, agent_name, action, details)
            VALUES (%s, %s, %s, %s)
            """,
            (
                application_id, agent_name, action,
                json.dumps(details or {}, default=str),
            )
        )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}