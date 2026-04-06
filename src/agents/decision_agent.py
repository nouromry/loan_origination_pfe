# src/agents/decision_agent.py

"""
Decision Agent — Makes the final credit decision.

DETERMINISTIC PIPELINE (overrides BaseAgent.run()):
  1. apply_hard_rules tool → immediate reject if any fail
  2. calculate_interest_rate tool → based on risk category
  3. LLM writes explanation (the only LLM call)
  4. render_decision_letter tool → HTML + PDF
  5. save_application + save_decision tools → PostgreSQL

Same pattern as RiskAssessmentAgent: tools are called deterministically,
LLM only handles the qualitative writing.
"""

import json
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent
from src.tools.decision_tools import (
    apply_hard_rules,
    calculate_interest_rate,
    render_decision_letter,
)
from src.tools.db_tools import save_application, save_decision


class DecisionAgent(BaseAgent):
    """
    Makes the final loan decision.
    Applies hard rules, calculates interest rate, generates the decision letter,
    saves to DB.
    """

    def get_tools(self) -> List[BaseTool]:
        return [
            apply_hard_rules,
            calculate_interest_rate,
            render_decision_letter,
            save_application,
            save_decision,
        ]

    def get_prompt_key(self) -> str:
        return "decision_agent"

    # ------------------------------------------------------------------
    # Override BaseAgent.run() with deterministic pipeline
    # ------------------------------------------------------------------
    def run(self, goal: str, **ctx) -> Dict[str, Any]:
        """
        Custom pipeline:
          1. Hard rules check
          2. Interest rate calculation
          3. Decision logic (deterministic)
          4. LLM explanation
          5. Letter rendering
          6. DB save
        """

        loan_type = ctx.get("loan_type", "personal")
        scoring = ctx.get("scoring_result", {})
        risk = ctx.get("risk_result", {})

        # ----- Step 1: Hard rules -----
        hard_rules_result = self._check_hard_rules(ctx, scoring, loan_type)

        if not hard_rules_result["passed"]:
            # Immediate rejection
            explanation = self._write_explanation(ctx, "rejected", hard_rules_result, {}, scoring, risk)
            result = {
                "decision": "rejected",
                "reason": "; ".join(hard_rules_result["failed_rules"]),
                "failed_rules": hard_rules_result["failed_rules"],
                "reason_codes": hard_rules_result["failed_rules"],
                "explanation": explanation,
            }
            # Still render letter + save DB for rejections
            self._render_letter(ctx, result)
            self._save_to_db(ctx, result, scoring, risk)
            return result

        # ----- Step 2: Interest rate -----
        risk_category = scoring.get("risk_category", "unknown")
        rate_result = calculate_interest_rate.invoke({
            "loan_type": loan_type,
            "risk_category": risk_category,
        })

        # ----- Step 3: Decision logic (deterministic) -----
        decision, approved_amount, conditions = self._decide(ctx, scoring, risk, rate_result)

        # ----- Step 4: LLM explanation -----
        explanation = self._write_explanation(ctx, decision, hard_rules_result, rate_result, scoring, risk)

        # ----- Build result -----
        result = {
            "decision": decision,
            "interest_rate": rate_result.get("interest_rate"),
            "rate_breakdown": rate_result.get("breakdown"),
            "approved_amount": approved_amount,
            "conditions": conditions,
            "explanation": explanation,
        }

        # ----- Step 5: Letter -----
        self._render_letter(ctx, result)

        # ----- Step 6: DB save -----
        self._save_to_db(ctx, result, scoring, risk)

        return result

    # ------------------------------------------------------------------
    # Step 1: Hard rules
    # ------------------------------------------------------------------
    def _check_hard_rules(self, ctx: dict, scoring: dict, loan_type: str) -> dict:
        """Call the apply_hard_rules tool."""
        params = {
            "loan_type": loan_type,
            "dti_ratio": scoring.get("dti"),
            "credit_score": ctx.get("credit_score"),
        }
        # Business-specific
        if loan_type == "business":
            params["dscr"] = ctx.get("dscr")
            params["current_ratio"] = ctx.get("current_ratio")
            params["years_in_operation"] = ctx.get("years_in_operation")

        return apply_hard_rules.invoke(params)

    # ------------------------------------------------------------------
    # Step 3: Decision logic
    # ------------------------------------------------------------------
    def _decide(self, ctx: dict, scoring: dict, risk: dict, rate: dict) -> tuple:
        """
        Deterministic decision matrix.
        Returns (decision, approved_amount, conditions).
        """
        credit_score = ctx.get("credit_score") or 0
        loan_amount = ctx.get("loan_amount", 0)
        risk_category = scoring.get("risk_category", "unknown")
        risk_level = risk.get("risk_level", risk_category)  # business uses risk_result

        failed_ratios = risk.get("failed_count", 0)

        # Low risk path
        if risk_level == "low" and credit_score >= 700:
            return "approved", loan_amount, []

        # Medium risk path
        if risk_level in ("low", "medium") and credit_score >= 650 and failed_ratios <= 1:
            if credit_score >= 700:
                return "approved", loan_amount, []
            else:
                reduced = round(loan_amount * 0.8, 2)
                return "conditional", reduced, [
                    "Reduced to 80% of requested amount",
                    "Additional income verification may be requested",
                ]

        # High risk but salvageable
        if credit_score >= 620 and failed_ratios <= 1:
            reduced = round(loan_amount * 0.6, 2)
            return "conditional", reduced, [
                "Reduced to 60% of requested amount",
                "Guarantor or collateral required",
                "Higher interest rate applied",
            ]

        # Edge case: unknown risk but decent credit
        if risk_category == "unknown" and credit_score >= 650:
            return "conditional", loan_amount, [
                "ML scoring unavailable — manual underwriting review required",
            ]

        # Reject
        return "rejected", 0, []

    # ------------------------------------------------------------------
    # Step 4: LLM explanation
    # ------------------------------------------------------------------
    def _write_explanation(self, ctx: dict, decision: str, hard_rules: dict,
                           rate: dict, scoring: dict, risk: dict) -> str:
        """Use LLM to write a human-friendly explanation."""
        name = ctx.get("name", "Applicant")
        loan_amount = ctx.get("loan_amount", 0)
        loan_type = ctx.get("loan_type", "personal")

        facts = (
            f"Decision: {decision.upper()}\n"
            f"Applicant: {name}\n"
            f"Loan: {loan_amount:,.0f} TND ({loan_type})\n"
            f"Credit score: {ctx.get('credit_score', 'N/A')}\n"
            f"DTI: {scoring.get('dti', 'N/A')}\n"
            f"Risk category: {scoring.get('risk_category', 'N/A')}\n"
        )

        if hard_rules.get("failed_rules"):
            facts += f"Failed hard rules: {', '.join(hard_rules['failed_rules'])}\n"
        if rate:
            facts += f"Interest rate: {rate.get('breakdown', 'N/A')}\n"
        if risk.get("analysis"):
            facts += f"Risk analysis summary: {risk['analysis'][:200]}...\n"

        prompt = (
            f"Write a 2-paragraph explanation for the customer about their loan decision.\n\n"
            f"Facts:\n{facts}\n\n"
            f"If approved: be warm and congratulatory. Mention key terms.\n"
            f"If conditional: be encouraging, explain what's needed.\n"
            f"If rejected: be empathetic, explain why, suggest improvements.\n"
            f"Address the customer by name. Plain prose, no bullet points."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a bank loan advisor writing to a customer. Be warm, clear, professional."),
                HumanMessage(content=prompt),
            ])
            return response.content
        except Exception as e:
            # Deterministic fallback
            if decision == "approved":
                return f"Congratulations {name}! Your loan of {loan_amount:,.0f} TND has been approved."
            elif decision == "conditional":
                return f"Good news {name}, your loan has been conditionally approved. Please contact your advisor for next steps."
            else:
                return f"Thank you {name} for your application. Unfortunately, we are unable to approve your loan at this time."

    # ------------------------------------------------------------------
    # Step 5: Letter rendering
    # ------------------------------------------------------------------
    def _render_letter(self, ctx: dict, result: dict) -> None:
        """Render decision letter HTML + PDF."""
        try:
            decision = result.get("decision", "rejected")
            letter_data = {
                "application_id": ctx.get("application_id"),
                "name": ctx.get("name", "Applicant"),
                "decision": decision,
                "loan_type": ctx.get("loan_type", "personal"),
                "loan_amount": ctx.get("loan_amount", 0),
                "loan_purpose_category": ctx.get("loan_purpose_category"),
                "preferred_currency": "TND",
            }

            if decision in ("approved", "conditional"):
                amount = result.get("approved_amount", ctx.get("loan_amount", 0))
                rate = result.get("interest_rate", 0)
                months = ctx.get("loan_term_months", 0)

                letter_data["offered_amount"] = amount
                letter_data["interest_rate"] = rate
                letter_data["offered_term_months"] = months

                # Amortization formula
                if rate and rate > 0 and months and months > 0:
                    mr = rate / 12
                    letter_data["monthly_payment"] = round(
                        amount * (mr * (1 + mr) ** months) / ((1 + mr) ** months - 1), 2
                    )
                else:
                    letter_data["monthly_payment"] = round(amount / max(months, 1), 2)

            if decision == "conditional":
                letter_data["conditions"] = result.get("conditions", [])

            if decision == "rejected":
                letter_data["reason_codes"] = result.get("reason_codes",
                    result.get("failed_rules", ["does_not_meet_criteria"]))

            letter_result = render_decision_letter.invoke({"decision_data": letter_data})

            if letter_result.get("success"):
                result["letter_html_path"] = letter_result.get("html_path")
                result["letter_pdf_path"] = letter_result.get("pdf_path")

        except Exception as e:
            result["letter_error"] = str(e)

    # ------------------------------------------------------------------
    # Step 6: DB persistence
    # ------------------------------------------------------------------
    def _save_to_db(self, ctx: dict, result: dict, scoring: dict, risk: dict) -> None:
        """Save application + decision to PostgreSQL (best-effort)."""
        try:
            save_application.invoke({"state": ctx})
        except Exception:
            pass  # Non-blocking

        try:
            save_decision.invoke({
                "application_id": ctx.get("application_id"),
                "decision_result": result,
                "scoring_result": scoring,
                "risk_result": risk,
            })
        except Exception:
            pass  # Non-blocking