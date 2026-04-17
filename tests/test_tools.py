# tests/test_tools.py

"""
Unit tests for all pure-function tools.
These tests require NO LLM, NO API, NO external services.
Run with: python -m pytest tests/test_tools.py -v
"""

import pytest


# ---------------------------------------------------------------
# Extraction Tools
# ---------------------------------------------------------------
class TestExtractionTools:

    def test_extract_national_id_valid(self):
        from src.tools.extraction_tools import extract_national_id
        assert extract_national_id.invoke({"message": "my ID is 12345678"}) == "12345678"

    def test_extract_national_id_none(self):
        from src.tools.extraction_tools import extract_national_id
        assert extract_national_id.invoke({"message": "hello there"}) is None

    def test_extract_national_id_ignores_longer(self):
        from src.tools.extraction_tools import extract_national_id
        # 9+ digits should not match
        result = extract_national_id.invoke({"message": "amount is 123456789"})
        assert result is None

    def test_extract_email_valid(self):
        from src.tools.extraction_tools import extract_email
        assert extract_email.invoke({"message": "my email is ahmed@gmail.com"}) == "ahmed@gmail.com"

    def test_extract_email_none(self):
        from src.tools.extraction_tools import extract_email
        assert extract_email.invoke({"message": "no email here"}) is None

    def test_extract_phone_valid(self):
        from src.tools.extraction_tools import extract_phone
        assert extract_phone.invoke({"message": "call me at 98765432"}) == "98765432"

    def test_extract_phone_with_country_code(self):
        from src.tools.extraction_tools import extract_phone
        assert extract_phone.invoke({"message": "+216 98765432"}) == "98765432"

    def test_extract_amount_plain(self):
        from src.tools.extraction_tools import extract_amount
        assert extract_amount.invoke({"message": "I need 50000"}) == 50000.0

    def test_extract_amount_with_k(self):
        from src.tools.extraction_tools import extract_amount
        assert extract_amount.invoke({"message": "50K please"}) == 50000.0

    def test_extract_amount_with_m(self):
        from src.tools.extraction_tools import extract_amount
        assert extract_amount.invoke({"message": "1.5M"}) == 1500000.0

    def test_extract_amount_with_commas(self):
        from src.tools.extraction_tools import extract_amount
        assert extract_amount.invoke({"message": "50,000 TND"}) == 50000.0

    def test_extract_loan_term_months(self):
        from src.tools.extraction_tools import extract_loan_term
        assert extract_loan_term.invoke({"message": "36 months"}) == 36

    def test_extract_loan_term_years(self):
        from src.tools.extraction_tools import extract_loan_term
        assert extract_loan_term.invoke({"message": "3 years"}) == 36


# ---------------------------------------------------------------
# Financial Tools
# ---------------------------------------------------------------
class TestFinancialTools:

    def test_dti_normal(self):
        from src.tools.financial_tools import calculate_dti
        result = calculate_dti.invoke({
            "loan_amount": 30000, "term_months": 36, "monthly_income": 3000
        })
        assert result["dti"] == round(833.33 / 3000, 4)
        assert result["dti_passed"] is True

    def test_dti_fails(self):
        from src.tools.financial_tools import calculate_dti
        result = calculate_dti.invoke({
            "loan_amount": 100000, "term_months": 12, "monthly_income": 3000
        })
        # 100000/12 = 8333.33, DTI = 2.7778 — way over 0.45
        assert result["dti_passed"] is False

    def test_dti_zero_income(self):
        from src.tools.financial_tools import calculate_dti
        result = calculate_dti.invoke({
            "loan_amount": 30000, "term_months": 36, "monthly_income": 0
        })
        assert result["dti_passed"] is False

    def test_dscr_normal(self):
        from src.tools.financial_tools import calculate_dscr
        result = calculate_dscr.invoke({
            "net_income": 120000, "loan_amount": 50000, "term_months": 36
        })
        # annual_debt = (50000/36)*12 = 16666.67, DSCR = 120000/16666.67 = 7.2
        assert result["dscr_passed"] is True

    def test_current_ratio(self):
        from src.tools.financial_tools import calculate_current_ratio
        result = calculate_current_ratio.invoke({
            "current_assets": 100000, "current_liabilities": 80000
        })
        assert result["current_ratio"] == 1.25
        assert result["current_ratio_passed"] is True

    def test_debt_to_equity(self):
        from src.tools.financial_tools import calculate_debt_to_equity
        result = calculate_debt_to_equity.invoke({
            "total_liabilities": 200000, "equity": 100000
        })
        assert result["debt_to_equity"] == 2.0
        assert result["debt_to_equity_passed"] is True

    def test_net_profit_margin(self):
        from src.tools.financial_tools import calculate_net_profit_margin
        result = calculate_net_profit_margin.invoke({
            "net_income": 50000, "total_revenue": 400000
        })
        assert result["net_profit_margin"] == 0.125
        assert result["npm_passed"] is True

    def test_quick_ratio(self):
        from src.tools.financial_tools import calculate_quick_ratio
        result = calculate_quick_ratio.invoke({
            "current_assets": 100000, "inventory": 20000, "current_liabilities": 80000
        })
        assert result["quick_ratio"] == 1.0
        assert result["quick_ratio_passed"] is True


# ---------------------------------------------------------------
# Decision Tools
# ---------------------------------------------------------------
class TestDecisionTools:

    def test_hard_rules_pass(self):
        from src.tools.decision_tools import apply_hard_rules
        result = apply_hard_rules.invoke({
            "loan_type": "personal", "dti_ratio": 0.30, "credit_score": 750
        })
        assert result["passed"] is True
        assert len(result["failed_rules"]) == 0

    def test_hard_rules_fail_dti(self):
        from src.tools.decision_tools import apply_hard_rules
        result = apply_hard_rules.invoke({
            "loan_type": "personal", "dti_ratio": 0.55, "credit_score": 750
        })
        assert result["passed"] is False
        assert len(result["failed_rules"]) == 1

    def test_hard_rules_fail_credit(self):
        from src.tools.decision_tools import apply_hard_rules
        result = apply_hard_rules.invoke({
            "loan_type": "personal", "dti_ratio": 0.30, "credit_score": 500
        })
        assert result["passed"] is False

    def test_hard_rules_business_dscr(self):
        from src.tools.decision_tools import apply_hard_rules
        result = apply_hard_rules.invoke({
            "loan_type": "business", "dti_ratio": 0.30,
            "credit_score": 750, "dscr": 0.90
        })
        assert result["passed"] is False
        assert any("DSCR" in r for r in result["failed_rules"])

    def test_interest_rate_personal_low(self):
        from src.tools.decision_tools import calculate_interest_rate
        result = calculate_interest_rate.invoke({
            "loan_type": "personal", "risk_category": "low"
        })
        # 8.5% + 0.5% + 1.0% = 10.0%
        assert result["interest_rate"] == 0.10

    def test_interest_rate_business_high(self):
        from src.tools.decision_tools import calculate_interest_rate
        result = calculate_interest_rate.invoke({
            "loan_type": "business", "risk_category": "high"
        })
        # 8.5% + 4.0% + 0.0% = 12.5%
        assert result["interest_rate"] == 0.125


# ---------------------------------------------------------------
# Global State Helpers
# ---------------------------------------------------------------
class TestGlobalStateHelpers:

    def test_get_fields_personal(self):
        from src.models.global_state import get_fields_to_ask
        state = {"loan_type": "personal"}
        missing = get_fields_to_ask(state)
        assert "national_id" in missing
        assert "number_of_dependents" in missing
        assert "industry" not in missing  # business-only field

    def test_get_fields_business(self):
        from src.models.global_state import get_fields_to_ask
        state = {"loan_type": "business"}
        missing = get_fields_to_ask(state)
        assert "industry" in missing
        assert "number_of_dependents" not in missing  # personal-only field

    def test_compute_tier_personal(self):
        from src.models.global_state import compute_tier
        assert compute_tier({"loan_type": "personal"}) == "personal"

    def test_compute_tier_small(self):
        from src.models.global_state import compute_tier
        assert compute_tier({"loan_type": "business", "loan_amount": 30000}) == "small"

    def test_compute_tier_medium(self):
        from src.models.global_state import compute_tier
        assert compute_tier({"loan_type": "business", "loan_amount": 150000}) == "medium"

    def test_compute_tier_large(self):
        from src.models.global_state import compute_tier
        assert compute_tier({"loan_type": "business", "loan_amount": 400000}) == "large"

    def test_compute_tier_very_large(self):
        from src.models.global_state import compute_tier
        assert compute_tier({"loan_type": "business", "loan_amount": 600000}) == "very_large"

    def test_validate_document_extraction_is_data_driven_not_file_driven(self):
        from src.models.global_state import validate_document_extraction

        state = {
            "loan_type": "personal",
            "compliance_tier": "personal",
            "monthly_income": 3169,
            "monthly_cash_flow": 1999,
            "document_result": {
                "cin.pdf": {"type": "cin_card", "result": {"cin_national_id": "12714074"}}
            },
        }
        report = validate_document_extraction(state)
        assert report["has_issues"] is False
        assert "salary_slip" in report["missing_required_documents"]
        assert "bank_statement" in report["missing_required_documents"]
