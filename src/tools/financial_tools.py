# src/tools/financial_tools.py

from langchain_core.tools import tool


@tool
def calculate_dti(loan_amount: float, term_months: int, monthly_income: float) -> dict:
    """Calculate the Debt-to-Income (DTI) ratio.
    
    DTI = (loan_amount / term_months) / monthly_income
    Threshold: ≤ 0.45 (45%)
    
    Returns dict with dti value and whether it passed.
    """
    if monthly_income is None or monthly_income <= 0:
        return {"dti": None, "dti_passed": False, "error": "Invalid or missing monthly income"}

    if term_months is None or term_months <= 0:
        return {"dti": None, "dti_passed": False, "error": "Invalid or missing loan term"}

    monthly_payment = loan_amount / term_months
    dti = round(monthly_payment / monthly_income, 4)

    return {
        "dti": dti,
        "monthly_payment": round(monthly_payment, 2),
        "dti_passed": dti <= 0.45,
    }


@tool
def calculate_dscr(net_income: float, loan_amount: float, term_months: int) -> dict:
    """Calculate Debt Service Coverage Ratio (DSCR).
    
    DSCR = net_income / annual_debt_service
    Threshold: ≥ 1.25
    """
    annual_debt_service = (loan_amount / term_months) * 12
    if annual_debt_service <= 0:
        return {"dscr": None, "dscr_passed": False, "error": "Invalid debt service"}

    dscr = round(net_income / annual_debt_service, 4)
    return {
        "dscr": dscr,
        "dscr_passed": dscr >= 1.25,
    }


@tool
def calculate_current_ratio(current_assets: float, current_liabilities: float) -> dict:
    """Calculate Current Ratio (liquidity).
    
    Current Ratio = current_assets / current_liabilities
    Threshold: ≥ 1.0
    """
    if current_liabilities is None or current_liabilities <= 0:
        return {"current_ratio": None, "current_ratio_passed": False, "error": "Invalid current liabilities"}

    ratio = round(current_assets / current_liabilities, 4)
    return {
        "current_ratio": ratio,
        "current_ratio_passed": ratio >= 1.0,
    }


@tool
def calculate_debt_to_equity(total_liabilities: float, equity: float) -> dict:
    """Calculate Debt-to-Equity Ratio (leverage).
    
    D/E = total_liabilities / equity
    Threshold: ≤ 2.0
    """
    if equity is None or equity <= 0:
        return {"debt_to_equity": None, "debt_to_equity_passed": False, "error": "Invalid equity"}

    ratio = round(total_liabilities / equity, 4)
    return {
        "debt_to_equity": ratio,
        "debt_to_equity_passed": ratio <= 2.0,
    }


@tool
def calculate_net_profit_margin(net_income: float, total_revenue: float) -> dict:
    """Calculate Net Profit Margin (profitability).
    
    NPM = net_income / total_revenue
    Threshold: ≥ 0.10 (10%)
    """
    if total_revenue is None or total_revenue <= 0:
        return {"net_profit_margin": None, "npm_passed": False, "error": "Invalid revenue"}

    margin = round(net_income / total_revenue, 4)
    return {
        "net_profit_margin": margin,
        "npm_passed": margin >= 0.10,
    }


@tool
def calculate_quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> dict:
    """Calculate Quick Ratio (acid test / stress liquidity).
    
    Quick Ratio = (current_assets - inventory) / current_liabilities
    Threshold: ≥ 0.8
    """
    if current_liabilities is None or current_liabilities <= 0:
        return {"quick_ratio": None, "quick_ratio_passed": False, "error": "Invalid current liabilities"}

    # Default inventory to 0 if not provided
    inv = inventory if inventory is not None else 0.0
    ratio = round((current_assets - inv) / current_liabilities, 4)
    return {
        "quick_ratio": ratio,
        "quick_ratio_passed": ratio >= 0.8,
    }
