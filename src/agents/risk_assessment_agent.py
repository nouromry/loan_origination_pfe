# src/agents/risk_assessment_agent.py

"""
Risk Assessment Agent — Business Loans Only.

HYBRID APPROACH:
  - Tools handle pure math (ratio calculations) + RAG (benchmark lookup)
  - Agent orchestrates the pipeline deterministically (no ReAct guessing)
  - LLM is called ONCE at the end for qualitative analysis writing

This overrides BaseAgent.run() with a custom pipeline instead of
relying on the generic ReAct loop, because we need deterministic
ordering: calculate → fetch benchmarks → compare → write analysis.
"""

import json
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage

from .base_agent import BaseAgent
from src.tools.financial_tools import (
    calculate_dscr,
    calculate_current_ratio,
    calculate_debt_to_equity,
    calculate_net_profit_margin,
    calculate_quick_ratio,
)
from src.tools.rag_tools import query_benchmarks


class RiskAssessmentAgent(BaseAgent):
    """
    Business loans ONLY.
    Calculates all 5 business financial ratios, compares them against
    industry benchmarks via RAG, and writes a qualitative risk analysis.

    Uses the 70B model for high-quality reasoning and writing.
    """

    def get_tools(self) -> List[BaseTool]:
        return [
            calculate_dscr,
            calculate_current_ratio,
            calculate_debt_to_equity,
            calculate_net_profit_margin,
            calculate_quick_ratio,
            query_benchmarks,
        ]

    def get_prompt_key(self) -> str:
        return "risk_assessment_agent"

    # ------------------------------------------------------------------
    # Override BaseAgent.run() with our deterministic pipeline
    # ------------------------------------------------------------------
    def run(self, goal: str, **context) -> Dict[str, Any]:
        """
        Custom pipeline (NOT the generic ReAct loop):
          1. Calculate all 5 ratios via tools (deterministic)
          2. Query RAG benchmarks for the industry
          3. Compare each ratio vs thresholds + benchmarks
          4. Determine risk level (pure logic)
          5. LLM writes qualitative analysis (the only LLM call)
        """

        # Step 1: Calculate ratios using tools directly
        ratios = self._calculate_all_ratios(context)

        # Step 2: Fetch industry benchmarks from RAG
        industry = context.get("industry", "General")
        benchmarks = self._fetch_benchmarks(industry)

        # Step 3: Compare ratios vs thresholds + benchmarks
        comparisons = self._compare_ratios(ratios, benchmarks)
        summary = comparisons.pop("_summary", {})

        # Step 4: Determine risk level
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        total = summary.get("total_ratios", 0)

        if failed == 0 and total > 0:
            risk_level = "low"
            recommendation = "approve"
        elif failed <= 1:
            risk_level = "medium"
            recommendation = "conditional"
        else:
            risk_level = "high"
            recommendation = "reject"

        # Step 5: LLM qualitative analysis
        loan_amount = context.get("loan_amount", 0)
        analysis = self._write_analysis(
            industry, loan_amount, ratios, comparisons, benchmarks, summary
        )

        return {
            "ratios": ratios,
            "comparisons": comparisons,
            "benchmarks": benchmarks,
            "passed_count": passed,
            "failed_count": failed,
            "total_ratios": total,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "analysis": analysis,
        }

    # ------------------------------------------------------------------
    # Step 1: Call each ratio tool directly (no ReAct)
    # ------------------------------------------------------------------
    def _calculate_all_ratios(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke each ratio tool with the right arguments."""
        ratios = {}

        # DSCR
        try:
            result = calculate_dscr.invoke({
                "net_income": ctx.get("net_income") or 0,
                "loan_amount": ctx.get("loan_amount") or 0,
                "term_months": ctx.get("loan_term_months") or 1,
            })
            if result.get("dscr") is not None:
                ratios["dscr"] = result
        except Exception as e:
            ratios["dscr"] = {"error": str(e)}

        # Current Ratio
        try:
            result = calculate_current_ratio.invoke({
                "current_assets": ctx.get("current_assets") or 0,
                "current_liabilities": ctx.get("current_liabilities") or 1,
            })
            if result.get("current_ratio") is not None:
                ratios["current_ratio"] = result
        except Exception as e:
            ratios["current_ratio"] = {"error": str(e)}

        # Debt-to-Equity
        try:
            result = calculate_debt_to_equity.invoke({
                "total_liabilities": ctx.get("total_liabilities") or 0,
                "equity": ctx.get("equity") or 1,
            })
            if result.get("debt_to_equity") is not None:
                ratios["debt_to_equity"] = result
        except Exception as e:
            ratios["debt_to_equity"] = {"error": str(e)}

        # Net Profit Margin
        try:
            result = calculate_net_profit_margin.invoke({
                "net_income": ctx.get("net_income") or 0,
                "total_revenue": ctx.get("total_revenue") or 1,
            })
            if result.get("net_profit_margin") is not None:
                ratios["net_profit_margin"] = result
        except Exception as e:
            ratios["net_profit_margin"] = {"error": str(e)}

        # Quick Ratio (inventory defaults to 0 inside the tool)
        try:
            result = calculate_quick_ratio.invoke({
                "current_assets": ctx.get("current_assets") or 0,
                "inventory": 0.0,  # We don't track inventory separately
                "current_liabilities": ctx.get("current_liabilities") or 1,
            })
            if result.get("quick_ratio") is not None:
                ratios["quick_ratio"] = result
        except Exception as e:
            ratios["quick_ratio"] = {"error": str(e)}

        return ratios

    # ------------------------------------------------------------------
    # Step 2: RAG benchmark lookup
    # ------------------------------------------------------------------
    def _fetch_benchmarks(self, industry: str) -> Optional[Dict[str, Any]]:
        """Query ChromaDB for industry benchmarks via the tool."""
        try:
            raw = query_benchmarks.invoke({
                "industry": industry,
            })
            # raw is a string like "Industry: Tech\nMedian DSCR: 1.8\n..."
            # Parse it into a dict
            if isinstance(raw, str) and "No benchmark" not in raw:
                return self._parse_benchmark_text(raw)
            return None
        except Exception as e:
            print(f"[RiskAssessmentAgent] Benchmark query failed: {e}")
            return None

    @staticmethod
    def _parse_benchmark_text(text: str) -> Dict[str, Any]:
        """Parse the benchmark text from ChromaDB into a structured dict."""
        result = {}
        key_map = {
            "Industry": "industry",
            "Median DSCR": "median_dscr",
            "Median Current Ratio": "median_current_ratio",
            "Median Net Profit Margin": "median_npm",
            "Median Debt-to-Equity": "median_debt_to_equity",
            "Default Rate": "default_rate",
            "Risk Score": "risk_score",
        }
        for line in text.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                mapped = key_map.get(key)
                if mapped:
                    try:
                        result[mapped] = float(value)
                    except ValueError:
                        result[mapped] = value
        return result if result else None

    # ------------------------------------------------------------------
    # Step 3: Compare ratios vs thresholds + benchmarks
    # ------------------------------------------------------------------
    def _compare_ratios(
        self,
        ratios: Dict[str, Any],
        benchmarks: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare each ratio against thresholds and benchmarks."""

        settings = self._load_settings()
        thresholds = settings.get("thresholds", {})

        # (ratio_key, threshold_key, direction, benchmark_key)
        ratio_config = [
            ("dscr", "dscr_min", ">=", "median_dscr"),
            ("current_ratio", "current_ratio_min", ">=", "median_current_ratio"),
            ("debt_to_equity", "debt_to_equity_max", "<=", "median_debt_to_equity"),
            ("net_profit_margin", "net_profit_margin_min", ">=", "median_npm"),
            ("quick_ratio", "quick_ratio_min", ">=", None),
        ]

        comparisons = {}
        passed_count = 0
        failed_count = 0

        for ratio_key, threshold_key, direction, benchmark_key in ratio_config:
            ratio_data = ratios.get(ratio_key, {})

            # Handle missing or error
            if not ratio_data or "error" in ratio_data:
                comparisons[ratio_key] = {"status": "missing", "error": ratio_data.get("error")}
                continue

            # Extract the value (tool returns {"dscr": 1.5, "dscr_passed": True})
            value = ratio_data.get(ratio_key)
            tool_passed = ratio_data.get(f"{ratio_key}_passed")
            # NPM uses different key
            if value is None:
                value = ratio_data.get("net_profit_margin")
                tool_passed = ratio_data.get("npm_passed")

            if value is None:
                comparisons[ratio_key] = {"status": "missing"}
                continue

            threshold = thresholds.get(threshold_key)

            # Get benchmark
            benchmark_val = None
            vs_benchmark = None
            if benchmarks and benchmark_key:
                benchmark_val = benchmarks.get(benchmark_key)
                if benchmark_val is not None:
                    vs_benchmark = "above" if value >= benchmark_val else "below"
                    if direction == "<=":
                        # For D/E, below benchmark is good
                        vs_benchmark = "below" if value <= benchmark_val else "above"

            comparisons[ratio_key] = {
                "value": value,
                "passed": tool_passed,
                "threshold": threshold,
                "direction": direction,
                "benchmark": benchmark_val,
                "vs_benchmark": vs_benchmark,
            }

            if tool_passed:
                passed_count += 1
            else:
                failed_count += 1

        comparisons["_summary"] = {
            "total_ratios": passed_count + failed_count,
            "passed": passed_count,
            "failed": failed_count,
        }

        return comparisons

    # ------------------------------------------------------------------
    # Step 5: LLM qualitative analysis (the ONLY LLM call)
    # ------------------------------------------------------------------
    def _write_analysis(
        self,
        industry: str,
        loan_amount: float,
        ratios: Dict[str, Any],
        comparisons: Dict[str, Any],
        benchmarks: Optional[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> str:
        """Use the LLM to write a 3-paragraph qualitative risk analysis."""

        # Build structured context for the LLM
        ratio_lines = []
        for name, comp in comparisons.items():
            if name.startswith("_"):
                continue
            if comp.get("status") == "missing":
                ratio_lines.append(f"- {name}: NOT AVAILABLE")
                continue

            status = "PASS" if comp.get("passed") else "FAIL"
            line = f"- {name}: {comp['value']:.4f} (threshold: {comp['direction']} {comp['threshold']}) → {status}"
            if comp.get("benchmark") is not None:
                line += f" | Industry median: {comp['benchmark']} ({comp['vs_benchmark']} benchmark)"
            ratio_lines.append(line)

        benchmark_text = ""
        if benchmarks:
            dr = benchmarks.get("default_rate")
            rs = benchmarks.get("risk_score")
            if dr is not None:
                benchmark_text = f"\nIndustry default rate: {dr}%. Industry risk score: {rs}/100."

        prompt = (
            f"You are a senior credit analyst writing a risk assessment for a {industry} business "
            f"requesting a loan of {loan_amount:,.0f} TND.\n\n"
            f"Financial Ratios:\n" + "\n".join(ratio_lines) + "\n"
            f"{benchmark_text}\n\n"
            f"Summary: {summary.get('passed', 0)} of {summary.get('total_ratios', 0)} ratios passed.\n\n"
            f"Write exactly 3 paragraphs:\n"
            f"1. STRENGTHS: What ratios look healthy and why.\n"
            f"2. WEAKNESSES: What ratios are concerning and the specific risks.\n"
            f"3. RECOMMENDATION: Overall risk level (Low/Medium/High) and lending recommendation.\n\n"
            f"Be specific with numbers. No bullet points. No markdown headers. Plain prose only."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a senior credit risk analyst. Write concise, factual assessments."),
                HumanMessage(content=prompt),
            ])
            return response.content

        except Exception as e:
            print(f"[RiskAssessmentAgent] LLM analysis failed: {e}")
            # Deterministic fallback
            p = summary.get("passed", 0)
            t = summary.get("total_ratios", 0)
            if p == t and t > 0:
                return f"All {t} financial ratios passed for this {industry} business. The financial profile appears healthy."
            elif p >= t * 0.6:
                return f"{p} of {t} ratios passed. The business shows mixed financial health with some areas needing attention."
            else:
                return f"Only {p} of {t} ratios passed. This {industry} business presents elevated financial risk."