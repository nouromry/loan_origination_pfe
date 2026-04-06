#!/usr/bin/env python3
# main.py

"""
AXE Finance — Generative AI Loan Origination System
Entry point. Provides:
  - Interactive chat loop
  - --setup flag for first-time ChromaDB initialization
  - --test flag for running a quick smoke test
"""

import sys
import uuid
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph.orchestrator import app_graph
from src.models.global_state import GlobalState


def create_initial_state() -> GlobalState:
    """Create a fresh GlobalState for a new application."""
    app_id = f"APP_{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now().isoformat()

    return {
        "application_id": app_id,
        "loan_type": None,
        "compliance_tier": None,
        "preferred_currency": "TND",
        "intent": None,
        "stage": "collecting",
        "application_status": "collecting_data",
        "rejection_reason": None,
        "credit_score_fetched": False,
        "documents_uploaded": False,
        "document_result": {},
        "scoring_result": {},
        "risk_result": {},
        "decision_result": {},
        "thought_steps": [],
        "messages": [],
        "last_response": "",
        "status_message": None,
        "created_at": now,
        "updated_at": now,
    }


def setup_knowledge_base():
    """First-time setup: load policies and benchmarks into ChromaDB."""
    print("Setting up knowledge base...")

    from src.tools.rag_tools import load_policies, load_benchmarks

    try:
        count = load_policies()
        print(f"  ✅ Loaded {count} policy sections into ChromaDB")
    except FileNotFoundError:
        print("  ⚠️  data/policies/loan_policies.txt not found — skipping policies")

    try:
        count = load_benchmarks()
        print(f"  ✅ Loaded {count} industry benchmarks into ChromaDB")
    except FileNotFoundError:
        print("  ⚠️  data/benchmarks/industry_benchmarks.csv not found — skipping benchmarks")

    print("Setup complete!")


def run_chat():
    """Interactive chat loop."""
    state = create_initial_state()
    app_id = state["application_id"]

    print("=" * 60)
    print("  AXE Finance — AI Loan Assistant")
    print(f"  Application: {app_id}")
    print("  Type 'quit' to exit, 'state' to see current state")
    print("  Type 'upload' to simulate document upload")
    print("=" * 60)
    print()

    # Greeting
    greeting = (
        "Hello! Welcome to AXE Finance. I'm your AI loan assistant. "
        "How can I help you today?"
    )
    print(f"🏦 Assistant: {greeting}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Thank you for using AXE Finance. Goodbye!")
            break

        if user_input.lower() == "state":
            _print_state_summary(state)
            continue

        if user_input.lower() == "upload":
            state["documents_uploaded"] = True
            print("📎 [Documents marked as uploaded]")
            user_input = "I've uploaded my documents"

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        state["updated_at"] = datetime.now().isoformat()

        # Run through the graph
        try:
            result = app_graph.invoke(state)

            if not isinstance(result, dict):
                print("\n❌ Graph returned invalid state")
                continue

            # Merge safely (prevents losing messages)
            state.update(result)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let's try again.\n")
            continue

        # Display response
        response = state.get("last_response", "I'm sorry, something went wrong.")
        print(f"\n🏦 Assistant: {response}\n")

        # Show thought steps if any new ones
        thoughts = state.get("thought_steps", [])
        if thoughts:
            latest = thoughts[-1] if thoughts else ""
            print(f"   💭 [{latest}]")
            print()

        # Check if application is complete
        if state.get("stage") == "complete":
            print("=" * 60)
            print("  Application processing complete!")
            print(f"  Status: {state.get('application_status', 'unknown').upper()}")
            print("=" * 60)


def _print_state_summary(state: GlobalState):
    """Print a formatted summary of the current state."""
    from src.models.global_state import get_status_summary
    
    print("\n" + "─" * 50)
    print("  Current Application State")
    print("─" * 50)
    print(f"  App ID:      {state.get('application_id')}")
    print(f"  Loan Type:   {state.get('loan_type', 'not set')}")
    print(f"  Status:      {state.get('application_status', 'collecting_data')}")
    print(f"  Stage:       {state.get('stage', 'collecting')}")
    print(f"  Tier:        {state.get('compliance_tier', 'not set')}")
    print()

    # Show collected fields
    key_fields = [
        "national_id", "email", "phone", "loan_amount",
        "loan_term_months", "credit_score",
    ]

    # Add type-specific fields
    loan_type = state.get("loan_type")
    if loan_type == "personal":
        key_fields += ["date_of_birth", "marital_status", "housing_status", "number_of_dependents"]
    elif loan_type == "business":
        key_fields += ["industry", "number_of_employees", "applicant_ownership_percentage"]

    print("  Collected Fields:")
    for f in key_fields:
        val = state.get(f)
        status = f"✅ {val}" if val is not None else "❌ missing"
        print(f"    {f}: {status}")

    print()
    print("  Pipeline Progress:")
    print(f"    Documents uploaded:  {'✅' if state.get('documents_uploaded') else '❌'}")
    print(f"    Documents processed: {'✅' if state.get('document_result') else '❌'}")
    print(f"    Scoring done:        {'✅' if state.get('scoring_result') else '❌'}")
    print(f"    Risk assessed:       {'✅' if state.get('risk_result') else '❌' if loan_type == 'business' else '➖ N/A'}")
    print(f"    Decision made:       {'✅' if state.get('decision_result') else '❌'}")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXE Finance AI Loan Assistant")
    parser.add_argument("--setup", action="store_true", help="Load knowledge base into ChromaDB")
    parser.add_argument("--test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    if args.setup:
        setup_knowledge_base()
    elif args.test:
        print("Running smoke test...")
        state = create_initial_state()
        state["messages"].append(HumanMessage(content="Hi, I need a personal loan of 30000 TND"))
        try:
            result = app_graph.invoke(state)
            print(f"✅ Graph executed successfully")
            print(f"   Intent: {result.get('intent')}")
            print(f"   Response: {result.get('last_response', '')[:100]}...")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        run_chat()