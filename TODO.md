# AXE Finance Fix: Credit Score Query Handling
Current Issue: "tell me about your credit score here" → Asks for national ID instead of explaining.

## Plan Summary
- Fix triage: Route credit score questions to credit_workflow (collect/explain) not policy_question.
- Strengthen responder: Force use of policy_answer; add credit score explanation fallback.
- Test: Verify no premature collection.

## Steps (Completed: ✅ | Pending: ⭕)
⭕ 1. Update src/prompts/agent_prompts.yaml:
       - triage_agent: Add credit score examples → credit_workflow.
       - responder_node: Mandate "use policy_answer", credit score explanation.

⭕ 2. Test classification:
       - python main.py → "tell me about your credit score here"
       - Expected: intent=credit_workflow or policy_answer used properly.

⭕ 3. If still issues: Add debug print in triage_node.py for intent logging.

⭕ 4. Full test: Complete loan flow post-fix.

⭕ 5. Verify ChromaDB: python main.py --setup
