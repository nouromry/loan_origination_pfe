# src/models/agent_state.py

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Isolated scratchpad for each agent's ReAct loop.
    This is NOT the GlobalState — it's a temporary workspace
    that exists only during one agent invocation.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    goal: str
    context: Dict[str, Any]
