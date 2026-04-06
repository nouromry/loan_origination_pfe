# src/graph/node_factory.py

"""
Singleton Node Factory.

All nodes are registered here. The orchestrator imports from this module
instead of importing individual node files. This provides a single place
to manage node initialization and prevents duplicate agent instantiation.
"""

from src.nodes.triage_node import triage_node
from src.nodes.collect_node import collect_node
from src.nodes.document_node import document_node
from src.nodes.scoring_node import scoring_node
from src.nodes.risk_assessment_node import risk_assessment_node
from src.nodes.decision_node import decision_node
from src.nodes.policy_node import policy_node
from src.nodes.responder_node import responder_node


class NodeFactory:
    """
    Provides access to all node functions.
    Each node function has signature: (GlobalState) -> GlobalState
    """

    @staticmethod
    def triage(state):
        return triage_node(state)

    @staticmethod
    def collect(state):
        return collect_node(state)

    @staticmethod
    def document(state):
        return document_node(state)

    @staticmethod
    def scoring(state):
        return scoring_node(state)

    @staticmethod
    def risk_assessment(state):
        return risk_assessment_node(state)

    @staticmethod
    def decision(state):
        return decision_node(state)

    @staticmethod
    def policy(state):
        return policy_node(state)

    @staticmethod
    def responder(state):
        return responder_node(state)


# Module-level singleton
node_factory = NodeFactory()
