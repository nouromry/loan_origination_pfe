# src/tools/rag_tools.py

import os
from langchain_core.tools import tool
from typing import Optional

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------
# Module-level singleton for ChromaDB client
# ---------------------------------------------------------------
_client = None
_policies = None
_benchmarks = None


def _get_collections():
    """Lazy-initialize ChromaDB collections."""
    global _client, _policies, _benchmarks

    if _client is None:
        if chromadb is None:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")
        chroma_path = os.getenv("CHROMADB_PATH", "./storage/chromadb")
        _client = chromadb.PersistentClient(path=chroma_path)
        _policies = _client.get_or_create_collection("loan_policies")
        _benchmarks = _client.get_or_create_collection("industry_benchmarks")

    return _policies, _benchmarks


def load_policies(filepath: str = "data/policies/loan_policies.txt") -> int:
    """Load loan policies into ChromaDB. Returns count of sections loaded."""
    policies, _ = _get_collections()

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = [s.strip() for s in content.split('\n\n') if s.strip()]
    documents = []
    ids = []

    for i, section in enumerate(sections):
        documents.append(section)
        ids.append(f"policy_{i}")

    if documents:
        policies.upsert(documents=documents, ids=ids)

    return len(documents)


def load_benchmarks(filepath: str = "data/benchmarks/industry_benchmarks.csv") -> int:
    """Load industry benchmarks CSV into ChromaDB. Returns count loaded."""
    if pd is None:
        raise ImportError("pandas is not installed. Run: pip install pandas")

    _, benchmarks = _get_collections()
    df = pd.read_csv(filepath)

    documents = []
    ids = []
    metadatas = []

    for _, row in df.iterrows():
        doc = (
            f"Industry: {row['industry']}\n"
            f"Median DSCR: {row['median_dscr']}\n"
            f"Median Current Ratio: {row['median_current_ratio']}\n"
            f"Median Net Profit Margin: {row['median_npm']}\n"
            f"Median Debt-to-Equity: {row['median_debt_to_equity']}\n"
            f"Default Rate: {row['default_rate']}\n"
            f"Risk Score: {row['risk_score']}"
        )
        documents.append(doc)
        ids.append(f"industry_{row['industry'].replace(' ', '_')}")
        metadatas.append({
            "industry": str(row['industry']),
            "median_dscr": float(row['median_dscr']),
            "median_current_ratio": float(row['median_current_ratio']),
            "median_npm": float(row['median_npm']),
            "median_debt_to_equity": float(row['median_debt_to_equity']),
            "default_rate": float(row['default_rate']),
            "risk_score": int(row['risk_score']),
        })

    if documents:
        benchmarks.upsert(documents=documents, ids=ids, metadatas=metadatas)

    return len(documents)


@tool
def query_policy(question: str) -> str:
    """Query loan policies from the knowledge base.
    
    Uses semantic search over ChromaDB to find the most relevant
    policy sections for the given question. Returns up to 3 matches.
    """
    try:
        policies, _ = _get_collections()
        results = policies.query(query_texts=[question], n_results=3)

        if results['documents'] and results['documents'][0]:
            return "\n\n---\n\n".join(results['documents'][0])

        return "No relevant policy found for this question."
    except Exception as e:
        return f"Policy query failed: {str(e)}"


@tool
def query_benchmarks(industry: str, metric: Optional[str] = None) -> str:
    """Query industry benchmarks for ratio comparison.
    
    Returns benchmark data for the specified industry.
    Optionally filter by a specific metric (e.g., 'dscr', 'current_ratio').
    """
    try:
        _, benchmarks = _get_collections()
        query = industry
        if metric:
            query += f" {metric}"

        results = benchmarks.query(query_texts=[query], n_results=1)

        if results['documents'] and results['documents'][0]:
            return results['documents'][0][0]

        return f"No benchmark data found for industry: {industry}"
    except Exception as e:
        return f"Benchmark query failed: {str(e)}"
