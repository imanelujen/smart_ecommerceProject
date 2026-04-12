"""
module5/mcp/mcp_server_llm.py
-------------------------------
MCP Server: LLM — exposes the 4 LangChain chains as tools.

Principle of isolation: the LLM server only receives the data it needs
for each task — it never has direct access to the full database.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from module5.llm_client import LLMClient
from module5.chains import (
    summarize_product, generate_trend_report,
    build_client_profile, recommend_strategy,
)
from module4.data_loader import load_scored, load_top_k, load_clusters, load_rules


_client = None

def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


TOOLS = {
    "summarize_product": {
        "description": "Generates a 2-3 sentence marketing summary for a single product.",
        "parameters": {
            "product": {"type": "object", "description": "Product dict with title, description, price, rating"},
        },
        "handler": lambda p: summarize_product(_get_client(), p["product"]),
    },
    "generate_trend_report": {
        "description": "Generates a weekly market trend report from all scraped products.",
        "parameters": {
            "top_k": {"type": "integer", "default": 20, "description": "Number of top products to highlight"},
        },
        "handler": lambda p: _trend_report(p.get("top_k", 20)),
    },
    "build_client_profile": {
        "description": "Builds a target customer persona from the Top-K product list.",
        "parameters": {
            "top_k": {"type": "integer", "default": 20},
        },
        "handler": lambda p: build_client_profile(_get_client(), load_top_k().head(p.get("top_k", 20))),
    },
    "recommend_strategy": {
        "description": "Generates actionable marketing strategy recommendations.",
        "parameters": {},
        "handler": lambda p: recommend_strategy(_get_client(), load_clusters(), load_rules()),
    },
    "answer_question": {
        "description": "Answers a free-form business question about the product data.",
        "parameters": {
            "question": {"type": "string", "description": "The user's question"},
        },
        "handler": lambda p: _answer_question(p["question"]),
    },
}


def call_tool(tool_name: str, params: dict) -> dict:
    if tool_name not in TOOLS:
        return {"error": f"Unknown LLM tool: {tool_name}"}
    try:
        result = TOOLS[tool_name]["handler"](params)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def _trend_report(top_k: int) -> str:
    df  = load_scored()
    top = load_top_k().head(top_k)
    return generate_trend_report(_get_client(), df, top)


def _answer_question(question: str) -> str:
    """
    Answers a free-form question by injecting a compact data context.
    This is the chatbot backend — implements RAG-lite with inline data.
    """
    df  = load_scored()
    top = load_top_k().head(10)

    context = f"""Données disponibles :
- {len(df)} produits scrapés au total
- Score moyen : {df['score'].mean():.3f}
- Prix médian : {df['price'].median():.2f}$
- Note moyenne : {df['rating'].mean():.1f}/5

Top 5 produits :
""" + "\n".join(
        f"  {i+1}. {r.get('title','?')} — score={r.get('score',0):.3f}, "
        f"prix={r.get('price',0):.2f}$, rating={r.get('rating','N/A')}"
        for i, (_, r) in enumerate(top.head(5).iterrows())
    )

    prompt = f"""Contexte marché :\n{context}\n\nQuestion : {question}\n
Réponds de façon concise et orientée décision business."""
    return _get_client().chat(
        prompt,
        system="Tu es un assistant BI spécialisé en analyse e-commerce.",
        max_tokens=400,
    )