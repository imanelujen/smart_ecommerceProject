"""
LLM/mcp/mcp_server_llm.py
-------------------------------
MCP Server: LLM — exposes the 4 LangChain chains as tools.

Principle of isolation: the LLM server only receives the data it needs
for each task — it never has direct access to the full database.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LLM.llm_client import LLMClient
from LLM.chains import (
    BASE_SYSTEM,
    summarize_product,
    generate_trend_report,
    build_client_profile,
    recommend_strategy,
)
from DashboardBI.data_loader import load_scored, load_top_k, load_clusters, load_rules


_client = None

def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _filter_catalog(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply optional category / platform / price filters from the dashboard."""
    if df is None or df.empty:
        return df
    out = df
    pmin, pmax = params.get("price_min"), params.get("price_max")
    if pmin is not None and pmax is not None and "price" in out.columns:
        out = out[(out["price"] >= float(pmin)) & (out["price"] <= float(pmax))]
    cat = params.get("category")
    if cat and str(cat).strip() and str(cat).lower() != "all" and "category" in out.columns:
        out = out[out["category"] == cat]
    plat = params.get("platform")
    if plat and str(plat).strip() and str(plat).lower() != "all" and "platform" in out.columns:
        out = out[out["platform"] == plat]
    return out


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
    "generate_bi_report": {
        "description": (
            "One-shot BI report for the dashboard: top-K digest, market trend, "
            "pricing analysis, cross-sell, or segment strategy."
        ),
        "parameters": {
            "report_kind": {
                "type": "string",
                "description": (
                    "topk_summary | market_trend | pricing | cross_sell | segment"
                ),
            },
            "tone": {"type": "string", "default": "Executive summary"},
            "n_products": {"type": "integer", "default": 10},
            "category": {"type": "string", "default": None},
            "platform": {"type": "string", "default": None},
            "price_min": {"type": "number", "default": None},
            "price_max": {"type": "number", "default": None},
        },
        "handler": lambda p: _generate_bi_report(p),
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


def _generate_bi_report(params: dict) -> str:
    """
    Maps dashboard report types to data + LLM (single audited MCP call).
    """
    kind = (params.get("report_kind") or "market_trend").strip().lower()
    tone = params.get("tone") or "Executive summary"
    n = int(params.get("n_products") or 10)
    n = max(5, min(50, n))

    client = _get_client()
    df_scored = _filter_catalog(load_scored(), params)
    if df_scored.empty:
        df_scored = load_scored()
    df_top_full = _filter_catalog(load_top_k(), params)
    if df_top_full.empty:
        df_top_full = load_top_k()
    df_top = df_top_full.head(n)

    tone_line = (
        f"Format demandé : **{tone}**. Structure claire en markdown "
        f"(titres ##, listes à puces si pertinent)."
    )

    if kind == "topk_summary":
        cols = [
            c
            for c in [
                "title",
                "category",
                "price",
                "rating",
                "review_count",
                "score",
                "shop_name",
                "platform",
                "discount_pct",
            ]
            if c in df_top.columns
        ]
        ctx = df_top[cols].to_json(orient="records", indent=2, force_ascii=False)
        prompt = f"""Tu es analyste e-commerce senior.

Données produits Top-K (JSON) :
{ctx}

{tone_line}

Tâche : produire un **résumé analytique** des produits ci-dessus (pas une simple liste).
- Patterns : gammes de prix, qualité perçue (notes/avis), catégories dominantes, plateformes.
- 2 à 4 recommandations actionnables pour la direction.
"""
        return client.chat(prompt, system=BASE_SYSTEM, max_tokens=1400)

    if kind == "market_trend":
        return generate_trend_report(client, df_scored, df_top)

    if kind == "pricing":
        lines = []
        if "platform" in df_scored.columns:
            plat = (
                df_scored.groupby("platform")["price"]
                .agg(["count", "mean", "median", "min", "max"])
                .round(2)
            )
            lines.append("Par plateforme :\n" + plat.to_string())
        if "category" in df_scored.columns:
            cat = (
                df_scored.groupby("category")["price"]
                .agg(["count", "mean", "median"])
                .sort_values("mean", ascending=False)
                .head(10)
                .round(2)
            )
            lines.append("Par catégorie (top 10 par prix moyen) :\n" + cat.to_string())
        if "discount_pct" in df_scored.columns:
            disc = (df_scored["discount_pct"].fillna(0) > 0).mean() * 100
            lines.append(f"Part des produits en promotion : {disc:.1f}%")
        blob = "\n\n".join(lines) if lines else "Statistiques agrégées indisponibles."
        prompt = f"""Tu es expert pricing e-commerce.

Statistiques calculées sur le catalogue :
{blob}

{tone_line}

Analyse **compétitive et pricing** : segments sous/sur-valorisés, effet plateforme, leviers prix/promo, risques — conclus par 3 actions concrètes."""
        return client.chat(prompt, system=BASE_SYSTEM, max_tokens=1200)

    if kind in ("cross_sell", "segment"):
        return recommend_strategy(client, load_clusters(), load_rules())

    avg_score = (
        round(float(df_scored["score"].mean()), 3)
        if "score" in df_scored.columns
        else 0.0
    )
    return client.chat(
        f"Type de rapport inconnu : {kind}. Résume les tendances générales du marché.\n\n"
        f"Statistiques rapides : {len(df_scored)} produits, score moyen {avg_score}.\n{tone_line}",
        system=BASE_SYSTEM,
        max_tokens=800,
    )


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